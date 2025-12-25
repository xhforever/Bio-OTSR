# -*- coding: utf-8 -*-
import os
import random
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import hydra
import numpy as np
import torch
from datasets import build_dataset
from datasets.moyo_datasets import EvalMoyoDataset
from engine import evaluate_hmr2, evaluate_moyo, train_one_epoch
from models.ema import EmaModel
from models.skelvit import build_model
from run_hsmr_test import get_data
import util.misc as utils
from util.pylogger import get_pylogger 
from torch.utils.data import DistributedSampler, DataLoader

@hydra.main(version_base='1.2', config_path="config", config_name="train.yaml")
def main(cfg):

    utils.init_distributed_mode(cfg)
    
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger = get_pylogger(__name__)

    if utils.get_rank() == 0:
        writter = SummaryWriter(output_dir / 'tensorboards')
        writter_ema = SummaryWriter(output_dir / 'tensorboards/ema')
    else:
        writter = None
        writter_ema = None
        
    device = torch.device(f'cuda:{cfg.gpu}')
    seed = cfg.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model, criterion = build_model(cfg)
    model.to(device)

    model_without_ddp = model
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], gradient_as_bucket_view=True)
        model_without_ddp = model.module

    ema_model = EmaModel(cfg, model_without_ddp)

    # ================= [加载预训练权重逻辑] =================
    _checkpoint = None
    if cfg.get('pretrained_ckpt', None):
        checkpoint_path = cfg.pretrained_ckpt
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading pretrained checkpoint from {checkpoint_path}")
            _checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 1. 加载模型权重
            if 'model' in _checkpoint:
                msg = model_without_ddp.load_state_dict(_checkpoint['model'], strict=False)
                logger.info(f"Model load result: {msg}")
            else:
                msg = model_without_ddp.load_state_dict(_checkpoint, strict=False)
                logger.info(f"Model load result: {msg}")
                
            # 2. 加载 EMA 模型 (如果存在)
            if 'ema_model' in _checkpoint and hasattr(ema_model, 'ema'):
                 try:
                     ema_model.ema.load_state_dict(_checkpoint['ema_model'], strict=False)
                     logger.info("EMA model loaded.")
                 except Exception as e:
                     logger.warning(f"EMA load warning: {e}")
        else:
            logger.warning(f"Checkpoint {checkpoint_path} does not exist!")
    # ============================================================   

    n_parameters = sum(p.numel() for p in model_without_ddp.get_trainable_parameters() if p.requires_grad) / 1_000_000
    logger.info(f'number of params: {n_parameters} M')
    
    dataset_train = build_dataset(cfg=cfg, image_set='train')

    if cfg.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, cfg.trainer.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, 
                                    batch_sampler=batch_sampler_train, 
                                    num_workers=cfg.general.num_workers,
                                    pin_memory=True,
                                    persistent_workers=True if cfg.general.num_workers > 0 else False,
                                    prefetch_factor=2)
   
    # load evaluation data     
    ds_conf = cfg.eval_list.datasets
    data_list = []
    ds_list = ds_conf.split('_')
    for ds_name in ds_list:
        data = get_data(ds_name, cfg)
        data_list.append(data)

    npz_file = Path(cfg.paths.data_inputs) / 'skel-evaluation-labels' / 'moyo_hard.npz'
    moyo_dataset = EvalMoyoDataset(cfg, npz_fn=npz_file, ignore_img=False)
    moyo_data_loader = DataLoader(moyo_dataset, cfg.trainer.test_batch_size, drop_last=False, num_workers=cfg.general.num_workers)

    param_dicts = [
        {
            "params": [p for p in model_without_ddp.get_trainable_parameters() if p.requires_grad]
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.trainer.max_lr, weight_decay=cfg.trainer.weight_decay)

    warmup_steps = cfg.trainer.warmup_epochs * len(data_loader_train)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        else:
            return 1.0
        
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ================= [恢复训练状态 (Optimizer/Epoch)] =================
    start_epoch = 0
    best_pve = float('inf')
    
    # 只有当加载了 Checkpoint 且配置允许 Resume 时才恢复状态
    if _checkpoint is not None and cfg.get('resume_training', True):
        # 1. 恢复 Optimizer
        if 'optimizer' in _checkpoint:
            try:
                optimizer.load_state_dict(_checkpoint['optimizer'])
                logger.info("Optimizer state loaded.")
            except Exception as e:
                logger.warning(f"Optimizer load failed (ok if finetuning): {e}")
        
        # 2. 恢复 Scheduler
        if 'lr_scheduler' in _checkpoint:
            try:
                lr_scheduler.load_state_dict(_checkpoint['lr_scheduler'])
                logger.info("LR scheduler state loaded.")
            except Exception as e:
                logger.warning(f"Scheduler load failed: {e}")
        
        # 3. 恢复 Epoch
        if 'epoch' in _checkpoint:
            # [Modified] 如果是中途断掉的 checkpoint，建议重跑当前 epoch，防止数据跳过
            # 如果您确定 checkpoint 是 epoch 结束时保存的，可以用 +1
            # 这里保守起见，直接用 saved_epoch，意味着可能会重复跑一部分数据，但比丢失数据好
            start_epoch = _checkpoint['epoch']
            logger.info(f"Resuming from epoch {start_epoch}")
            
    # ====================================================================
    
    logger.info("Start training")
    for epoch in range(start_epoch, cfg.trainer.total_epochs):
        
        if cfg.distributed:
            sampler_train.set_epoch(epoch)
        
        # [Fix] 删除了 global_step_offset 参数，因为 engine.py 中未定义
        train_one_epoch(
            cfg=cfg, model=model, ema_model=ema_model, 
            criterion=criterion, data_loader=data_loader_train, optimizer=optimizer, 
            lr_scheduler=lr_scheduler, device=device, writter=writter, epoch=epoch
        )
        
        # Evaluation Loop
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == cfg.trainer.total_epochs - 1:
             # COCO
            evaluate_hmr2(
                cfg=cfg, model=model_without_ddp, data_list=data_list
            )
            # MOYO-HARD
            mpjpe, pampjpe, pve, _= evaluate_moyo(cfg, model_without_ddp, moyo_data_loader, device, writter, epoch, ema=False)
            
            # COCO EMA
            evaluate_hmr2(
                cfg=cfg, model=ema_model.model, data_list=data_list
            )
            # MOYO-HARD EMA
            ema_mpjpe, ema_pampjpe, ema_pve, _ =  evaluate_moyo(cfg, ema_model.model, moyo_data_loader, device, writter_ema, epoch, ema=True)

            if cfg.output_dir:
                # Save Best
                if ema_pve < best_pve:
                    best_pve = ema_pve
                    ckpt_path = output_dir / 'checkpoints/best.pth'
                    logger.info(f'BEST epoch {epoch}: PVE->{pve:.4f} | EMA PVE->{ema_pve:.4f}')
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'ema_model': ema_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'cfg': cfg,
                    }, ckpt_path)

if __name__ == '__main__':
    main()