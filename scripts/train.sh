export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=4

# 核心修改：batch_size 降至极低，学习率减小
torchrun --nproc_per_node=4 run_train.py \
            +trainer.freeze_encoder=False \
            trainer.batch_size=20 \
            trainer.test_batch_size=4 \
            exp_name=1226-TRAIN-p40-4gpu-finetune-h36m-coco \
            trainer.max_lr=2e-5 \
            trainer.min_lr=1e-6 \
            trainer.total_epochs=30 \
            general.num_workers=8 \
            logger.logger_interval=50 \
            trainer.ema_decaybase=0.9995 \
            hub.skel_head.transformer_decoder.context_dim=1280 \
            trainer.ema_tau=4000 \
            policy.img_patch_size=256 \
            trainer.lamda=1. \
            misc.aux_loss_weight=0.1\
            +pretrained_ckpt="data_outputs/exp/1226-TRAIN-p40-4gpu-finetune-h36m-coco/checkpoints/last_step.pth"