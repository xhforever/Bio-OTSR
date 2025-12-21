from lib.kits.basic import *

from concurrent.futures import ThreadPoolExecutor
from lightning_fabric.utilities.rank_zero import _get_rank

from lib.data.augmentation.skel import rot_skel_on_plane
from lib.utils.data import to_tensor, to_numpy
from lib.utils.camera import perspective_projection, estimate_camera_trans
from lib.utils.vis import Wis3D
from lib.modeling.optim.skelify.skelify import SKELify
from lib.body_models.common import make_SKEL

# 调试开关
DEBUG = False        # 是否启用调试模式（保存图像等额外信息）
DEBUG_ROUND = False  # 是否启用调试轮次（强制执行优化）


class SKELifySPIN(pl.Callback):
    '''
    SKELify SPIN优化回调类
    
    这个回调类实现了SPIN（Self-improving Pseudo-labeling for Iterative Training）机制，
    通过SKELify优化器来不断改善预测结果，并用改善后的结果作为伪标签来训练网络。
    
    数据概念说明：
    1. `gt` (Ground Truth): 静态真值数据
       - 从静态训练数据集加载的数据，可能是真实标签（如2D关键点）或伪标签（如SKEL参数）
       - 这些数据会在迭代过程中逐渐被更好的伪真值替换
       
    2. `opgt` (Old Pseudo-Ground Truth): 旧伪真值数据
       - 在静态数据集和动态数据集中选择的较好真值数据
       - 作为训练网络的标签使用
       
    3. `pd` (Predicted Results): 网络预测结果
       - 来自网络输出的预测结果，将被进一步优化
       - 优化后称为`res`(Results from optimization)
       
    4. `bpgt` (Better Pseudo Ground Truth): 更好的伪真值数据
       - 存储在文件和内存中的优化结果
       - 从静态真值、缓存的伪真值或预测优化数据中选择的最高质量数据
    '''

    # TODO: 目前只考虑使用kp2d来评估性能（因为不是所有数据都提供kp3d）
    # TODO: 但未来需要考虑kp3d，如果有的话就使用它

    def __init__(
        self,
        cfg     : DictConfig,
        skelify : DictConfig,
        **kwargs,
    ):
        '''
        初始化SKELify SPIN回调
        
        ### 参数说明
        - cfg: 配置字典，包含SPIN相关的所有配置参数
        - skelify: SKELify优化器的配置
        '''
        super().__init__()
        
        # ==================== 基本配置 ====================
        self.interval = cfg.interval                    # SPIN优化的执行间隔（每多少步执行一次）
        self.B = cfg.batch_size                         # 批次大小
        self.kb_pr = cfg.get('max_batches_per_round', None)  # 每轮SPIN优化的最大批次数
        self.better_pgt_fn = Path(cfg.better_pgt_fn)    # 更好伪真值数据的存储文件路径
        self.skip_warm_up_steps = cfg.skip_warm_up_steps # 跳过的预热步数（避免早期无意义的优化）
        self.update_better_pgt = cfg.update_better_pgt   # 是否更新更好的伪真值数据
        self.skelify_cfg = skelify                       # SKELify优化器配置

        # 用于判断结果是否有效的阈值（防止某些数据初始没有参数但被更新为坏参数）
        self.valid_betas_threshold = cfg.valid_betas_threshold

        # 初始化预测数据缓存字典
        self._init_pd_dict()

        # 更好的伪真值数据缓存（延迟初始化）
        self.better_pgt = None


    def on_train_batch_start(self, trainer, pl_module, raw_batch, batch_idx):
        '''
        训练批次开始时的回调处理
        
        主要功能：
        1. 延迟初始化更好的伪真值数据缓存
        2. 使用缓存的更好伪真值数据更新当前批次的标签
        '''
        # 延迟初始化更好的伪真值数据（支持分布式训练）
        if self.better_pgt is None:
            self._init_better_pgt()

        # GPU_monitor.snapshot('GPU-Mem-Before-Train-Before-SPIN-Update')
        device = pl_module.device
        batch = raw_batch['img_ds']

        # 如果不更新伪真值，则直接返回
        if not self.update_better_pgt:
            return

        # ==================== 1. 从批次数据中组合样本标识 ====================
        seq_key_list = batch['__key__']                      # 序列键列表
        batch_do_flip_list = batch['augm_args']['do_flip']   # 翻转增强参数列表
        # 构建唯一样本ID（区分原始和翻转版本）
        sample_uid_list = [
                f'{seq_key}_flip' if do_flip else f'{seq_key}_orig'
                for seq_key, do_flip in zip(seq_key_list, batch_do_flip_list)
            ]

        # ==================== 2. 使用更好的伪真值更新标签 ====================
        for i, sample_uid in enumerate(sample_uid_list):
            # 如果在缓存中找到了更好的伪真值，则更新当前批次的标签
            if sample_uid in self.better_pgt['poses'].keys():
                batch['raw_skel_params']['poses'][i] = to_tensor(self.better_pgt['poses'][sample_uid], device=device)  # (46,) 姿态参数
                batch['raw_skel_params']['betas'][i] = to_tensor(self.better_pgt['betas'][sample_uid], device=device)  # (10,) 形状参数
                batch['has_skel_params']['poses'][i] = self.better_pgt['has_poses'][sample_uid]  # 0 or 1，是否有姿态参数
                batch['has_skel_params']['betas'][i] = self.better_pgt['has_betas'][sample_uid]  # 0 or 1，是否有形状参数
                batch['updated_by_spin'][i] = True  # 标记该样本已被SPIN更新（用于检查）
                # get_logger().trace(f'Update the pseudo-gt for {sample_uid}.')

        # GPU_monitor.snapshot('GPU-Mem-Before-Train-After-SPIN-Update')


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        '''
        训练批次结束时的回调处理
        
        主要功能：
        1. 收集网络的预测结果
        2. 定期执行SKELify优化来改善预测结果
        3. 管理GPU内存
        '''
        # GPU_monitor.snapshot('GPU-Mem-After-Train-Before-SPIN-Update')

        # 由于网络在训练初期的预测可能远离真值，我们跳过一些步骤来避免无意义的优化
        if trainer.global_step > self.skip_warm_up_steps or DEBUG_ROUND:
            # 收集预测结果到缓存中
            self._save_pd(batch['img_ds'], outputs)

            # 按设定间隔执行SPIN优化
            if self.interval > 0 and trainer.global_step % self.interval == 0 or DEBUG_ROUND:
                # 清理GPU缓存以释放内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 执行SKELify优化过程
                with PM.time_monitor('SPIN'):
                    self._spin(trainer.logger, pl_module.device)
                
                # 优化完成后再次清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # GPU_monitor.snapshot('GPU-Mem-After-Train-After-SPIN-Update')
        # GPU_monitor.report_latest(k=4)


    def _init_pd_dict(self):
        '''
        初始化预测数据缓存字典
        
        每次SPIN优化前清理内存，使用numpy存储数值以节省GPU内存。
        '''
        self.cache = {
                # ==================== 样本标识信息 ====================
                'seq_key_list' : [],                    # 序列键列表，用于标识样本
                
                # ==================== 用于比较的旧伪真值数据 ====================
                'opgt_poses_list'     : [],             # 旧伪真值姿态参数列表
                'opgt_betas_list'     : [],             # 旧伪真值形状参数列表
                'opgt_has_poses_list' : [],             # 是否有旧伪真值姿态参数的标记
                'opgt_has_betas_list' : [],             # 是否有旧伪真值形状参数的标记
                
                # ==================== 用于优化和自迭代的数据 ====================
                'gt_kp2d_list'  : [],                  # 真值2D关键点列表
                'pd_poses_list' : [],                   # 预测姿态参数列表
                'pd_betas_list' : [],                   # 预测形状参数列表
                'pd_cam_t_list' : [],                   # 预测相机平移参数列表
                'do_flip_list'  : [],                   # 是否进行翻转增强的标记列表
                'rot_deg_list'  : [],                   # 旋转增强角度列表
                'do_extreme_crop_list': [],             # 是否进行极端裁剪的标记（如果是，则不更新伪真值）
                
                # ==================== 用于可视化的数据 ====================
                'img_patch': [],                        # 图像块列表（仅在调试模式下使用）
                
                # 注意：没有gt_cam_t_list，因为相机参数通常需要估计
            }


    def _format_pd(self):
        '''
        格式化预测数据缓存为numpy格式
        
        将缓存中的数据转换为numpy格式，并只保留最新的k个批次的数据以节省内存。
        '''
        if self.kb_pr is None:
            last_k = len(self.cache['seq_key_list'])
        else:
            last_k = self.kb_pr * self.B  # the latest k samples to be optimized.
        self.cache['seq_key_list']         = to_numpy(self.cache['seq_key_list'])[-last_k:]
        self.cache['opgt_poses_list']      = to_numpy(self.cache['opgt_poses_list'])[-last_k:]
        self.cache['opgt_betas_list']      = to_numpy(self.cache['opgt_betas_list'])[-last_k:]
        self.cache['opgt_has_poses_list']  = to_numpy(self.cache['opgt_has_poses_list'])[-last_k:]
        self.cache['opgt_has_betas_list']  = to_numpy(self.cache['opgt_has_betas_list'])[-last_k:]
        self.cache['gt_kp2d_list']         = to_numpy(self.cache['gt_kp2d_list'])[-last_k:]
        self.cache['pd_poses_list']        = to_numpy(self.cache['pd_poses_list'])[-last_k:]
        self.cache['pd_betas_list']        = to_numpy(self.cache['pd_betas_list'])[-last_k:]
        self.cache['pd_cam_t_list']        = to_numpy(self.cache['pd_cam_t_list'])[-last_k:]
        self.cache['do_flip_list']         = to_numpy(self.cache['do_flip_list'])[-last_k:]
        self.cache['rot_deg_list']         = to_numpy(self.cache['rot_deg_list'])[-last_k:]
        self.cache['do_extreme_crop_list'] = to_numpy(self.cache['do_extreme_crop_list'])[-last_k:]

        if DEBUG:
            self.cache['img_patch'] = to_numpy(self.cache['img_patch'])[-last_k:]


    def _save_pd(self, batch, outputs):
        '''
        保存预测结果和相关标签
        
        从网络输出中收集所有预测结果和相关的标签信息，存储到缓存中供后续优化使用。
        '''
        B = len(batch['__key__'])

        self.cache['seq_key_list'].extend(batch['__key__'])  # (NS,)

        self.cache['opgt_poses_list'].extend(to_numpy(batch['raw_skel_params']['poses']))  # (NS, 46)
        self.cache['opgt_betas_list'].extend(to_numpy(batch['raw_skel_params']['betas']))  # (NS, 10)
        self.cache['opgt_has_poses_list'].extend(to_numpy(batch['has_skel_params']['poses']))  # (NS,) 0 or 1
        self.cache['opgt_has_betas_list'].extend(to_numpy(batch['has_skel_params']['betas']))  # (NS,) 0 or 1
        self.cache['gt_kp2d_list'].extend(to_numpy(batch['kp2d']))  # (NS, 44, 3)

        self.cache['pd_poses_list'].extend(to_numpy(outputs['pd_params']['poses']))
        self.cache['pd_betas_list'].extend(to_numpy(outputs['pd_params']['betas']))
        self.cache['pd_cam_t_list'].extend(to_numpy(outputs['pd_cam_t']))
        self.cache['do_flip_list'].extend(to_numpy(batch['augm_args']['do_flip']))
        self.cache['rot_deg_list'].extend(to_numpy(batch['augm_args']['rot_deg']))
        self.cache['do_extreme_crop_list'].extend(to_numpy(batch['augm_args']['do_extreme_crop']))

        if DEBUG:
            img_patch = batch['img_patch'].clone().permute(0, 2, 3, 1)  # (NS, 256, 256, 3)
            mean = torch.tensor([0.485, 0.456, 0.406], device=img_patch.device).reshape(1, 1, 1, 3)
            std = torch.tensor([0.229, 0.224, 0.225], device=img_patch.device).reshape(1, 1, 1, 3)
            img_patch = 255 * (img_patch * std + mean)
            self.cache['img_patch'].extend(to_numpy(img_patch).astype(np.uint8))  # (NS, 256, 256, 3)


    def _init_better_pgt(self):
        '''
        初始化更好的伪真值数据缓存
        
        支持分布式数据并行（DDP）的初始化方式，每个进程维护独立的缓存文件。
        '''
        self.rank = _get_rank()
        get_logger().info(f'Initializing better pgt cache @ rank {self.rank}')

        if self.rank is not None:
            self.better_pgt_fn = Path(f'{self.better_pgt_fn}_r{self.rank}')
            get_logger().info(f'Redirecting better pgt cache to {self.better_pgt_fn}')

        if self.better_pgt_fn.exists():
            better_pgt_z = np.load(self.better_pgt_fn, allow_pickle=True)
            self.better_pgt = {k: better_pgt_z[k].item() for k in better_pgt_z.files}
        else:
            self.better_pgt = {'poses': {}, 'betas': {}, 'has_poses': {}, 'has_betas': {}}


    def _spin(self, tb_logger, device):
        '''
        执行SPIN优化过程
        
        这是核心的SPIN优化方法，包括：
        1. 准备优化数据
        2. 运行SKELify优化
        3. 评估和更新更好的伪真值
        4. 异步保存结果
        
        ### 参数
        - tb_logger: TensorBoard日志记录器
        - device: 计算设备（GPU/CPU）
        '''
        skelify : SKELify = instantiate(self.skelify_cfg, tb_logger=tb_logger, device=device, _recursive_=False)
        skel_model = skelify.skel_model

        self._format_pd()

        # 1. Make up the cache to run SKELify. 准备缓存以运行SKELify
        with PM.time_monitor('preparation'):
            sample_uid_list = [
                    f'{seq_key}_flip' if do_flip else f'{seq_key}_orig'
                    for seq_key, do_flip in zip(self.cache['seq_key_list'], self.cache['do_flip_list'])
                ]

            all_gt_kp2d    = self.cache['gt_kp2d_list']   # (NS, 44, 2)
            all_init_poses = self.cache['pd_poses_list']  # (NS, 46)
            all_init_betas = self.cache['pd_betas_list']  # (NS, 10)
            all_init_cam_t = self.cache['pd_cam_t_list']  # (NS, 3)
            all_do_extreme_crop = self.cache['do_extreme_crop_list']  # (NS,)
            all_res_poses = []
            all_res_betas = []
            all_res_cam_t = []
            all_res_kp2d_err = []  # the evaluation of the keypoints 2D error 关键点2D误差的评估

        # 2. Run SKELify optimization here to get better results. 在这里运行SKELify优化以获得更好的结果
        with PM.time_monitor('SKELify') as tm:  # SKELify优化时间监控
            get_logger().info(f'Start to run SKELify optimization. GPU-Mem: {torch.cuda.memory_allocated() / 1e9:.2f}G.')
            n_samples = len(self.cache['seq_key_list'])  # 样本数量
            n_round = (n_samples - 1) // self.B + 1
            get_logger().info(f'Running SKELify optimization for {n_samples} samples in {n_round} rounds.')
            for rid in range(n_round):
                sid = rid * self.B
                eid = min(sid + self.B, n_samples)

                gt_kp2d_with_conf = to_tensor(all_gt_kp2d[sid:eid], device=device)  # (B, 44, 3) 真值2D关键点
                init_poses = to_tensor(all_init_poses[sid:eid], device=device)  # (B, 46) 初始姿态参数
                init_betas = to_tensor(all_init_betas[sid:eid], device=device)  # (B, 10) 初始形状参数
                init_cam_t = to_tensor(all_init_cam_t[sid:eid], device=device)  # (B, 3) 初始相机平移参数

                # Run the SKELify optimization. 运行SKELify优化
                outputs = skelify(
                        gt_kp2d    = gt_kp2d_with_conf,
                        init_poses = init_poses,
                        init_betas = init_betas,
                        init_cam_t = init_cam_t,
                        img_patch  = self.cache['img_patch'][sid:eid] if DEBUG else None,
                    )

                # Store the results. 存储结果
                all_res_poses.extend(to_numpy(outputs['poses']))  # (~NS, 46)
                all_res_betas.extend(to_numpy(outputs['betas']))  # (~NS, 10)
                all_res_cam_t.extend(to_numpy(outputs['cam_t']))  # (~NS, 3)
                all_res_kp2d_err.extend(to_numpy(outputs['kp2d_err']))  # (~NS,)

                tm.tick(f'SKELify round {rid} finished.')

        # 3. Initialize the uninitialized better pseudo-gt with old ground truth. 初始化未初始化的更好伪真值与旧真值
        with PM.time_monitor('init_bpgt'):
            get_logger().info(f'Initializing bgbt. GPU-Mem: {torch.cuda.memory_allocated() / 1e9:.2f}G.')
            for i in range(n_samples):
                sample_uid = sample_uid_list[i]
                if sample_uid not in self.better_pgt.keys():
                    self.better_pgt['poses'][sample_uid] = self.cache['opgt_poses_list'][i]
                    self.better_pgt['betas'][sample_uid] = self.cache['opgt_betas_list'][i]
                    self.better_pgt['has_poses'][sample_uid] = self.cache['opgt_has_poses_list'][i]
                    self.better_pgt['has_betas'][sample_uid] = self.cache['opgt_has_betas_list'][i]

        # 4. Update the results. 更新结果
        with PM.time_monitor('upd_bpgt'):
            upd_cnt = 0  # Count the number of updated samples. 计数更新的样本数量
            get_logger().info(f'Update the results. GPU-Mem: {torch.cuda.memory_allocated() / 1e9:.2f}G.')
            for rid in range(n_round):
                torch.cuda.empty_cache()
                sid = rid * self.B
                eid = min(sid + self.B, n_samples)

                focal_length = np.ones(2) * 5000 / 256  # TODO: These data should be loaded from configuration files. 这些数据应该从配置文件中加载
                focal_length = focal_length.reshape(1, 2).repeat(eid - sid, 1)  # (B, 2)    
                gt_kp2d_with_conf = to_tensor(all_gt_kp2d[sid:eid], device=device)  # (B, 44, 3) 真值2D关键点
                rot_deg = to_tensor(self.cache['rot_deg_list'][sid:eid], device=device)  # (B,) 数据增强旋转角度

                # 4.1. Prepare the better pseudo-gt and the results. 准备更好的伪真值和结果
                res_betas  = to_tensor(all_res_betas[sid:eid], device=device)  # (B, 10) 形状参数
                res_poses_after_augm  = to_tensor(all_res_poses[sid:eid], device=device)  # (B, 46) 姿态参数
                res_poses_before_augm = rot_skel_on_plane(res_poses_after_augm, -rot_deg)  # recover the augmentation rotation 恢复数据增强旋转
                res_kp2d_err = to_tensor(all_res_kp2d_err[sid:eid], device=device)  # (B,) 关键点2D误差的评估
                cur_do_extreme_crop = all_do_extreme_crop[sid:eid]

                # 4.2. Evaluate the quality of the existing better pseudo-gt. 评估现有更好伪真值的质量
                uids = sample_uid_list[sid:eid]  # [sid ~ eid] -> sample_uids   
                bpgt_betas = to_tensor([self.better_pgt['betas'][uid] for uid in uids], device=device)  # (B, 10) 形状参数  
                bpgt_poses_before_augm = to_tensor([self.better_pgt['poses'][uid] for uid in uids], device=device)  # (B, 46) 姿态参数
                bpgt_poses_after_augm = rot_skel_on_plane(bpgt_poses_before_augm.clone(), rot_deg)  # recover the augmentation rotation 恢复数据增强旋转

                skel_outputs = skel_model(poses=bpgt_poses_after_augm, betas=bpgt_betas, skelmesh=False)    # 3D关键点
                bpgt_kp3d = skel_outputs.joints.detach()  # (B, 44, 3) 3D关键点
                bpgt_est_cam_t = estimate_camera_trans(
                        S            = bpgt_kp3d,
                        joints_2d    = gt_kp2d_with_conf.clone(),
                        focal_length = 5000,
                        img_size     = 256,
                    )  # estimate camera translation from inference 3D keypoints and GT 2D keypoints 从推理3D关键点和GT 2D关键点估计相机平移
                bpgt_reproj_kp2d = perspective_projection(
                        points       = to_tensor(bpgt_kp3d, device=device),
                        translation  = to_tensor(bpgt_est_cam_t, device=device),
                        focal_length = to_tensor(focal_length, device=device),
                    )   # 2D重投影
                bpgt_kp2d_err = SKELify.eval_kp2d_err(gt_kp2d_with_conf, bpgt_reproj_kp2d)  # (B, 44) 关键点2D误差的评估

                valid_betas_mask = res_betas.abs().max(dim=-1)[0] < self.valid_betas_threshold  # (B,) 形状参数有效性检查
                better_mask = res_kp2d_err < bpgt_kp2d_err  # (B,) 质量比较
                upd_mask = torch.logical_and(valid_betas_mask, better_mask)  # (B,) 综合判断
                upd_ids = torch.arange(eid-sid, device=device)[upd_mask]  # uids -> ids 获取更新索引

                # Update one by one. 逐个更新
                for upd_id in upd_ids:
                    # `uid` for dynamic dataset unique id, `id` for in-round batch data. 动态数据集唯一标识，批次数据索引
                    # Notes: id starts from zeros, it should be applied to [sid ~ eid] directly. 注意：id从零开始，应该直接应用于[sid ~ eid]
                    #        Either `all_res_poses[upd_id]` or `res_poses[upd_id - sid]` is wrong. 注意：id从零开始，应该直接应用于[sid ~ eid]
                    if cur_do_extreme_crop[upd_id]:
                        # Skip the extreme crop data. 跳过极端裁剪数据
                        continue
                    sample_uid = uids[upd_id]
                    self.better_pgt['poses'][sample_uid] = to_numpy(res_poses_before_augm[upd_id])
                    self.better_pgt['betas'][sample_uid] = to_numpy(res_betas[upd_id])
                    self.better_pgt['has_poses'][sample_uid] = 1.  # If updated, then must have. 如果更新，则必须有
                    self.better_pgt['has_betas'][sample_uid] = 1.  # If updated, then must have. 如果更新，则必须有
                    upd_cnt += 1

            get_logger().info(f'Update {upd_cnt} samples among all {n_samples} samples.')  # 更新X个样本/总Y个样本

        # 5. [Async] Save the results. 异步保存结果
        with PM.time_monitor('async_dumping'):
            # TODO: Use lock and other techniques to achieve a better submission system. 使用锁和其他技术实现更好的提交系统
            # TODO: We need to design a better way to solve the synchronization problem. 我们需要设计一个更好的方法来解决同步问题
            if hasattr(self, 'dump_thread'):
                self.dump_thread.result()  # Wait for the previous dump to finish. 等待之前的dump完成
            with ThreadPoolExecutor() as executor:
                self.dump_thread = executor.submit(lambda: np.savez(self.better_pgt_fn, **self.better_pgt))

        # 5. Clean up the memory. 清理内存
        del skelify, skel_model
        self._init_pd_dict()