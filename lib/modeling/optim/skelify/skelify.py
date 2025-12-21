from lib.kits.basic import *
# 本文件实现了基于 SKEL 模型的人体参数优化器 SKELify：
# - 输入：2D 关键点（带置信度）、初始姿态/形状/相机平移、可选可视化图像
# - 过程：按配置的多阶段（phases）对 poses/betas/cam_t 进行梯度优化
# - 输出：优化后的 poses/betas/cam_t，以及用于质量判断的 2D 重投影误差

import cv2
import traceback
from tqdm import tqdm

from lib.body_models.common import make_SKEL
from lib.body_models.abstract_skeletons import Skeleton_OpenPose25
from lib.utils.vis import render_mesh_overlay_img
from lib.utils.data import to_tensor
from lib.utils.media import draw_kp2d_on_img, annotate_img, splice_img
from lib.utils.camera import perspective_projection

from .utils import (
    compute_rel_change,
    gmof,
)

from .closure import build_closure

class SKELify():

    def __init__(self, cfg, tb_logger=None, device='cuda:0', name='SKELify'):
        # cfg：Hydra 配置，包含优化阶段、学习率、可视化与日志频率等
        # tb_logger：TensorBoard 记录器（可为空）
        # device：计算设备
        # name：名称标识
        self.cfg = cfg
        self.name = name
        self.eq_thre = cfg.early_quit_thresholds

        self.tb_logger = tb_logger

        self.device = device
        # self.skel_model = make_SKEL(device=device)
        # 通过 Hydra 动态实例化 SKEL 模型（支持性别/关节回归器等可配置）
        self.skel_model = instantiate(cfg.skel_model).to(device)

        # Shortcuts.
        self.n_samples = cfg.logger.samples_per_record


    def __call__(
        self,
        gt_kp2d    : Union[torch.Tensor, np.ndarray],
        init_poses : Union[torch.Tensor, np.ndarray],
        init_betas : Union[torch.Tensor, np.ndarray],
        init_cam_t : Union[torch.Tensor, np.ndarray],
        img_patch  : Optional[np.ndarray] = None,
        **kwargs
    ):
        '''
        使用数值优化将 SKEL 参数拟合到 2D 关键点。

        ### 参数
        - gt_kp2d: (B, J, 3)，最后一维为 [x, y, conf]，坐标在 [-0.5, 0.5] 归一化空间
        - init_poses: (B, 46) 初始姿态（四元数或内部表示，前 3 维为全局朝向）
        - init_betas: (B, 10) 初始形状
        - init_cam_t: (B, 3) 初始相机平移
        - img_patch: (B, H, W, 3) 可选用于可视化的图像块（归一化框内尺寸），None 时以黑底可视化

        ### 返回
        - 字典：
            - poses: (B, 46) 优化后姿态
            - betas: (B, 10) 优化后形状
            - cam_t: (B, 3) 优化后相机平移
            - kp2d_err: (B,) 与 GT 2D 关键点的加权 L2 误差（置信度加权）
        '''
        self.init_v = None
        self.init_ct = None
        self.init_kp2d_err = None

        with PM.time_monitor('Input Preparation'):  # 输入准备时间监控
            # 将各输入转为目标设备上的 float tensor，并断开梯度
            gt_kp2d = to_tensor(gt_kp2d, device=self.device).detach().float().clone()  # (B, J, 3)
            init_poses = to_tensor(init_poses, device=self.device).detach().float().clone()  # (B, 46)
            init_betas = to_tensor(init_betas, device=self.device).detach().float().clone()  # (B, 10)
            init_cam_t = to_tensor(init_cam_t, device=self.device).detach().float().clone()  # (B, 3)
            # 拆分姿态为全局朝向与身体关节两部分，便于按阶段选择性优化
            inputs = {
                    'poses_orient' : init_poses[:, :3],  # (B, 3)
                    'poses_body'   : init_poses[:, 3:],  # (B, 43)
                    'betas'        : init_betas,         # (B, 10)
                    'cam_t'        : init_cam_t,         # (B, 3)
                }
            # 将像素焦距缩放到归一化图像坐标（[-0.5, 0.5] 对应 img_patch_size）
            focal_length = float(self.cfg.focal_length / self.cfg.img_patch_size)  # float

        # ⛩️ Optimization phases, controlled by config file.    优化阶段，由配置文件控制
        with PM.time_monitor('Optim') as tm:
            prev_steps = 0  # accumulate the steps are *supposed* to be done in the previous phases 在之前的阶段中累积的步骤
            n_phases = len(self.cfg.phases)  # 阶段数量
            for phase_id, phase_name in enumerate(self.cfg.phases):  # 遍历每个阶段
                phase_cfg = self.cfg.phases[phase_name]
                # 📦 Data preparation.
                optim_params = []  # 优化参数   
                for k in inputs.keys():  # 遍历输入参数
                    if k in phase_cfg.params_keys:
                        # 仅对当前阶段配置的参数开放梯度
                        inputs[k].requires_grad = True
                        optim_params.append(inputs[k])  # (B, D)
                    else:
                        inputs[k].requires_grad = False
                log_data = {}  # 日志数据
                tm.tick(f'Data preparation')

                # ⚙️ Optimization preparation.
                # 通过 Hydra 实例化优化器（如 Adam/LBFGS/SGD），并构建 loss-closure 
                optimizer = instantiate(phase_cfg.optimizer, optim_params, _recursive_=True)  # 实例化优化器
                closure = self._build_closure(  # 构建损失闭包
                        cfg=phase_cfg, optimizer=optimizer,  # basic 基础配置
                        inputs=inputs, focal_length=focal_length, gt_kp2d=gt_kp2d,  # data reference    数据参考
                        log_data=log_data,  # monitoring    监控 日志数据
                    )
                tm.tick(f'Optimizer * closure prepared.')  # 优化器和闭包准备完成

                # 🚀 Optimization loop. 优化循环
                with tqdm(range(phase_cfg.max_loop)) as bar:     # 遍历每个阶段
                    prev_loss = None
                    bar.set_description(f'[{phase_name}] Loss: ???')
                    for i in bar:
                        # 1. Main part of the optimization loop.    优化循环的主要部分
                        log_data.clear()  # 清空日志数据
                        curr_loss = optimizer.step(closure)

                        # 2. Log. 记录
                        if self.tb_logger is not None:  # 如果日志记录器不为空
                            # 可视化记录：只抽样前 n_samples 以控制显存与日志体量
                            log_data.update({
                                'img_patch' : img_patch[:self.n_samples] if img_patch is not None else None,  # 图像块
                                'gt_kp2d'   : gt_kp2d[:self.n_samples].detach().clone(),  # 真值2D关键点
                            })
                            self._tb_log(prev_steps + i, phase_name, log_data)  # 记录日志

                        # 3. The end of one optimization loop. 优化循环的结束
                        bar.set_description(f'[{phase_id+1}/{n_phases}] @ {phase_name} - Loss: {curr_loss:.4f}')  # 设置描述
                        # 早停：相对收敛或梯度绝对幅度很小
                        if self._can_early_quit(optim_params, prev_loss, curr_loss):  # 早停判断
                            break
                        prev_loss = curr_loss

                    prev_steps += phase_cfg.max_loop  # 累积步骤
                    tm.tick(f'{phase_name} finished.')  # 阶段完成

        with PM.time_monitor('Last Inference'):  # 最终推理时间监控
            # 将拆分的姿态合并，做一次最终前向与 2D 投影以评估误差
            poses = torch.cat([inputs['poses_orient'], inputs['poses_body']], dim=-1).detach().clone()  # (B, 46)   姿态
            betas = inputs['betas'].detach().clone()  # (B, 10)
            cam_t = inputs['cam_t'].detach().clone()  # (B, 3)
            skel_outputs = self.skel_model(poses=poses, betas=betas, skelmesh=False)  # (B, 44, 3)
            skel_outputs = self.skel_model(poses=poses, betas=betas, skelmesh=False)  # (B, 44, 3)
            optim_kp3d = skel_outputs.joints  # (B, 44, 3) 优化后的3D关键点
            # Evaluate the confidence of the results.
            focal_length_xy = np.ones((len(poses), 2)) * focal_length  # (B, 2) 焦距
            optim_kp2d = perspective_projection(
                    points       = optim_kp3d,
                    translation  = cam_t,
                    focal_length = to_tensor(focal_length_xy, device=self.device),
                )
            kp2d_err = SKELify.eval_kp2d_err(gt_kp2d, optim_kp2d)  # (B,) 2D关键点误差

        # ⛩️ Prepare the output data.
        outputs = {
                'poses'    : poses,     # (B, 46) 姿态
                'betas'    : betas,     # (B, 10) 形状
                'cam_t'    : cam_t,     # (B, 3)    
                'kp2d_err' : kp2d_err,  # (B,) 2D关键点误差
            }
        return outputs


    def _can_early_quit(self, opt_params, prev_loss, curr_loss):
        ''' 判断是否可以提前结束当前阶段优化：
        - 若未配置 early_quit，则不提前结束
        - 相对变化（前后损失相对变化）低于阈值则早停
        - 绝对变化（参数梯度最大值）低于阈值则早停
        '''
        if self.cfg.early_quit_thresholds is None:  # 如果早停阈值为空
            # Never early quit.
            return False

        # Relative change test. 相对变化测试
        if prev_loss is not None:
            loss_rel_change = compute_rel_change(prev_loss, curr_loss)  # 相对变化
            if loss_rel_change < self.cfg.early_quit_thresholds.rel:
                get_logger().info(f'Early quit due to relative change: {loss_rel_change} = rel({prev_loss}, {curr_loss})')  # 相对变化早停
                return True

        # Absolute change test. 绝对变化测试
        if all([
            torch.abs(param.grad.max()).item() < self.cfg.early_quit_thresholds.abs
            for param in opt_params if param.grad is not None
        ]):
            get_logger().info(f'Early quit due to absolute change.')  # 绝对变化早停
            return True

        return False


    def _build_closure(self, *args, **kwargs):
        # Using this way to hide the very details and simplify the code. 使用这种方式隐藏非常细节，简化代码
        return build_closure(self, *args, **kwargs)  # 构建损失闭包


    @staticmethod
    def eval_kp2d_err(gt_kp2d_with_conf:torch.Tensor, pd_kp2d:torch.Tensor):
        ''' Evaluate the mean 2D keypoints L2 error. The formula is: ∑(gt - pd)^2 * conf / ∑conf. '''
        assert len(gt_kp2d_with_conf.shape) == len(gt_kp2d_with_conf.shape), f'gt_kp2d_wi cccth_conf.shape={gt_kp2d_with_conf.shape}, pd_kp2d.shape={pd_kp2d.shape} but they should both be ((B,) J, D).'
        if len(gt_kp2d_with_conf.shape) == 2:
            gt_kp2d_with_conf, pd_kp2d = gt_kp2d_with_conf[None], pd_kp2d[None]
        assert len(gt_kp2d_with_conf.shape) == 3, f'gt_kp2d_with_conf.shape={gt_kp2d_with_conf.shape}, pd_kp2d.shape={pd_kp2d.shape} but they should both be ((B,) J, D).'
        B, J, _ = gt_kp2d_with_conf.shape
        assert gt_kp2d_with_conf.shape == (B, J, 3), f'gt_kp2d_with_conf.shape={gt_kp2d_with_conf.shape} but it should be ((B,) J, 3).'
        assert pd_kp2d.shape == (B, J, 2), f'pd_kp2d.shape={pd_kp2d.shape} but it should be ((B,) J, 2).'

        conf = gt_kp2d_with_conf[..., 2]  # (B, J)
        gt_kp2d = gt_kp2d_with_conf[..., :2]  # (B, J, 2)
        kp2d_err = torch.sum((gt_kp2d - pd_kp2d) ** 2, dim=-1) * conf  # (B, J)
        kp2d_err = kp2d_err.sum(dim=-1) / (torch.sum(conf, dim=-1) + 1e-6)  # (B,)
        return kp2d_err


    @rank_zero_only
    def _tb_log(self, step_cnt:int, phase_name:str, log_data:Dict, *args, **kwargs):
        ''' 将优化过程中的关键数据（loss、mesh 叠加、2D kp）写入 TensorBoard：
        - 仅在设定的日志间隔写入，避免过量 I/O
        - 首次记录时缓存初始 mesh/cam/kp2d_err 以便对比
        - 输出拼接图：raw / gt_kp2d / pd_kp2d(叠加mesh) / 仅mesh / init
        '''
        if step_cnt != 0 and (step_cnt + 1) % self.cfg.logger.interval_skelify != 0:
            return

        summary_writer = self.tb_logger.experiment

        # Save losses.
        for loss_name, loss_val in log_data['losses'].items():
            summary_writer.add_scalar(f'skelify/{loss_name}', loss_val, step_cnt)

        # 优化过程可视化（后续可做更优雅的封装）
        if log_data['img_patch'] is None:
            log_data['img_patch'] = [np.zeros((self.cfg.img_patch_size, self.cfg.img_patch_size, 3), dtype=np.uint8)] \
                                  * len(log_data['gt_kp2d'])

        if self.init_v is None:
            self.init_v = log_data['pd_verts']
            self.init_ct = log_data['cam_t']
            self.init_kp2d_err = log_data['kp2d_err']

        # 将结果 skin mesh 覆盖到原始图像上，并叠加关键点用于对比
        try:
            imgs_spliced = []
            for i, img_patch in enumerate(log_data['img_patch']):
                kp2d_err = log_data['kp2d_err'][i].item()

                img_with_init = render_mesh_overlay_img(
                        faces      = self.skel_model.skin_f,
                        verts      = self.init_v[i],
                        K4         = [self.cfg.focal_length, self.cfg.focal_length, 128, 128],
                        img        = img_patch,
                        Rt         = [torch.eye(3), self.init_ct[i]],
                        mesh_color = 'pink',
                    )
                img_with_init = annotate_img(img_with_init, 'init')
                img_with_init = annotate_img(img_with_init, f'Quality: {self.init_kp2d_err[i].item()*1000:.3f}/1e3', pos='tl')

                img_with_mesh = render_mesh_overlay_img(
                        faces      = self.skel_model.skin_f,
                        verts      = log_data['pd_verts'][i],
                        K4         = [self.cfg.focal_length, self.cfg.focal_length, 128, 128],
                        img        = img_patch,
                        Rt         = [torch.eye(3), log_data['cam_t'][i]],
                        mesh_color = 'pink',
                    )
                betas_max = log_data['optim_betas'][i].abs().max().item()
                img_patch_raw = annotate_img(img_patch, 'raw')

                # 将归一化坐标转换回像素坐标以便绘制
                log_data['gt_kp2d'][i][..., :2] = (log_data['gt_kp2d'][i][..., :2] + 0.5) * self.cfg.img_patch_size
                img_with_gt = annotate_img(img_patch, 'gt_kp2d')
                img_with_gt = draw_kp2d_on_img(
                        img_with_gt,
                        log_data['gt_kp2d'][i],
                        Skeleton_OpenPose25.bones,
                        Skeleton_OpenPose25.bone_colors,
                    )

                log_data['pd_kp2d'][i] = (log_data['pd_kp2d'][i] + 0.5) * self.cfg.img_patch_size
                img_with_pd = cv2.addWeighted(img_with_mesh, 0.7, img_patch, 0.3, 0)
                img_with_pd = draw_kp2d_on_img(
                        img_with_pd,
                        log_data['pd_kp2d'][i],
                        Skeleton_OpenPose25.bones,
                        Skeleton_OpenPose25.bone_colors,
                    )

                img_with_pd = annotate_img(img_with_pd, 'pd')
                img_with_pd = annotate_img(img_with_pd, f'Quality: {kp2d_err*1000:.3f}/1e3\nbetas_max: {betas_max:.3f}', pos='tl')
                img_with_mesh = annotate_img(img_with_mesh, f'Quality: {kp2d_err*1000:.3f}/1e3\nbetas_max: {betas_max:.3f}', pos='tl')
                img_with_mesh = annotate_img(img_with_mesh, 'pd_mesh')

                img_spliced = splice_img(
                        img_grids = [img_patch_raw, img_with_gt, img_with_pd, img_with_mesh, img_with_init],
                        grid_ids  = [[1, 2, 3, 4]],
                    )
                img_spliced = annotate_img(img_spliced, f'{phase_name}/{step_cnt}', pos=(32, 224))
                imgs_spliced.append(img_spliced)

            img_final = splice_img(imgs_spliced, grid_ids=[[i] for i in range(len(log_data['img_patch']))])

            img_final = to_tensor(img_final, device=None).permute(2, 0, 1)  # (3, H, W)
            summary_writer.add_image('skelify/visualization', img_final, step_cnt)
        except Exception as e:
            get_logger().error(f'Failed to visualize the optimization process: {e}')
            traceback.print_exc()