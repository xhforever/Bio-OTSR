from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F  # [Added] 引入 functional 模块
from body_models.skel.skel_model import SKELOutput
from body_models.skel_utils.transforms import params_q2rep
from util.misc import trans_points2d_parallel
from util.geometry import axis_angle_to_matrix, pose_params_to_rot

from util.pylogger import get_pylogger


logger = get_pylogger(__name__)

class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        # batch_size = conf.shape[0] # unused
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1,2))
        return loss.sum()
    

class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 39):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        # batch_size = pred_keypoints_3d.shape[0] # unused
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1,2))
        return loss.sum()


class ParameterLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        SKEL parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')


    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor):
        """
        Compute SKEL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth SMPL parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        loss_param = self.loss_fn(pred_param, gt_param)
        return loss_param.sum()
 


class HPE_Loss(nn.Module):
    """
    Human loss module that combines 2D keypoint loss, 3D keypoint loss, SKEL parameter loss,
    and Bio-OTSR specific losses (Swing, Twist, Scalar).
    """

    def __init__(self, cfg):
        super(HPE_Loss, self).__init__()
        self.cfg = cfg 
        # using 2DLossScaled
        loss_type = 'l1'
        self.kp2d_loss = Keypoint2DLoss(loss_type=loss_type)
        self.kp3d_loss = Keypoint3DLoss(loss_type=loss_type)
        self.param_loss = ParameterLoss(loss_type=loss_type)
        
        # [Added] Bio-OTSR Loss Weights
        # 优先从配置中读取，如果没有则使用默认值
        self.w_swing = getattr(cfg.loss_weight, 'w_swing', 10.0)
        self.w_twist = getattr(cfg.loss_weight, 'w_twist', 1.0)
        self.w_scalar = getattr(cfg.loss_weight, 'w_scalar', 1.0)
        
        logger.info(f'using the loss type: {loss_type} for 2D and 3D keypoint loss, HPE Loss init!')
        logger.info(f'Bio-OTSR Weights - Swing: {self.w_swing}, Twist: {self.w_twist}, Scalar: {self.w_scalar}')

    def forward(self, output, target, _trans) -> Dict:
        """
        Compute the total HPE loss.
        Args:
            output (Dict): Model outputs containing predicted keypoints, parameters, and Bio-OTSR intermediates (raw_kp3d, raw_ortho).
            target (Dict): Ground truth data containing target keypoints and parameters.
            _trans: transformation parameters for 2D cropping.
        Returns:
            Dict: Loss dictionary.
        """
        
        B = target['kp3d'].shape[0]

        # -----------------------------------------------------------
        # 1. Standard End-to-End Losses (based on Final Mesh/Pose)
        # -----------------------------------------------------------
        
        pd_poses_mat, _ = pose_params_to_rot(output['poses'])  # (B, J=24, 3, 3)
        pd_poses = pd_poses_mat.reshape(-1, 24*3*3)  # (B, 24*3*3)

        gt_poses_mat, _ = pose_params_to_rot(target['poses'])  # (B, J=24, 3, 3)
        gt_poses = gt_poses_mat.reshape(-1, 24*3*3)  # (B, 24*3*3)
        
        # Get Cropped gt keypoints2d
        gt_kp2d = target['kp2d']
        pd_kp2d = output['kp2d']
        pd_kp2d_cropped = trans_points2d_parallel(pd_kp2d, _trans)
        pd_kp2d_cropped = pd_kp2d_cropped / self.cfg.policy.img_patch_size - 0.5
        
        # Calculate standard losses
        loss_kp2d = self.kp2d_loss(pd_kp2d_cropped, gt_kp2d) / B
        loss_kp3d = self.kp3d_loss(output['kp3d'], target['kp3d']) / B 
        loss_betas = self.param_loss(output['betas'], target['betas']) / B 
        loss_poses = self.param_loss(pd_poses[:, 9:], gt_poses[:, 9:]) / B 
        loss_orient = self.param_loss(pd_poses[:, :9], gt_poses[:, :9]) / B
        
        loss_dict = {
            'kp3d': loss_kp3d,
            'kp2d': loss_kp2d,
            'betas': loss_betas,
            'poses_body': loss_poses,
            'poses_orient': loss_orient,
        }

        # -----------------------------------------------------------
        # 2. [Added] Bio-OTSR Geometric Losses (Intermediate Supervision)
        # -----------------------------------------------------------
        
        # A. Swing Loss (L_swing): 监督 Swing Head 预测的 3D 关节位置
        # 用于帮助网络确定骨骼主轴方向
        if 'raw_kp3d' in output:
            # 假设 target 中包含 'kp3d' 或专门的 'skel_joints' (Root-Relative)
            # 这里复用 target['kp3d'] 的前三维 (去除置信度)，注意维度匹配
            # target['kp3d'] shape: [B, S, N, 4] -> 取 [:, 0, :, :3] 假设 S=1
            gt_kp3d_raw = target['kp3d'][:, 0, :, :3]
            
            # 确保预测值维度一致
            pred_kp3d_raw = output['raw_kp3d']
            
            # 计算 L1 Loss
            loss_swing = F.l1_loss(pred_kp3d_raw, gt_kp3d_raw)
            loss_dict['loss_swing'] = loss_swing * self.w_swing

        # B. Twist Loss (L_twist): 监督 Twist Head 预测的骨骼自旋方向向量
        # 需要 dataset 提供 'ortho_gt'
        if 'raw_ortho' in output and 'ortho_gt' in target:
            pred_ortho = output['raw_ortho'] # (B, 6, 3)
            gt_ortho = target['ortho_gt']    # (B, 6, 3)
            
            # Cosine Embedding Loss: 1 - cos(pred, gt)
            loss_twist = 1.0 - F.cosine_similarity(pred_ortho, gt_ortho, dim=-1).mean()
            loss_dict['loss_twist'] = loss_twist * self.w_twist

        # C. Scalar Loss (L_scalar): 监督标量参数 (Type B & D)
        # 需要 dataset 提供 'scalar_gt'
        if 'raw_scalar' in output and 'scalar_gt' in target:
            loss_scalar = F.mse_loss(output['raw_scalar'], target['scalar_gt'])
            loss_dict['loss_scalar'] = loss_scalar * self.w_scalar

        return loss_dict