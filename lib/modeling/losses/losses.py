"""
HSMR损失函数模块
实现2D/3D关键点损失和参数损失，用于HSMR模型训练
"""

import torch
import torch.nn as nn

from smplx.body_models import SMPLOutput
from lib.body_models.skel.skel_model import SKELOutput


class Keypoint2DLoss(nn.Module):
    """
    2D关键点损失模块
    计算预测的2D关键点与真实标注之间的损失，支持置信度加权
    """

    def __init__(self, loss_type: str = 'l1'):
        """
        初始化2D关键点损失
        
        Args:
            loss_type (str): 损失函数类型，可选'l1'或'l2'
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
        计算2D关键点重投影损失
        
        Args:
            pred_keypoints_2d (torch.Tensor): 预测的2D关键点，形状为 [B, N, 2]
                                             (B: 批次大小, N: 关键点数量)
            gt_keypoints_2d (torch.Tensor): 真实2D关键点和置信度，形状为 [B, N, 3]
                                           最后一维包含置信度信息
        Returns:
            torch.Tensor: 2D关键点损失标量值
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1,2))
        return loss.sum()


class Keypoint3DLoss(nn.Module):
    """
    3D关键点损失模块
    计算预测的3D关键点与真实标注之间的损失，使用骨盆对齐消除全局平移影响
    """

    def __init__(self, loss_type: str = 'l1'):
        """
        初始化3D关键点损失
        
        Args:
            loss_type (str): 损失函数类型，可选'l1'或'l2'
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
        计算3D关键点损失
        
        使用骨盆关节进行对齐，消除全局平移的影响
        
        Args:
            pred_keypoints_3d (torch.Tensor): 预测的3D关键点，形状为 [B, N, 3]
                                             (B: 批次大小, N: 关键点数量)
            gt_keypoints_3d (torch.Tensor): 真实3D关键点和置信度，形状为 [B, N, 4]
                                           最后一维包含置信度信息
            pelvis_id (int): 骨盆关节ID，用于对齐，默认为39
        Returns:
            torch.Tensor: 3D关键点损失标量值
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        # 使用骨盆关节进行对齐，消除全局平移影响
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        # 提取置信度并计算加权损失
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1,2))
        return loss.sum()


class ParameterLoss(nn.Module):
    """
    SKEL/SMPL参数损失模块
    计算预测的人体模型参数与真实标注之间的损失
    """

    def __init__(self):
        """
        初始化参数损失模块
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor):
        """
        计算SKEL/SMPL参数损失
        
        Args:
            pred_param (torch.Tensor): 预测的参数，形状为 [B, ...] 
                                     (身体姿态/全局方向/形状参数)
            gt_param (torch.Tensor): 真实的SKEL/SMPL参数，形状与pred_param相同
            has_param (torch.Tensor): 参数有效性掩码，形状为 [B]
        Returns:
            torch.Tensor: L2参数损失标量值
        """
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims-1)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = (has_param * self.loss_fn(pred_param, gt_param))
        return loss_param.sum()
