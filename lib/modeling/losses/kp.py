"""
关键点损失函数模块
实现2D和3D关键点的L1损失计算，支持置信度加权
"""

from lib.kits.basic import *


def compute_kp3d_loss(gt_kp3d, pd_kp3d, ref_jid=25+14):
    """
    计算3D关键点损失
    
    使用骨盆关节作为参考点进行对齐，避免全局平移的影响
    
    Args:
        gt_kp3d (Tensor): 真实3D关键点，形状为 (B, J, 4)，最后一维为置信度
        pd_kp3d (Tensor): 预测3D关键点，形状为 (B, J, 3)
        ref_jid (int): 参考关节ID，默认为骨盆关节 (25+14=39)
        
    Returns:
        Tensor: 加权L1损失标量值
    """
    conf = gt_kp3d[:, :, 3:].clone()  # 提取置信度 (B, J, 1)
    # 使用参考关节对齐，消除全局平移影响
    gt_kp3d_a = gt_kp3d[:, :, :3] - gt_kp3d[:, [ref_jid], :3]  # 对齐后的真实关键点 (B, J, 3)
    pd_kp3d_a = pd_kp3d[:, :, :3] - pd_kp3d[:, [ref_jid], :3]  # 对齐后的预测关键点 (B, J, 3)
    # 计算置信度加权的L1损失
    kp3d_loss = conf * F.l1_loss(pd_kp3d_a, gt_kp3d_a, reduction='none')  # (B, J, 3)
    return kp3d_loss.sum()  # 返回标量损失


def compute_kp2d_loss(gt_kp2d, pd_kp2d):
    """
    计算2D关键点损失
    
    Args:
        gt_kp2d (Tensor): 真实2D关键点，形状为 (B, J, 3)，最后一维为置信度
        pd_kp2d (Tensor): 预测2D关键点，形状为 (B, J, 2)
        
    Returns:
        Tensor: 加权L1损失标量值
    """
    conf = gt_kp2d[:, :, 2:].clone()  # 提取置信度 (B, J, 1)
    # 计算置信度加权的L1损失
    kp2d_loss = conf * F.l1_loss(pd_kp2d, gt_kp2d[:, :, :2], reduction='none')  # (B, J, 2)
    return kp2d_loss.sum()  # 返回标量损失