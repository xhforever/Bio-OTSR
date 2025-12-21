# 骨骼模型变换工具模块
# 主要用于处理骨骼参数的各种表示形式之间的转换，包括旋转矩阵、欧拉角、连续表示等

from lib.kits.basic import *

import torch
import numpy as np
import torch.nn.functional as F

# 导入骨骼定义相关的常量和配置
from .definition import JOINTS_DEF, N_JOINTS, JID2DOF, JID2QIDS, DoF1_JIDS, DoF2_JIDS, DoF3_JIDS, DoF1_QIDS, DoF2_QIDS, DoF3_QIDS

from lib.utils.data import to_tensor
from lib.utils.geometry.rotation import (
    matrix_to_euler_angles,      # 旋转矩阵转欧拉角
    matrix_to_rotation_6d,       # 旋转矩阵转6D连续表示
    euler_angles_to_matrix,      # 欧拉角转旋转矩阵
    rotation_6d_to_matrix,       # 6D连续表示转旋转矩阵
)


# ====== 内部工具函数 ======


def axis2convention(axis:List):
    ''' 
    将轴向量转换为约定字符串表示
    [1,0,0] -> 'X', [0,1,0] -> 'Y', [0,0,1] -> 'Z' 
    '''
    if axis == [1, 0, 0]:
        return 'X'
    elif axis == [0, 1, 0]:
        return 'Y'
    elif axis == [0, 0, 1]:
        return 'Z'
    else:
        raise ValueError(f'不支持的轴向量: {axis}.')


def rotation_2d_to_angle(r2d:torch.Tensor):
    '''
    从2D旋转向量中提取单个角度，该向量是2x2旋转矩阵的第一列

    ### 参数
    - r2d: torch.Tensor
        - 形状 = (...B, 2)

    ### 返回值
    - 形状 = (...B,)
    '''
    cos, sin = r2d[..., [0]], -r2d[..., [1]]
    return torch.atan2(sin, cos)

# ====== 工具函数 ======


# SMPL到SKEL方向转换的翻转参数和约定
OS2S_FLIP = [-1, 1, 1]  # 方向翻转参数
OS2S_CONV = 'YZX'       # 欧拉角约定

def real_orient_mat2q(orient_mat:torch.Tensor) -> torch.Tensor:
    '''
    将SMPL的方向矩阵转换为SKEL的方向参数q
    
    SKEL使用的旋转矩阵与SMPL的方向矩阵不同。
    下面的旋转表示函数不能直接用于转换旋转矩阵。
    此函数用于将SMPL的方向矩阵转换为SKEL的方向q。
    但这真的重要吗？也许我们不应该将SMPL的方向与SKEL对齐，它们可以不同吗？

    ### 参数
    - orient_mat: torch.Tensor, 形状 = (..., 3, 3)

    ### 返回值
    - orient_q: torch.Tensor, 形状 = (..., 3)
    '''
    device = orient_mat.device
    flip = to_tensor(OS2S_FLIP, device=device)  # (3,)
    orient_ea = matrix_to_euler_angles(orient_mat.clone(), OS2S_CONV)  # (..., 3)
    orient_ea = orient_ea[..., [2, 1, 0]]  # 重新排列顺序
    orient_q = orient_ea * flip[None]
    return orient_q


def real_orient_q2mat(orient_q:torch.Tensor) -> torch.Tensor:
    '''
    将SKEL的方向参数q转换为SMPL的方向矩阵
    
    SKEL使用的旋转矩阵与SMPL的方向矩阵不同。
    下面的旋转表示函数不能直接用于转换旋转矩阵。
    此函数用于将SKEL的方向q转换为SMPL的方向矩阵。
    但这真的重要吗？也许我们不应该将SMPL的方向与SKEL对齐，它们可以不同吗？

    ### 参数
    - orient_q: torch.Tensor, 形状 = (..., 3)

    ### 返回值
    - orient_mat: torch.Tensor, 形状 = (..., 3, 3)
    '''
    device = orient_q.device
    flip = to_tensor(OS2S_FLIP, device=device)  # (3,)
    orient_ea = orient_q * flip[None]
    orient_ea = orient_ea[..., [2, 1, 0]]  # 重新排列顺序
    orient_mat = euler_angles_to_matrix(orient_ea, OS2S_CONV)
    return orient_mat


def flip_params_lr(poses:torch.Tensor) -> torch.Tensor:
    '''
    通过交换身体左右部分的参数来翻转骨骼，这对数据增强很有用。
    注意，"左右"是在身体面向z+方向时定义的，这对方向很重要。

    ### 参数
    - poses: torch.Tensor, 形状 = (B, L, 46) 或 (L, 46)

    ### 返回值
    - flipped_poses: torch.Tensor, 形状 = (B, L, 46) 或 (L, 46)
    '''
    assert len(poses.shape) in [2, 3] and poses.shape[-1] == 46, f'姿态形状应为 (B, L, 46) 或 (L, 46)，但得到 {poses.shape}。'

    # 1. 通过重新排列交换每对的值
    flipped_perm = [
             0,  1,  2,  # 骨盆
            10, 11, 12,  # 股骨-右 -> 股骨-左
            13,          # 胫骨-右 -> 胫骨-左
            14,          # 距骨-右 -> 距骨-左
            15,          # 跟骨-右 -> 跟骨-左
            16,          # 脚趾-右 -> 脚趾-左
             3,  4,  5,  # 股骨-左 -> 股骨-右
             6,          # 胫骨-左 -> 胫骨-右
             7,          # 距骨-左 -> 距骨-右
             8,          # 跟骨-左 -> 跟骨-右
             9,          # 脚趾-左 -> 脚趾-右
            17, 18, 19,  # 腰椎
            20, 21, 22,  # 胸椎
            23, 24, 25,  # 头部
            36, 37, 38,  # 肩胛骨-右 -> 肩胛骨-左
            39, 40, 41,  # 肱骨-右 -> 肱骨-左
            42,          # 尺骨-右 -> 尺骨-左
            43,          # 桡骨-右 -> 桡骨-左
            44, 45,      # 手-右 -> 手-左
            26, 27, 28,  # 肩胛骨-左 -> 肩胛骨-右
            29, 30, 31,  # 肱骨-左 -> 肱骨-右
            32,          # 尺骨-左 -> 尺骨-右
            33,          # 桡骨-左 -> 桡骨-右
            34, 35       # 手-左 -> 手-右
        ]

    flipped_poses = poses[..., flipped_perm]

    # 2. 通过应用-1来镜像旋转方向
    flipped_sign = [
             1, -1, -1,  # 骨盆
             1,  1,  1,  # 股骨-右'
             1,          # 胫骨-右'
             1,          # 距骨-右'
             1,          # 跟骨-右'
             1,          # 脚趾-右'
             1,  1,  1,  # 股骨-左'
             1,          # 胫骨-左'
             1,          # 距骨-左'
             1,          # 跟骨-左'
             1,          # 脚趾-左'
            -1,  1, -1,  # 腰椎
            -1,  1, -1,  # 胸椎
            -1,  1, -1,  # 头部
            -1, -1,  1,  # 肩胛骨-右'
            -1, -1,  1,  # 肱骨-右'
             1,          # 尺骨-右'
             1,          # 桡骨-右'
             1,  1,      # 手-右'
            -1, -1,  1,  # 肩胛骨-左'
            -1, -1,  1,  # 肱骨-左'
             1,          # 尺骨-左'
             1,          # 桡骨-左'
             1,  1       # 手-左'
        ]
    flipped_sign = torch.tensor(flipped_sign, dtype=poses.dtype, device=poses.device)  # (46,)
    flipped_poses = flipped_sign * flipped_poses

    return flipped_poses



# 已弃用的函数：绕Z轴旋转SKEL方向
# def rotate_orient_around_z(q, rot):
#     """
#     旋转SKEL方向
#     参数:
#         q (np.ndarray): SKEL风格的旋转表示 (3,)
#         rot (np.ndarray): 以度为单位的旋转角度
#     返回:
#         np.ndarray: 旋转后的轴角向量
#     """
#     import torch
#     from lib.body_models.skel.osim_rot import CustomJoint
#     # q转换为矩阵
#     root = CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, 1, 1])  # 骨盆
#     q = torch.from_numpy(q).unsqueeze(0)
#     q = q[:, [2, 1, 0]]
#     Rp = euler_angles_to_matrix(q, convention="YXZ")
#     # 绕z轴旋转
#     R = torch.Tensor([[np.deg2rad(-rot), 0, 0]])
#     R = axis_angle_to_matrix(R)
#     R = torch.matmul(R, Rp)
#     # 矩阵转换为q
#     q = matrix_to_euler_angles(R, convention="YXZ")
#     q = q[:, [2, 1, 0]]
#     q = q.numpy().squeeze()

#     return q.astype(np.float32)


def params_q2rot(params_q:Union[torch.Tensor, np.ndarray]):
    '''
    将类欧拉角的SKEL参数表示的所有部分转换为旋转矩阵

    ### 参数
    - params_q: Union[torch.Tensor, np.ndarray], 形状 = (...B, 46) 或 (...B, 46)

    ### 返回值
    - 形状 = (...B, 24, 9)  # 24个关节，每个关节有一个3x3矩阵，但对于某些关节，矩阵并非全部使用
    '''
    # 检查类型并统一为torch
    is_np = isinstance(params_q, np.ndarray)
    if is_np:
        params_q = torch.from_numpy(params_q)

    # 准备必要的变量
    Bs = params_q.shape[:-1]
    ident = torch.eye(3, dtype=params_q.dtype).to(params_q.device)  # (3, 3)
    params_rot = ident.repeat(*Bs, N_JOINTS, 1, 1)  # (...B, 24, 3, 3)

    # 分别处理每个关节。修改自`skel_model.py`但是静态版本
    sid = 0
    for jid in range(N_JOINTS):
        joint_obj = JOINTS_DEF[jid]
        eid = sid + joint_obj.nb_dof.item()
        params_rot[..., jid, :, :] = joint_obj.q_to_rot(params_q[..., sid:eid])
        sid = eid

    if is_np:
        params_rot = params_rot.detach().cpu().numpy()
    return params_rot


def params_q2rep(params_q:Union[torch.Tensor, np.ndarray]):
    '''
    将类欧拉角的SKEL参数表示转换为连续表示。
    此函数不应在训练过程中使用，仅用于调试。
    实际重要的函数是此函数的逆函数。

    ### 参数
    - params_q: Union[torch.Tensor, np.ndarray], 形状 = (...B, 46) 或 (...B, 46)

    ### 返回值
    - 形状 = (...B, 24, 6)
        - 在24个关节中，对于3自由度的关节，使用所有6个值来表示旋转；
          但对于1自由度关节，仅使用前2个值。其余将表示为零。
    '''
    # 检查类型并统一为torch
    is_np = isinstance(params_q, np.ndarray)
    if is_np:
        params_q = torch.from_numpy(params_q)

    # 准备必要的变量
    Bs = params_q.shape[:-1]
    params_rep = params_q.new_zeros(*Bs, N_JOINTS, 6)  # (...B, 24, 6)

    # 分别处理每个关节。修改自`skel_model.py`但是静态版本
    sid = 0
    for jid in range(N_JOINTS):
        joint_obj = JOINTS_DEF[jid]
        dof = joint_obj.nb_dof.item()
        eid = sid + dof
        if dof == 3:
            mat = joint_obj.q_to_rot(params_q[..., sid:eid])  # (...B, 3, 3)
            params_rep[..., jid, :] = matrix_to_rotation_6d(mat)  # (...B, 6)
        elif dof == 2:
            # mat = joint_obj.q_to_rot(params_q[..., sid:eid])  # (...B, 3, 3)
            # params_rep[..., jid, :] = matrix_to_rotation_6d(mat)  # (...B, 6)
            params_rep[..., jid, :2] = params_q[..., sid:eid]
        elif dof == 1:
            cos = torch.cos(params_q[..., sid])
            sin = torch.sin(params_q[..., sid])
            params_rep[..., jid, 0] = cos
            params_rep[..., jid, 1] = -sin

        sid = eid

    if is_np:
        params_rep = params_rep.detach().cpu().numpy()
    return params_rep


# 已弃用
def dof3_to_q(rot, axises:List, flip:List):
    '''
    将旋转矩阵转换为SKEL风格的旋转表示

    ### 参数
    - rot: torch.Tensor, 形状 (...B, 3, 3)
        - 旋转矩阵
    - axises: list
        - [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2]]
        - 在SKEL的joint_dict中定义的轴。xi, yi, zi中只有一个是1，其他是0
    - flip: list
        - [f0, f1, f2]
        - 在SKEL的joint_dict中定义的翻转值。fi是1或-1

    ### 返回值
    - 形状 = (...B, 3)
    '''
    convention = [axis2convention(axis) for axis in reversed(axises)]  # SKEL使用反向顺序的欧拉角
    convention = ''.join(convention)
    q = matrix_to_euler_angles(rot[..., :, :], convention=convention)  # (...B, 3)
    q = q[..., [2, 1, 0]]  # SKEL使用反向顺序的欧拉角
    flip = rot.new_tensor(flip)  # (3,)
    q = flip * q
    return q


### 慢版本，已弃用 ###
# def params_rep2q(params_rot:Union[torch.Tensor, np.ndarray]):
#     '''
#     将连续表示转换回SKEL风格的类欧拉角表示
#
#     ### 参数
#     - params_rot: Union[torch.Tensor, np.ndarray]
#         - 形状 = (...B, 24, 6)
#
#     ### 返回值
#     - 形状 = (...B, 46)
#     '''
#
#     # 检查类型并统一为torch
#     is_np = isinstance(params_rot, np.ndarray)
#     if is_np:
#         params_rot = torch.from_numpy(params_rot)
#
#     # 准备必要的变量
#     Bs = params_rot.shape[:-2]
#     params_q = params_rot.new_zeros((*Bs, 46))  # (...B, 46)
#
#     for jid in range(N_JOINTS):
#         joint_obj = JOINTS_DEF[jid]
#         dof = joint_obj.nb_dof.item()
#         sid, eid = JID2QIDS[jid][0], JID2QIDS[jid][-1] + 1
#
#         if dof == 3:
#             mat = rotation_6d_to_matrix(params_rot[..., jid, :])  # (...B, 3, 3)
#             params_q[..., sid:eid] = dof3_to_q(
#                     mat,
#                     joint_obj.axis.tolist(),
#                     joint_obj.axis_flip.detach().cpu().tolist(),
#                 )
#         elif dof == 2:
#             params_q[..., sid:eid] = params_rot[..., jid, :2]
#         else:
#             params_q[..., sid:eid] = rotation_2d_to_angle(params_rot[..., jid, :2])
#
#     if is_np:
#         params_q = params_q.detach().cpu().numpy()
#     return params_q

def orient_mat2q(orient_mat:torch.Tensor):
    ''' 这是一个仅用于检查的工具函数。orient_mat ~ (...B, 3, 3)'''
    poses_rep = orient_mat.new_zeros(orient_mat.shape[:-2] + (24, 6))  # (...B, 24, 6)
    orient_rep = matrix_to_rotation_6d(orient_mat)  # (...B, 6)
    poses_rep[..., 0, :] = orient_rep
    poses_q = params_rep2q(poses_rep)  # (...B, 46)
    return poses_q[..., :3]


# 为不同约定预分组的关节
CON_GROUP2JIDS  = {'YXZ': [0, 1, 6], 'YZX': [11, 12, 13], 'XZY': [14, 19], 'ZYX': [15, 20]}
CON_GROUP2FLIPS = {'YXZ': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, -1.0, -1.0]], 'YZX': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 'XZY': [[1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], 'ZYX': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]}

# 快速版本
def params_rep2q(params_rot:Union[torch.Tensor, np.ndarray]):
    '''
    将连续表示转换回SKEL风格的类欧拉角表示

    ### 参数
    - params_rot: Union[torch.Tensor, np.ndarray], 形状 = (...B, 24, 6)

    ### 返回值
    - 形状 = (...B, 46)
    '''

    with PM.time_monitor('params_rep2q'):
        with PM.time_monitor('preprocess'):
            params_rot, recover_type_back = to_tensor(params_rot, device=None, temporary=True)

            # 准备必要的变量
            Bs = params_rot.shape[:-2]
            params_q = params_rot.new_zeros((*Bs, 46))  # (...B, 46)

        with PM.time_monitor(f'dof1&dof2'):
            params_q[..., DoF1_QIDS] = rotation_2d_to_angle(params_rot[..., DoF1_JIDS, :2]).squeeze(-1)
            params_q[..., DoF2_QIDS] = params_rot[..., DoF2_JIDS, :2].reshape(*Bs, -1)  # (...B, J2=2 * 2)

        with PM.time_monitor(f'dof3'):
            dof3_6ds = params_rot[..., DoF3_JIDS, :].reshape(*Bs, len(DoF3_JIDS), 6)  # (...B, J3=10, 3, 6)
            dof3_mats = rotation_6d_to_matrix(dof3_6ds)  # (...B, J3=10, 3, 3)

            for convention, jids in CON_GROUP2JIDS.items():
                idxs = [DoF3_JIDS.index(jid) for jid in jids]
                mats = dof3_mats[..., idxs, :, :]  # (...B, J', 3, 3)
                qs = matrix_to_euler_angles(mats, convention=convention)  # (...B, J', 3)
                qs = qs[..., [2, 1, 0]]  # SKEL使用反向顺序的欧拉角
                flips = qs.new_tensor(CON_GROUP2FLIPS[convention])  # (J', 3)
                qs = qs * flips  # (...B, J', 3)
                qids = [qid for jid in jids for qid in JID2QIDS[jid]]
                params_q[..., qids] = qs.reshape(*Bs, -1)

        with PM.time_monitor('post_process'):
            params_q = recover_type_back(params_q)
    return params_q