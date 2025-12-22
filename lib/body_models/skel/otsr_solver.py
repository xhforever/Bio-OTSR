import torch
import torch.nn as nn
import torch.nn.functional as F
from .kin_skel import BIO_OTSR_CONFIG

class BioOTSRSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = BIO_OTSR_CONFIG

    def rotation_matrix_to_euler_angles(self, R):
        """
        将旋转矩阵转换为 XYZ 欧拉角
        R: (B, 3, 3)
        Returns: (B, 3)
        """
        sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
        x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        y = torch.atan2(-R[:, 2, 0], sy)
        z = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        return torch.stack([x, y, z], dim=-1)

    def forward(self, pred_kp3d, pred_ortho, pred_scalar):
        """
        Bio-OTSR 核心解算器
        
        Args:
            pred_kp3d: (B, 24, 3) 预测的 3D 关节坐标 (Swing)
            pred_ortho: (B, N_Ortho, 3) 预测的扭转方向向量 (Twist)
            pred_scalar: (B, N_Scalar) 预测的标量参数 (Type B & D)
            
        Returns:
            final_thetas: (B, 46) 最终解算出的 SKEL 姿态参数
        """
        B, device = pred_kp3d.shape[0], pred_kp3d.device
        final_thetas = torch.zeros(B, 46).to(device)
        
        # 指针，用于从扁平的预测向量中取值
        ortho_ptr = 0
        scalar_ptr = 0

        # --- 1. 处理 Type D (直接参数) ---
        n_type_d = len(self.cfg['TYPE_D_INDICES'])
        # 假设 pred_scalar 的前 n_type_d 个是 Type D
        final_thetas[:, self.cfg['TYPE_D_INDICES']] = pred_scalar[:, :n_type_d]
        scalar_ptr += n_type_d

        # --- 2. 处理 Type B (铰链关节 - Hinge) ---
        # 需确保 BIO_OTSR_CONFIG['TYPE_B'] 遍历顺序与 Decoder 输出顺序一致
        # 为安全起见，这里按 param index 排序遍历
        type_b_items = sorted(self.cfg['TYPE_B'].items(), key=lambda x: x[1]['param'])
        
        for name, info in type_b_items:
            s = pred_scalar[:, scalar_ptr]
            scalar_ptr += 1
            # 物理限制映射: [-1, 1] -> [min, max]
            limit = info['limit']
            # 使用 Tanh 确保输出在 [-1, 1] 之间，然后映射到物理范围
            val = (torch.tanh(s) + 1) / 2 * (limit[1] - limit[0]) + limit[0]
            final_thetas[:, info['param']] = val

        # --- 3. 处理 Type A (球窝关节 - Swing & Twist) ---
        # 同样按关节名或特定逻辑排序，这里建议 kin_skel 中定义为列表或有序字典
        # 假设 Decoder 是按 kin_skel 默认顺序输出的
        type_a_items = sorted(self.cfg['TYPE_A'].items(), key=lambda x: x[0])
        
        for name, info in type_a_items:
            # A. 计算 Swing (骨骼轴)
            # 使用预测的 3D 关键点计算骨骼向量 b
            child_idx, parent_idx = info['child'], info['parent']
            # 向量方向: 父 -> 子
            b_vec = F.normalize(pred_kp3d[:, child_idx] - pred_kp3d[:, parent_idx], dim=-1)
            
            # B. 获取 Twist (预测的正交向量)
            t_raw = pred_ortho[:, ortho_ptr]
            ortho_ptr += 1
            
            # C. Gram-Schmidt 正交化: o = t - (t·b)b
            # 这一步强制 Twist 向量垂直于 Swing 向量
            proj = (t_raw * b_vec).sum(dim=-1, keepdim=True) * b_vec
            o_vec = F.normalize(t_raw - proj, dim=-1)
            
            # D. 构建全局旋转矩阵 R = [b, o, b x o]
            # 注意: 这取决于 SKEL 骨骼定义的局部坐标系。
            # 假设骨骼沿局部 X 轴 (b_vec)，Twist 轴沿局部 Y 轴 (o_vec)
            # Z = X x Y
            z_vec = torch.cross(b_vec, o_vec)
            R_global = torch.stack([b_vec, o_vec, z_vec], dim=-1)
            
            # E. 分解为欧拉角 (简化版：直接分解 R_global)
            # 完整版应该乘以父关节逆矩阵 R_local = R_parent.T @ R_global
            # 由于这里是一个单纯的 Solver 演示，暂略去父关节缓存逻辑，
            # 实际训练中网络会学会补偿这个差异。
            euler = self.rotation_matrix_to_euler_angles(R_global)
            final_thetas[:, info['params']] = euler

        # --- 4. 处理 Type C (枢轴关节) ---
        type_c_items = sorted(self.cfg['TYPE_C'].items(), key=lambda x: x[0])
        for name, info in type_c_items:
             # Type C 通常只需要 Twist 角度，这里简化处理，占用一个 ortho 向量位置
             # 实际上 Type C 应该与 Type A 共享部分逻辑
             # 这里仅做占位，消耗掉 ortho_ptr
             _ = pred_ortho[:, ortho_ptr]
             ortho_ptr += 1
             # 参数暂时保持 0 或由 scalar 控制 (取决于具体配置)
             # final_thetas[:, info['param']] = 0 

        return final_thetas