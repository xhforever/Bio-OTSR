import torch
import torch.nn as nn
import torch.nn.functional as F
from .kin_skel import BIO_OTSR_CONFIG

class BioOTSRSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = BIO_OTSR_CONFIG

    def rotation_matrix_to_euler_angles(self, R):
        """将旋转矩阵转换为 XYZ 欧拉角"""
        # 确保 float32 避免数值不稳定
        R = R.float()
        sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
        
        # 加上极小值防止除以0
        sy = sy + 1e-6
        
        x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        y = torch.atan2(-R[:, 2, 0], sy)
        z = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        return torch.stack([x, y, z], dim=-1)

    def compute_twist_angle(self, u_vec, v_vec, t_vec):
        """
        计算 Twist 旋转角
        Args:
            u_vec: (B, 3) 前臂骨骼轴 (Swing Axis)
            v_vec: (B, 3) 上臂骨骼轴 (Reference Axis)
            t_vec: (B, 3) 预测的 Twist 向量
        Returns:
            angle: (B,) 旋转弧度
        """
        # 1. 构建参考坐标系 (Zero-Twist Frame)
        # 以前臂 u 和上臂 v 构成的平面的法线作为 X 轴参考 (肘关节转轴)
        # 注意：若手臂完全伸直(u, v平行)，此叉乘不稳定。实际应用通常加个扰动或使用默认轴。
        ref_x = torch.cross(v_vec, u_vec) # (B, 3)
        ref_x = F.normalize(ref_x, dim=-1)
        
        # 构建参考 Y 轴 (垂直于骨骼轴 u 和 参考 X)
        ref_y = torch.cross(u_vec, ref_x) # (B, 3)
        
        # 2. 将预测的 Twist 向量投影到垂直于骨骼轴 u 的平面上
        # t_proj = t - (t . u) * u
        proj = (t_vec * u_vec).sum(dim=-1, keepdim=True) * u_vec
        t_proj = t_vec - proj
        t_proj = F.normalize(t_proj, dim=-1)
        
        # 3. 计算夹角 (atan2)
        # cos_theta = t_proj . ref_x
        # sin_theta = t_proj . ref_y
        cos_theta = (t_proj * ref_x).sum(dim=-1)
        sin_theta = (t_proj * ref_y).sum(dim=-1)
        
        angle = torch.atan2(sin_theta, cos_theta)
        return angle

    def forward(self, pred_kp3d, pred_ortho, pred_scalar):
        # 强制转换为 float32 保证几何运算精度
        pred_kp3d = pred_kp3d.float()
        pred_ortho = pred_ortho.float()
        pred_scalar = pred_scalar.float()

        B, device = pred_kp3d.shape[0], pred_kp3d.device
        final_thetas = torch.zeros(B, 46, dtype=torch.float32).to(device)
        
        ortho_ptr = 0
        scalar_ptr = 0

        # --- 1. 处理 Type D (直接参数) ---
        n_type_d = len(self.cfg['TYPE_D_INDICES'])
        final_thetas[:, self.cfg['TYPE_D_INDICES']] = pred_scalar[:, :n_type_d]
        scalar_ptr += n_type_d

        # --- 2. 处理 Type B (铰链关节) ---
        type_b_items = sorted(self.cfg['TYPE_B'].items(), key=lambda x: x[1]['param'])
        for name, info in type_b_items:
            s = pred_scalar[:, scalar_ptr]
            scalar_ptr += 1
            limit = info['limit']
            val = (torch.tanh(s) + 1) / 2 * (limit[1] - limit[0]) + limit[0]
            final_thetas[:, info['param']] = val

        # --- 3. 处理 Type A (球窝关节 - 3D 旋转) ---
        type_a_items = sorted(self.cfg['TYPE_A'].items(), key=lambda x: x[0])
        for name, info in type_a_items:
            child_idx, parent_idx = info['child'], info['parent']
            b_vec = F.normalize(pred_kp3d[:, child_idx] - pred_kp3d[:, parent_idx], dim=-1)
            
            t_raw = pred_ortho[:, ortho_ptr]
            ortho_ptr += 1
            
            # Gram-Schmidt 正交化
            proj = (t_raw * b_vec).sum(dim=-1, keepdim=True) * b_vec
            o_vec = F.normalize(t_raw - proj, dim=-1)
            
            z_vec = torch.cross(b_vec, o_vec)
            R_global = torch.stack([b_vec, o_vec, z_vec], dim=-1)
            
            euler = self.rotation_matrix_to_euler_angles(R_global)
            final_thetas[:, info['params']] = euler

        # --- 4. [完整版] 处理 Type C (枢轴关节 - 1D 旋转) ---
        type_c_items = sorted(self.cfg['TYPE_C'].items(), key=lambda x: x[0])
        for name, info in type_c_items:
             t_raw = pred_ortho[:, ortho_ptr]
             ortho_ptr += 1
             
             # 确保 config 中定义了这三个关键节点
             if 'child' in info and 'parent' in info and 'grandparent' in info:
                 c_idx, p_idx, gp_idx = info['child'], info['parent'], info['grandparent']
                 
                 # 1. 前臂轴 (Swing)
                 u_vec = F.normalize(pred_kp3d[:, c_idx] - pred_kp3d[:, p_idx], dim=-1)
                 
                 # 2. 上臂轴 (Reference for 0-twist)
                 v_vec = F.normalize(pred_kp3d[:, p_idx] - pred_kp3d[:, gp_idx], dim=-1)
                 
                 # 3. 解算角度
                 angle = self.compute_twist_angle(u_vec, v_vec, t_raw)
                 
                 # 4. 赋值 (可根据需要添加 limit 限制)
                 final_thetas[:, info['param']] = angle
             else:
                 # 如果 config 没写全，保持静默 (或打印警告)
                 pass

        return final_thetas