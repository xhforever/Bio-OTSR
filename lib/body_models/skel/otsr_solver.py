import torch
import torch.nn as nn
import torch.nn.functional as F
from .kin_skel import BIO_OTSR_CONFIG

class BioOTSRSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = BIO_OTSR_CONFIG
        
        # --- 1. 预处理 Type A (球窝关节) 索引 ---
        # 目的：将字典配置转换为 Tensor 索引，以便在 Forward 中进行批量矩阵运算
        type_a_items = sorted(self.cfg['TYPE_A'].items(), key=lambda x: x[0])
        self.num_type_a = len(type_a_items)
        
        # 收集几何特征索引
        self.register_buffer('a_child_idx', torch.tensor([x[1]['child'] for x in type_a_items], dtype=torch.long))
        self.register_buffer('a_parent_idx', torch.tensor([x[1]['parent'] for x in type_a_items], dtype=torch.long))
        
        # 收集目标参数索引 (展平)
        # 例如: femur_r 对应 params [3, 4, 5], femur_l 对应 [10, 11, 12]
        # 我们需要构建 scatter 索引以便将计算出的 (B, N, 3) 填入 (B, 46)
        a_param_indices = []
        for _, info in type_a_items:
            a_param_indices.extend(info['params'])
        self.register_buffer('a_param_idx', torch.tensor(a_param_indices, dtype=torch.long))
        
        # 记录每个 Type A 关节对应的父节点类型，用于 Global-to-Local
        # 0: Pelvis (Root), 1: Thorax
        # 根据 kin_skel 定义: Femur(大腿)->Pelvis, Humerus(上臂)->Thorax
        # 我们构建一个 mask 或 index 来选取父节点的旋转矩阵
        # 假设顺序是 [femur_l, femur_r, humerus_l, humerus_r] (sorted by name)
        # femur_l/r 父节点是 Pelvis (索引0), humerus_l/r 父节点是 Thorax (索引1)
        a_parent_type = []
        for name, _ in type_a_items:
            if 'femur' in name:
                a_parent_type.append(0) # Pelvis
            elif 'humerus' in name:
                a_parent_type.append(1) # Thorax
            else:
                a_parent_type.append(0) # Default
        self.register_buffer('a_parent_type_idx', torch.tensor(a_parent_type, dtype=torch.long))

        # --- 2. 预处理 Type B (铰链关节) ---
        type_b_items = sorted(self.cfg['TYPE_B'].items(), key=lambda x: x[1]['param'])
        self.register_buffer('b_param_idx', torch.tensor([x[1]['param'] for x in type_b_items], dtype=torch.long))
        
        # 构建 Limit Tensor: (N_B, 2) -> [min, max]
        limits = []
        for _, info in type_b_items:
            limits.append(info['limit'])
        self.register_buffer('b_limits', torch.tensor(limits, dtype=torch.float32))

        # --- 3. 预处理 Type C (Twist) ---
        type_c_items = sorted(self.cfg['TYPE_C'].items(), key=lambda x: x[0])
        self.register_buffer('c_child_idx', torch.tensor([x[1]['child'] for x in type_c_items], dtype=torch.long))
        self.register_buffer('c_parent_idx', torch.tensor([x[1]['parent'] for x in type_c_items], dtype=torch.long))
        self.register_buffer('c_grandparent_idx', torch.tensor([x[1]['grandparent'] for x in type_c_items], dtype=torch.long))
        self.register_buffer('c_param_idx', torch.tensor([x[1]['param'] for x in type_c_items], dtype=torch.long))

        # --- 4. Type D 索引 ---
        self.register_buffer('d_indices', torch.tensor(self.cfg['TYPE_D_INDICES'], dtype=torch.long))
        
        # 脊柱参数索引 (用于提前计算躯干旋转)
        # Pelvis: 0-3, Lumbar: 17-20, Thorax: 20-23
        self.register_buffer('idx_pelvis', torch.tensor([0, 1, 2], dtype=torch.long))
        self.register_buffer('idx_lumbar', torch.tensor([17, 18, 19], dtype=torch.long))
        self.register_buffer('idx_thorax', torch.tensor([20, 21, 22], dtype=torch.long))

    def robust_matrix_to_euler_batch(self, R):
        """
        向量化的鲁棒旋转矩阵转欧拉角
        R: (B, N, 3, 3)
        Returns: (B, N, 3)
        """
        sy = torch.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
        singular = sy < 1e-6

        x = torch.zeros_like(sy)
        y = torch.zeros_like(sy)
        z = torch.zeros_like(sy)

        # 非奇异情况
        x[~singular] = torch.atan2(R[~singular, 2, 1], R[~singular, 2, 2])
        y[~singular] = torch.atan2(-R[~singular, 2, 0], sy[~singular])
        z[~singular] = torch.atan2(R[~singular, 1, 0], R[~singular, 0, 0])

        # 奇异情况
        x[singular] = 0
        y[singular] = torch.atan2(-R[singular, 2, 0], sy[singular])
        z[singular] = torch.atan2(-R[singular, 0, 1], R[singular, 1, 1])

        return torch.stack([x, y, z], dim=-1)

    def euler_to_matrix_batch(self, euler):
        """
        向量化欧拉角转矩阵
        euler: (B, 3) -> Returns: (B, 3, 3)
        """
        B = euler.shape[0]
        x, y, z = euler[:, 0], euler[:, 1], euler[:, 2]
        
        cx, sx = torch.cos(x), torch.sin(x)
        cy, sy = torch.cos(y), torch.sin(y)
        cz, sz = torch.cos(z), torch.sin(z)
        
        # Rz * Ry * Rx
        R = torch.stack([
            cy*cz,              cz*sx*sy - cx*sz,   cx*cz*sy + sx*sz,
            cy*sz,              cx*cz + sx*sy*sz,   -cz*sx + cx*sy*sz,
            -sy,                cy*sx,              cx*cy
        ], dim=-1).reshape(B, 3, 3)
        return R

    def compute_twist_batch(self, u_vec, v_vec, t_vec):
        """
        向量化 Twist 计算 + NaN 保护
        Inputs: (B, N, 3)
        Returns: (B, N)
        """
        # 1. 参考轴 (Cross Product)
        cross_prod = torch.cross(v_vec, u_vec, dim=-1)
        norm = torch.norm(cross_prod, dim=-1, keepdim=True)
        
        # [Fix] NaN Protection
        default_axis = torch.zeros_like(cross_prod)
        default_axis[..., 2] = 1.0 # Z-axis
        
        stable_axis = torch.where(norm < 1e-6, default_axis, cross_prod)
        ref_x = F.normalize(stable_axis, dim=-1)
        ref_y = torch.cross(u_vec, ref_x, dim=-1)
        
        # 2. 投影
        proj = (t_vec * u_vec).sum(dim=-1, keepdim=True) * u_vec
        t_proj = t_vec - proj
        t_proj = F.normalize(t_proj, dim=-1)
        
        # 3. 角度
        cos_theta = (t_proj * ref_x).sum(dim=-1)
        sin_theta = (t_proj * ref_y).sum(dim=-1)
        return torch.atan2(sin_theta, cos_theta)

    def forward(self, pred_kp3d, pred_ortho, pred_scalar):
        # pred_kp3d: (B, 24, 3)
        # pred_ortho: (B, 6, 3)
        # pred_scalar: (B, 32)
        
        B, device = pred_kp3d.shape[0], pred_kp3d.device
        pred_kp3d = pred_kp3d.float()
        pred_ortho = pred_ortho.float()
        pred_scalar = pred_scalar.float()

        # 初始化输出
        final_thetas = torch.zeros(B, 46, dtype=torch.float32, device=device)

        # =========================================================
        # 1. Type D (Spine, Head, Pelvis) - 纯标量复制 (向量化)
        # =========================================================
        n_type_d = self.d_indices.shape[0]
        final_thetas[:, self.d_indices] = pred_scalar[:, :n_type_d]

        # =========================================================
        # 2. 躯干 Global 旋转计算 (FK) - 必需串行但很短
        # =========================================================
        # 目的：为四肢计算提供父节点坐标系
        # R_pelvis (Root)
        r_pelvis = self.euler_to_matrix_batch(final_thetas[:, self.idx_pelvis])
        
        # R_lumbar = R_pelvis * R_lumbar_local
        r_lumbar_local = self.euler_to_matrix_batch(final_thetas[:, self.idx_lumbar])
        r_lumbar = torch.matmul(r_pelvis, r_lumbar_local)
        
        # R_thorax = R_lumbar * R_thorax_local (这是 Humerus 的父节点)
        r_thorax_local = self.euler_to_matrix_batch(final_thetas[:, self.idx_thorax])
        r_thorax = torch.matmul(r_lumbar, r_thorax_local)

        # 将父节点旋转堆叠: Stack [Pelvis, Thorax] -> (B, 2, 3, 3)
        # 索引 0: Pelvis (Femur父), 索引 1: Thorax (Humerus父)
        parent_rots_stack = torch.stack([r_pelvis, r_thorax], dim=1)

        # =========================================================
        # 3. Type A (Limbs) - 向量化几何解算 + Global-to-Local
        # =========================================================
        # 3.1 提取向量: (B, N_A, 3)
        child_p = pred_kp3d[:, self.a_child_idx]
        parent_p = pred_kp3d[:, self.a_parent_idx]
        b_vec = F.normalize(child_p - parent_p, dim=-1)
        
        # 3.2 提取 Ortho 向量 (假设 Ortho 前 4 个是 Type A)
        t_raw = pred_ortho[:, :self.num_type_a] 
        
        # 3.3 批量 Gram-Schmidt -> Global Rotations (B, N_A, 3, 3)
        proj = (t_raw * b_vec).sum(dim=-1, keepdim=True) * b_vec
        o_vec = F.normalize(t_raw - proj, dim=-1)
        z_vec = torch.cross(b_vec, o_vec, dim=-1)
        r_curr_global = torch.stack([b_vec, o_vec, z_vec], dim=-1)
        
        # 3.4 选取父节点旋转
        # parent_rots_stack: (B, 2, 3, 3)
        # self.a_parent_type_idx: (N_A,) -> [0, 0, 1, 1] (Indices into dim 1)
        # Expand indices for gather: (B, N_A, 3, 3)
        indices = self.a_parent_type_idx.view(1, -1, 1, 1).expand(B, -1, 3, 3)
        r_parent_global = torch.gather(parent_rots_stack, 1, indices)
        
        # 3.5 Global to Local: R_local = R_parent.T @ R_curr
        r_local = torch.matmul(r_parent_global.transpose(-1, -2), r_curr_global)
        
        # 3.6 转欧拉角并赋值
        euler_a = self.robust_matrix_to_euler_batch(r_local) # (B, N_A, 3)
        
        # Flatten and scatter
        final_thetas[:, self.a_param_idx] = euler_a.reshape(B, -1)

        # =========================================================
        # 4. Type B (Elbow/Knee) - 向量化 Tanh 映射
        # =========================================================
        # 取出标量 (跳过 Type D 的部分)
        s_vals = pred_scalar[:, n_type_d : n_type_d + self.b_param_idx.shape[0]]
        
        # limits: (N_B, 2) -> expand to (B, N_B, 2)
        limits = self.b_limits.unsqueeze(0)
        low, high = limits[..., 0], limits[..., 1]
        
        # Tanh Mapping
        b_vals = (torch.tanh(s_vals) + 1) / 2 * (high - low) + low
        final_thetas[:, self.b_param_idx] = b_vals

        # =========================================================
        # 5. Type C (Twist) - 向量化几何计算
        # =========================================================
        # 5.1 提取向量
        u_vec = F.normalize(pred_kp3d[:, self.c_child_idx] - pred_kp3d[:, self.c_parent_idx], dim=-1)
        v_vec = F.normalize(pred_kp3d[:, self.c_parent_idx] - pred_kp3d[:, self.c_grandparent_idx], dim=-1)
        
        # Ortho 向量 (接在 Type A 后面)
        t_raw = pred_ortho[:, self.num_type_a : self.num_type_a + self.c_param_idx.shape[0]]
        
        # 5.2 批量计算
        c_vals = self.compute_twist_batch(u_vec, v_vec, t_raw)
        final_thetas[:, self.c_param_idx] = c_vals

        return final_thetas