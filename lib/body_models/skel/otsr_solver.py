import torch
import torch.nn as nn
import torch.nn.functional as F
from .kin_skel import BIO_OTSR_CONFIG

class BioOTSRSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = BIO_OTSR_CONFIG
        
        # =========================================================
        # 1. 预处理 Type A (球窝关节/3DOF) 索引 - 用于向量化
        # =========================================================
        # 按照关节名称排序，确保确定性顺序
        type_a_items = sorted(self.cfg['TYPE_A'].items(), key=lambda x: x[0])
        self.num_type_a = len(type_a_items)
        
        # 收集几何特征索引: Child(当前关节), Parent(父关节)
        self.register_buffer('a_child_idx', torch.tensor([x[1]['child'] for x in type_a_items], dtype=torch.long))
        self.register_buffer('a_parent_idx', torch.tensor([x[1]['parent'] for x in type_a_items], dtype=torch.long))
        
        # 收集目标参数索引 (flattened)
        # 例如: femur_r 对应 params [3, 4, 5], 需展平以便 scatter
        a_param_indices = []
        for _, info in type_a_items:
            a_param_indices.extend(info['params'])
        self.register_buffer('a_param_idx', torch.tensor(a_param_indices, dtype=torch.long))
        
        # [关键] 记录每个 Type A 关节对应的"父骨骼"类型，用于 Global-to-Local 转换
        # 我们预先计算好躯干的旋转：0: Pelvis(骨盆), 1: Thorax(胸腔)
        # 规则: 腿部(Femur)父节点是 Pelvis; 手臂(Humerus/Scapula)父节点是 Thorax
        a_parent_type = []
        for name, _ in type_a_items:
            if 'femur' in name:
                a_parent_type.append(0) # Index 0 -> Pelvis
            elif 'humerus' in name or 'scapula' in name:
                a_parent_type.append(1) # Index 1 -> Thorax
            else:
                a_parent_type.append(0) # Default to Pelvis if unknown
        self.register_buffer('a_parent_type_idx', torch.tensor(a_parent_type, dtype=torch.long))

        # =========================================================
        # 2. 预处理 Type B (铰链关节/1DOF) - 如膝盖、手肘
        # =========================================================
        type_b_items = sorted(self.cfg['TYPE_B'].items(), key=lambda x: x[1]['param'])
        self.register_buffer('b_param_idx', torch.tensor([x[1]['param'] for x in type_b_items], dtype=torch.long))
        
        # 构建 Limits Tensor: (N_B, 2) -> [min, max]
        limits = [info['limit'] for _, info in type_b_items]
        self.register_buffer('b_limits', torch.tensor(limits, dtype=torch.float32))

        # =========================================================
        # 3. 预处理 Type C (Twist/枢轴关节) - 如前臂旋转
        # =========================================================
        type_c_items = sorted(self.cfg['TYPE_C'].items(), key=lambda x: x[0])
        self.register_buffer('c_child_idx', torch.tensor([x[1]['child'] for x in type_c_items], dtype=torch.long))
        self.register_buffer('c_parent_idx', torch.tensor([x[1]['parent'] for x in type_c_items], dtype=torch.long))
        self.register_buffer('c_grandparent_idx', torch.tensor([x[1]['grandparent'] for x in type_c_items], dtype=torch.long))
        self.register_buffer('c_param_idx', torch.tensor([x[1]['param'] for x in type_c_items], dtype=torch.long))

        # =========================================================
        # 4. Type D (直接参数) & 脊柱索引
        # =========================================================
        self.register_buffer('d_indices', torch.tensor(self.cfg['TYPE_D_INDICES'], dtype=torch.long))
        
        # 脊柱参数索引 (用于 FK 计算躯干全局旋转)
        # 假设 SMPL/SKEL 顺序: Pelvis(0-2), Lumbar(17-19), Thorax(20-22) (基于 XYZ 欧拉角)
        self.register_buffer('idx_pelvis', torch.tensor([0, 1, 2], dtype=torch.long))
        self.register_buffer('idx_lumbar', torch.tensor([17, 18, 19], dtype=torch.long))
        self.register_buffer('idx_thorax', torch.tensor([20, 21, 22], dtype=torch.long))

    def robust_matrix_to_euler_batch(self, R):
        """
        向量化的鲁棒旋转矩阵转 XYZ 欧拉角 (防止 Gimbal Lock 导致梯度爆炸)
        R: (B, N, 3, 3)
        Returns: (B, N, 3)
        """
        # sy = sqrt(R00^2 + R10^2)
        sy = torch.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
        singular = sy < 1e-6

        x = torch.zeros_like(sy)
        y = torch.zeros_like(sy)
        z = torch.zeros_like(sy)

        # 非奇异情况
        x[~singular] = torch.atan2(R[~singular, 2, 1], R[~singular, 2, 2])
        y[~singular] = torch.atan2(-R[~singular, 2, 0], sy[~singular])
        z[~singular] = torch.atan2(R[~singular, 1, 0], R[~singular, 0, 0])

        # 奇异情况 (Gimbal Lock)
        x[singular] = 0
        y[singular] = torch.atan2(-R[singular, 2, 0], sy[singular])
        z[singular] = torch.atan2(-R[singular, 0, 1], R[singular, 1, 1])

        return torch.stack([x, y, z], dim=-1)

    def euler_to_matrix_batch(self, euler):
        """
        向量化 XYZ 欧拉角转矩阵
        euler: (B, 3) -> Returns: (B, 3, 3)
        """
        B = euler.shape[0]
        x, y, z = euler[:, 0], euler[:, 1], euler[:, 2]
        
        cx, sx = torch.cos(x), torch.sin(x)
        cy, sy = torch.cos(y), torch.sin(y)
        cz, sz = torch.cos(z), torch.sin(z)
        
        # Rz * Ry * Rx
        # Row-major construction
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
        # 1. 计算参考轴 (u cross v)
        # 如果手臂伸直 (u || v), cross 为 0, normalize 会产生 NaN
        cross_prod = torch.cross(v_vec, u_vec, dim=-1)
        norm = torch.norm(cross_prod, dim=-1, keepdim=True)
        
        # [Fix] NaN Protection: 如果共线，使用默认 Z 轴作为参考
        default_axis = torch.zeros_like(cross_prod)
        default_axis[..., 2] = 1.0 
        
        stable_axis = torch.where(norm < 1e-6, default_axis, cross_prod)
        ref_x = F.normalize(stable_axis, dim=-1)
        ref_y = torch.cross(u_vec, ref_x, dim=-1)
        
        # 2. 投影 Twist 向量到 u_vec 的垂直平面
        proj = (t_vec * u_vec).sum(dim=-1, keepdim=True) * u_vec
        t_proj = t_vec - proj
        t_proj = F.normalize(t_proj, dim=-1)
        
        # 3. 解算角度
        cos_theta = (t_proj * ref_x).sum(dim=-1)
        sin_theta = (t_proj * ref_y).sum(dim=-1)
        return torch.atan2(sin_theta, cos_theta)

    def forward(self, pred_kp3d, pred_ortho, pred_scalar):
        """
        Args:
            pred_kp3d: (B, 24, 3) - 预测的3D关键点
            pred_ortho: (B, N_Ortho, 3) - 预测的正交向量
            pred_scalar: (B, N_Scalar) - 预测的标量参数
        Returns:
            final_thetas: (B, 46) - SKEL Pose 参数 (欧拉角)
        """
        B, device = pred_kp3d.shape[0], pred_kp3d.device
        pred_kp3d = pred_kp3d.float()
        pred_ortho = pred_ortho.float()
        pred_scalar = pred_scalar.float()

        # 初始化输出
        final_thetas = torch.zeros(B, 46, dtype=torch.float32, device=device)

        # =========================================================
        # 1. Type D (Spine, Head, Pelvis) - 纯标量复制
        # =========================================================
        n_type_d = self.d_indices.shape[0]
        final_thetas[:, self.d_indices] = pred_scalar[:, :n_type_d]

        # =========================================================
        # 2. 躯干 Global 旋转计算 (FK)
        # =========================================================
        # 目的：为四肢 (Type A) 计算提供父节点坐标系
        # R_pelvis (Root)
        r_pelvis = self.euler_to_matrix_batch(final_thetas[:, self.idx_pelvis])
        
        # R_lumbar = R_pelvis * R_lumbar_local
        r_lumbar_local = self.euler_to_matrix_batch(final_thetas[:, self.idx_lumbar])
        r_lumbar = torch.matmul(r_pelvis, r_lumbar_local)
        
        # R_thorax = R_lumbar * R_thorax_local (这是 Humerus/Scapula 的父节点)
        r_thorax_local = self.euler_to_matrix_batch(final_thetas[:, self.idx_thorax])
        r_thorax = torch.matmul(r_lumbar, r_thorax_local)

        # 堆叠父节点旋转: Stack [Pelvis, Thorax] -> (B, 2, 3, 3)
        # Index 0: Pelvis (Femur父), Index 1: Thorax (Humerus父)
        parent_rots_stack = torch.stack([r_pelvis, r_thorax], dim=1)

        # =========================================================
        # 3. Type A (Limbs) - 向量化几何解算 + Global-to-Local
        # =========================================================
        # 3.1 提取向量: (B, N_A, 3)
        child_p = pred_kp3d[:, self.a_child_idx]
        parent_p = pred_kp3d[:, self.a_parent_idx]
        
        # b_vec: 骨骼向量，对应局部坐标系的 Y 轴 [0, 1, 0]
        b_vec = F.normalize(child_p - parent_p, dim=-1)
        
        # t_raw: 预测的辅助向量，对应局部坐标系的 X 轴 [1, 0, 0]
        # (假设 pred_ortho 前 num_type_a 个对应 Type A)
        t_raw = pred_ortho[:, :self.num_type_a] 
        
        # 3.2 批量 Gram-Schmidt 构建正交基 [X, Y, Z]
        # 目标: 使得 Y 轴严格等于 b_vec，X 轴接近 t_raw
        
        # Step 1: 让 t_raw 正交于 b_vec，作为 X 轴
        proj = (t_raw * b_vec).sum(dim=-1, keepdim=True) * b_vec
        x_vec = F.normalize(t_raw - proj, dim=-1) # 这是真正的 X 轴
        
        # Step 2: 计算 Z = X cross Y (遵循右手定则)
        z_vec = torch.cross(x_vec, b_vec, dim=-1)
        
        # Step 3: 堆叠矩阵 [X, Y, Z] -> (B, N_A, 3, 3)
        # 注意: stack dim=-1 表示列向量堆叠
        r_curr_global = torch.stack([x_vec, b_vec, z_vec], dim=-1)
        
        # 3.3 选取父节点旋转
        # self.a_parent_type_idx: (N_A,) -> [0, 0, 1, 1...]
        indices = self.a_parent_type_idx.view(1, -1, 1, 1).expand(B, -1, 3, 3)
        r_parent_global = torch.gather(parent_rots_stack, 1, indices)
        
        # 3.4 Global to Local: R_local = R_parent.T @ R_curr
        # transpose(-1, -2) 是矩阵转置
        r_local = torch.matmul(r_parent_global.transpose(-1, -2), r_curr_global)
        
        # 3.5 转欧拉角并赋值
        euler_a = self.robust_matrix_to_euler_batch(r_local) # (B, N_A, 3)
        
        # Flatten and scatter to final_thetas
        final_thetas[:, self.a_param_idx] = euler_a.reshape(B, -1)

        # =========================================================
        # 4. Type B (Elbow/Knee) - 向量化 Tanh 映射
        # =========================================================
        # 取出标量 (跳过 Type D 的部分)
        s_vals = pred_scalar[:, n_type_d : n_type_d + self.b_param_idx.shape[0]]
        
        # limits: (N_B, 2) -> expand to (B, N_B, 2)
        limits = self.b_limits.unsqueeze(0)
        low, high = limits[..., 0], limits[..., 1]
        
        # Tanh Mapping: (-inf, inf) -> [low, high]
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