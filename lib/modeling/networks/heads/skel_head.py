from lib.kits.basic import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from pathlib import Path

from omegaconf import OmegaConf

from lib.platform import PM
from lib.body_models.skel_utils.transforms import params_q2rep, params_rep2q

from .utils.pose_transformer import TransformerDecoder


class SKELTransformerDecoderHead(nn.Module):
    """
    基于交叉注意力的SKEL Transformer解码器头部
    
    HSMR模型的预测头部，负责从ViT提取的特征中预测人体姿态、形状和相机参数。
    使用Transformer解码器架构，通过交叉注意力机制将图像特征转换为SKEL模型参数。
    
    模型架构：
    输入(ViT特征) → Reshape → TransformerDecoder → 参数解码器 → SKEL参数
                                    ↑
                              交叉注意力机制
                    (Query: 查询token, Context: ViT特征)
    
    核心组件：
    1. TransformerDecoder: 多层交叉注意力Transformer
    2. 三个参数解码器: 姿态/形状/相机参数预测
    3. 残差连接: 与SKEL平均参数相加提升训练稳定性
    """

    def __init__(self, cfg):
        """
        初始化SKEL Transformer解码器头部
        
        Args:
            cfg: 配置对象，包含模型架构和训练参数
        """
        super().__init__()
        self.cfg = cfg

        # ===== 第1步：确定输出参数维度 =====
        # 根据姿态表示方式确定姿态参数维度
        if cfg.pd_poses_repr == 'rotation_6d':
            n_poses = 24 * 6  # 24个关节 × 6D旋转表示
        elif cfg.pd_poses_repr == 'euler_angle':
            n_poses = 46      # 欧拉角表示：根关节3维 + 身体关节23×3维 - 6 = 46维
        else:
            raise ValueError(f"Unknown pose representation: {cfg.pd_poses_repr}")

        n_betas = 10  # SKEL形状参数维度
        n_cam   = 3   # 相机参数维度 [scale, tx, ty]
        self.input_is_mean_shape = False  # 查询token类型：False=零向量, True=平均参数

        # ===== 第2步：构建TransformerDecoder核心组件 =====
        # Transformer解码器配置：实现交叉注意力机制
        # Query来自查询token，Key/Value来自ViT特征
        transformer_args = {
                'num_tokens' : 1,    # 使用单个查询token
                'token_dim'  : (n_poses + n_betas + n_cam) if self.input_is_mean_shape else 1,
                'dim'        : 1024, # 特征维度
            }
        transformer_args.update(OmegaConf.to_container(cfg.transformer_decoder, resolve=True))
        self.transformer = TransformerDecoder(**transformer_args)

        # ===== 第3步：构建参数解码器 =====
        # 三个独立的线性解码器，将Transformer输出映射到具体参数
        dim = transformer_args['dim']
        self.poses_decoder = nn.Linear(dim, n_poses)  # 姿态参数解码器
        self.betas_decoder = nn.Linear(dim, n_betas)  # 形状参数解码器
        self.cam_decoder   = nn.Linear(dim, n_cam)    # 相机参数解码器

        # ===== 第4步：加载SKEL平均参数作为残差连接的基准值 =====
        # 这些参数用于残差连接，提供合理的初始预测并稳定训练
        skel_mean_path = Path(__file__).parent / 'SKEL_mean.npz'
        skel_mean_params = np.load(skel_mean_path)

        init_poses = torch.from_numpy(skel_mean_params['poses'].astype(np.float32)).unsqueeze(0) # (1, 46)
        if cfg.pd_poses_repr == 'rotation_6d':
            init_poses = params_q2rep(init_poses).reshape(1, 24*6)  # 转换为6D旋转表示
        init_betas = torch.from_numpy(skel_mean_params['betas'].astype(np.float32)).unsqueeze(0)
        init_cam = torch.from_numpy(skel_mean_params['cam'].astype(np.float32)).unsqueeze(0)

        # 注册为缓冲区（不参与梯度更新，但会随模型保存/加载）
        self.register_buffer('init_poses', init_poses)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, **kwargs):
        """
        前向传播过程：从ViT特征预测SKEL参数
        
        数据流：
        ViT特征(B,C,H,W) → Reshape → 查询Token + 上下文 → TransformerDecoder 
                                                              ↓
        SKEL参数 ← 残差连接 ← 参数解码器 ← Transformer输出(B,dim)
        
        Args:
            x (Tensor): ViT骨干网络提取的特征图，形状为 (B, C, H, W)
            
        Returns:
            tuple: (pd_skel_params, pd_cam)
                - pd_skel_params (dict): 预测的SKEL参数
                    - 'poses': 完整姿态参数 (B, 46)
                    - 'poses_orient': 根关节方向 (B, 3)  
                    - 'poses_body': 身体关节姿态 (B, 43)
                    - 'betas': 形状参数 (B, 10)
                - pd_cam (Tensor): 预测的相机参数 (B, 3)
        """
        B = x.shape[0]
        
        # ===== 步骤1：输入特征预处理 =====
        # ViT预训练骨干网络输出是通道优先格式(B,C,H,W)，转换为token序列格式(B,N,C)
        # 这样每个空间位置成为一个token，作为Transformer的上下文
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        # 初始化SKEL平均参数（扩展到当前批次维度）
        init_poses = self.init_poses.expand(B, -1)  # (B, 46)
        init_betas = self.init_betas.expand(B, -1)  # (B, 10)
        init_cam   = self.init_cam.expand(B, -1)    # (B, 3)

        # ===== 步骤2：准备查询Token =====
        # 查询token是Transformer解码器的"问题"，用于从ViT特征中提取相关信息
        with PM.time_monitor('init_token'):
            if self.input_is_mean_shape:
                # 方案A：使用SKEL平均参数作为初始查询token (更丰富的先验信息)
                token = torch.cat([init_poses, init_betas, init_cam], dim=1)[:, None, :]  # (B, 1, C)
            else:
                # 方案B：使用零向量作为初始查询token (让模型从头学习)
                token = x.new_zeros(B, 1, 1)

        # ===== 步骤3：TransformerDecoder核心处理 =====
        # 交叉注意力机制：查询token从ViT特征上下文中提取相关信息
        # Query: 来自token (学习的查询向量)
        # Key & Value: 来自x (ViT提取的图像特征)
        with PM.time_monitor('transformer'):
            # 推断特征图的空间尺寸 (H, W)
            # 如果Transformer使用Deformable Attention，需要知道2D特征图的尺寸
            context_len = x.shape[1]  # H*W
            
            # 尝试推断空间尺寸：优先尝试正方形，否则尝试常见的矩形尺寸
            H = W = int(context_len ** 0.5)
            if H * W == context_len:
                # 完美正方形特征图
                spatial_shapes = (H, W)
            else:
                # 非正方形特征图，尝试常见的矩形尺寸
                # 常见的ViT配置: 192 = 12×16, 384 = 16×24, 等等
                import math
                factors = []
                for i in range(1, int(math.sqrt(context_len)) + 1):
                    if context_len % i == 0:
                        factors.append((i, context_len // i))
                
                # 选择最接近正方形的因子对（宽高比最接近1）
                if factors:
                    spatial_shapes = min(factors, key=lambda x: abs(x[0] - x[1]))
                else:
                    spatial_shapes = None
            
            # 调用Transformer解码器
            # 如果使用Deformable Attention，spatial_shapes参数会被使用
            # 如果使用标准Cross-Attention，spatial_shapes参数会被忽略
            token_out = self.transformer(token, context=x, spatial_shapes=spatial_shapes)
            token_out = token_out.squeeze(1)  # (B, 1, dim) → (B, dim)

        # ===== 步骤4：参数解码与残差连接 =====
        # 三个独立解码器将Transformer输出映射到具体参数
        # 残差连接：预测值 = 解码器输出 + SKEL平均值 (提供合理基准并稳定训练)
        with PM.time_monitor('decode'):
            pd_poses = self.poses_decoder(token_out) + init_poses  # 姿态参数残差连接
            pd_betas = self.betas_decoder(token_out) + init_betas  # 形状参数残差连接  
            pd_cam = self.cam_decoder(token_out) + init_cam        # 相机参数残差连接

        # ===== 步骤5：旋转表示转换 =====
        # 将不同的旋转表示统一转换为欧拉角格式 (SKEL模型的标准输入)
        with PM.time_monitor('rot_repr_transform'):
            if self.cfg.pd_poses_repr == 'rotation_6d':
                # 6D旋转表示 → 欧拉角：更稳定的旋转表示，避免奇异性
                pd_poses = params_rep2q(pd_poses.reshape(-1, 24, 6))  # (B, 24*6) → (B, 46)
            elif self.cfg.pd_poses_repr == 'euler_angle':
                # 直接使用欧拉角表示，无需转换
                pd_poses = pd_poses.reshape(-1, 46)  # (B, 46)
            else:
                raise ValueError(f"Unknown pose representation: {self.cfg.pd_poses_repr}")

        # ===== 步骤6：组织输出参数 =====
        # 将预测的参数组织成SKEL模型所需的格式
        pd_skel_params = {
                'poses'        : pd_poses,           # 完整姿态参数 (B, 46)
                'poses_orient' : pd_poses[:, :3],    # 根关节方向 (B, 3) - 全局旋转
                'poses_body'   : pd_poses[:, 3:],    # 身体关节姿态 (B, 43) - 局部旋转
                'betas'        : pd_betas            # 形状参数 (B, 10) - 控制体型
            }
        return pd_skel_params, pd_cam  # 返回SKEL参数字典和相机参数