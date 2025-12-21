"""
姿态预测Transformer模块

本文件实现了用于人体姿态预测的Transformer架构组件，包括：
1. 基础注意力机制 (Self-Attention & Cross-Attention)
2. Transformer编码器和解码器
3. 特殊的Dropout策略
4. 预归一化层

主要用于HSMR模型中的SKELTransformerDecoderHead，通过交叉注意力机制
从ViT特征中提取人体姿态、形状和相机参数。

核心设计思想：
- 使用交叉注意力让查询token从图像特征中选择性提取信息
- 支持多种Dropout策略以提升训练稳定性
- 采用Pre-Norm结构提升训练效果
"""

from inspect import isfunction
from typing import Callable, Optional

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from .t_cond_mlp import (
    AdaptiveLayerNorm1D,
    FrequencyEmbedder,
    normalization_layer,
)
# from .vit import Attention, FeedForward


# ===== 工具函数 =====

def exists(val):
    """检查值是否存在（不为None）"""
    return val is not None


def default(val, d):
    """
    返回默认值的工具函数
    
    Args:
        val: 要检查的值
        d: 默认值或生成默认值的函数
        
    Returns:
        如果val存在则返回val，否则返回默认值d
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


# ===== 预归一化层 =====

class PreNorm(nn.Module):
    """
    预归一化包装器
    
    在Transformer中采用Pre-Norm结构，即在注意力/前馈网络之前先进行归一化。
    相比Post-Norm，Pre-Norm结构训练更稳定，收敛更快。
    
    结构：LayerNorm -> Function -> Residual Connection
    """
    
    def __init__(self, dim: int, fn: Callable, norm: str = "layer", norm_cond_dim: int = -1):
        """
        Args:
            dim: 特征维度
            fn: 要包装的函数（如注意力层、前馈网络）
            norm: 归一化类型，默认为"layer"
            norm_cond_dim: 条件归一化维度，-1表示不使用条件归一化
        """
        super().__init__()
        self.norm = normalization_layer(norm, dim, norm_cond_dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        前向传播：先归一化，再应用函数
        
        Args:
            x: 输入张量
            *args, **kwargs: 传递给包装函数的额外参数
            
        Returns:
            函数输出结果
        """
        if isinstance(self.norm, AdaptiveLayerNorm1D):
            # 自适应归一化需要额外参数
            return self.fn(self.norm(x, *args), **kwargs)
        else:
            # 标准归一化
            return self.fn(self.norm(x), **kwargs)


# ===== 前馈网络 =====

class FeedForward(nn.Module):
    """
    Transformer前馈网络（FFN）
    
    标准的两层MLP结构：Linear -> GELU -> Dropout -> Linear -> Dropout
    在Transformer中用于对每个位置的特征进行非线性变换。
    
    架构：
    Input(dim) -> Linear(dim->hidden_dim) -> GELU -> Dropout 
               -> Linear(hidden_dim->dim) -> Dropout -> Output(dim)
    """
    
    def __init__(self, dim, hidden_dim, dropout=0.0):
        """
        Args:
            dim: 输入/输出特征维度
            hidden_dim: 隐藏层维度，通常是dim的2-4倍
            dropout: Dropout概率
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),     # 扩展维度
            nn.GELU(),                      # 激活函数（比ReLU更平滑）
            nn.Dropout(dropout),            # 防止过拟合
            nn.Linear(hidden_dim, dim),     # 恢复原始维度
            nn.Dropout(dropout),            # 输出层Dropout
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, dim)
            
        Returns:
            变换后的张量 (batch_size, seq_len, dim)
        """
        return self.net(x)


# ===== 注意力机制 =====

class Attention(nn.Module):
    """
    多头自注意力机制（Multi-Head Self-Attention）
    
    实现标准的Transformer自注意力，让序列中的每个位置都能关注到其他所有位置。
    Query、Key、Value都来自同一个输入序列。
    
    计算公式：Attention(Q,K,V) = softmax(QK^T/√d_k)V
    
    架构流程：
    Input -> Linear(Q,K,V) -> Multi-Head -> Scaled Dot-Product Attention 
          -> Concat Heads -> Linear -> Output
    """
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        """
        Args:
            dim: 输入特征维度
            heads: 注意力头数
            dim_head: 每个头的维度
            dropout: Dropout概率
        """
        super().__init__()
        inner_dim = dim_head * heads  # 所有头的总维度
        project_out = not (heads == 1 and dim_head == dim)  # 是否需要输出投影

        self.heads = heads
        self.scale = dim_head**-0.5  # 缩放因子：1/√d_k

        self.attend = nn.Softmax(dim=-1)  # 注意力权重归一化
        self.dropout = nn.Dropout(dropout)

        # 一次性生成Q、K、V三个矩阵
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 输出投影层（如果需要的话）
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, dim)
            
        Returns:
            注意力输出 (batch_size, seq_len, dim)
        """
        # 生成Q、K、V并分割为多头
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # 计算注意力分数：QK^T/√d_k
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 应用softmax获得注意力权重
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 加权求和：注意力权重×Value
        out = torch.matmul(attn, v)
        # 合并多头输出
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    """
    多头交叉注意力机制（Multi-Head Cross-Attention）
    
    与自注意力不同，交叉注意力的Query来自一个序列，而Key和Value来自另一个序列。
    这是HSMR模型的核心组件，让查询token能从ViT特征中选择性提取信息。
    
    工作原理：
    - Query: 来自查询token（学习的参数）
    - Key & Value: 来自上下文（ViT图像特征）
    - 让查询token"询问"图像特征中的相关信息
    
    在HSMR中的作用：
    Query(姿态查询) × Context(图像特征) -> 人体姿态参数
    """
    
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        """
        Args:
            dim: 查询序列的特征维度
            context_dim: 上下文序列的特征维度，None时与dim相同
            heads: 注意力头数
            dim_head: 每个头的维度
            dropout: Dropout概率
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5  # 缩放因子

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # 设置上下文维度
        context_dim = default(context_dim, dim)
        
        # 分别处理Query和Key/Value
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)  # 从上下文生成K,V
        self.to_q = nn.Linear(dim, inner_dim, bias=False)               # 从查询生成Q

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context=None):
        """
        前向传播
        
        Args:
            x: 查询序列 (batch_size, query_len, dim)
            context: 上下文序列 (batch_size, context_len, context_dim)
                    如果为None，则退化为自注意力
            
        Returns:
            交叉注意力输出 (batch_size, query_len, dim)
        """
        # 如果没有提供上下文，使用输入本身（退化为自注意力）
        context = default(context, x)
        
        # 从上下文生成Key和Value
        k, v = self.to_kv(context).chunk(2, dim=-1)
        # 从查询生成Query
        q = self.to_q(x)
        
        # 重排为多头格式
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])

        # 计算交叉注意力分数：Query×Key^T
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 获得注意力权重
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 加权求和：注意力权重×Value
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class DeformableCrossAttention(nn.Module):
    """
    可变形交叉注意力机制（Deformable Cross-Attention）
    
    相比标准交叉注意力，Deformable Attention通过学习采样位置来提升效率和性能。
    它不是对所有空间位置计算注意力，而是只关注少量关键位置。
    
    核心思想：
    - 标准Cross-Attention: 对所有H*W个位置计算注意力权重 (计算量大)
    - Deformable Attention: 只对K个采样点计算注意力 (计算量小，更聚焦)
    
    工作原理：
    1. Query预测采样点偏移量：在特征图上应该采样哪些位置
    2. Query预测注意力权重：这些采样点的重要性
    3. 在Value特征图上进行双线性插值采样
    4. 加权求和得到最终输出
    
    在HSMR中的优势：
    - 更高效：只关注关键区域而非整个特征图
    - 更灵活：采样位置可学习，适应不同姿态
    - 更强的空间建模能力：显式建模空间位置关系
    
    参考：Deformable DETR (https://arxiv.org/abs/2010.04159)
    """
    
    def __init__(
        self, 
        dim, 
        context_dim=None, 
        heads=8, 
        dim_head=64, 
        dropout=0.0,
        n_points=8,  # 每个头采样的点数
    ):
        """
        Args:
            dim: 查询序列的特征维度
            context_dim: 上下文序列的特征维度，None时与dim相同
            heads: 注意力头数
            dim_head: 每个头的维度
            dropout: Dropout概率
            n_points: 每个注意力头采样的点数（默认8个）
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.n_points = n_points
        self.dropout = nn.Dropout(dropout)

        # 设置上下文维度
        context_dim = default(context_dim, dim)
        
        # === 原CrossAttention的线性投影层（用于生成Value） ===
        # 注意：Deformable Attention不需要显式的Key，因为采样位置由偏移量决定
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)  # 从上下文生成Value
        self.to_q = nn.Linear(dim, inner_dim, bias=False)          # 从查询生成Query特征
        
        # === 新增：采样偏移量预测网络 ===
        # 输入：Query特征 (dim)
        # 输出：每个头、每个采样点的2D偏移量 (heads * n_points * 2)
        # 偏移量范围：[-1, 1]，表示相对于参考点的归一化偏移
        self.offset_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, heads * n_points * 2),  # 2代表(x, y)偏移
        )
        
        # === 新增：注意力权重预测网络 ===
        # 输入：Query特征 (dim)
        # 输出：每个头、每个采样点的注意力权重 (heads * n_points)
        # 这些权重将通过softmax归一化
        self.attention_weights_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, heads * n_points),
        )

        # === 输出投影层 ===
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context=None, spatial_shapes=None):
        """
        前向传播：可变形交叉注意力
        
        计算流程：
        1. 从Query预测采样偏移量和注意力权重
        2. 将context重塑为2D特征图 (B, H, W, C)
        3. 根据偏移量在特征图上进行双线性插值采样
        4. 用预测的权重对采样特征进行加权求和
        
        Args:
            x: 查询序列 (batch_size, query_len, dim)
               在HSMR中通常是 (B, 1, dim) - 单个查询token
            context: 上下文序列 (batch_size, H*W, context_dim)
                    来自ViT的图像特征，已被展平
            spatial_shapes: 可选的元组 (H, W)，指定特征图的空间尺寸
                           如果为None，将尝试从context长度推断为正方形
            
        Returns:
            可变形交叉注意力输出 (batch_size, query_len, dim)
        """
        B, query_len, _ = x.shape
        
        # 如果没有提供上下文，使用输入本身
        context = default(context, x)
        context_len = context.shape[1]
        
        # === 步骤1：推断或使用提供的空间尺寸 ===
        if spatial_shapes is None:
            # 尝试推断为正方形特征图
            H = W = int(context_len ** 0.5)
            if H * W != context_len:
                raise ValueError(
                    f"Cannot infer spatial shape from context_len={context_len}. "
                    f"Please provide spatial_shapes explicitly."
                )
        else:
            H, W = spatial_shapes
            if H * W != context_len:
                raise ValueError(
                    f"spatial_shapes {spatial_shapes} does not match context_len={context_len}"
                )
        
        # === 步骤2：生成Query特征和Value特征 ===
        q = self.to_q(x)  # (B, query_len, inner_dim)
        v = self.to_v(context)  # (B, H*W, inner_dim)
        
        # 重排Value为2D特征图格式，便于采样
        # (B, H*W, inner_dim) -> (B, H, W, inner_dim)
        v = rearrange(v, 'b (h w) c -> b h w c', h=H, w=W)
        
        # === 步骤3：预测采样偏移量 ===
        # (B, query_len, heads * n_points * 2)
        sampling_offsets = self.offset_net(x)
        sampling_offsets = rearrange(
            sampling_offsets, 
            'b n (h p xy) -> b n h p xy', 
            h=self.heads, 
            p=self.n_points, 
            xy=2
        )  # (B, query_len, heads, n_points, 2)
        
        # 将偏移量通过tanh限制在[-1, 1]范围内
        # 这样采样点不会偏移太远，保证数值稳定性
        sampling_offsets = torch.tanh(sampling_offsets)
        
        # === 步骤4：生成参考点（特征图中心） ===
        # 对于单个查询token，参考点设为特征图中心
        # 归一化坐标：[-1, 1]范围
        reference_points = x.new_zeros(B, query_len, 1, 1, 2)  # (B, query_len, 1, 1, 2)
        # 中心点坐标为(0, 0)在归一化空间中
        
        # === 步骤5：计算实际采样位置 ===
        # 采样位置 = 参考点 + 偏移量
        # (B, query_len, heads, n_points, 2)
        sampling_locations = reference_points + sampling_offsets
        
        # === 步骤6：预测注意力权重 ===
        # (B, query_len, heads * n_points)
        attention_weights = self.attention_weights_net(x)
        attention_weights = rearrange(
            attention_weights,
            'b n (h p) -> b n h p',
            h=self.heads,
            p=self.n_points
        )  # (B, query_len, heads, n_points)
        
        # 对每个头的采样点应用softmax归一化
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # === 步骤7：在Value特征图上进行采样 ===
        # 使用grid_sample进行双线性插值采样
        # 重排为grid_sample所需的格式
        
        # 准备采样网格：(B, query_len, heads, n_points, 2) -> (B*query_len*heads, n_points, 1, 2)
        sampling_grids = rearrange(
            sampling_locations,
            'b n h p xy -> (b n h) p 1 xy'
        )
        
        # 准备Value特征图：(B, H, W, inner_dim) -> (B, heads, H, W, dim_head)
        v = rearrange(v, 'b h w (nh dh) -> b nh h w dh', nh=self.heads, dh=self.dim_head)
        # -> (B*query_len*heads, dim_head, H, W) 为每个查询和头复制
        v_sampled_list = []
        for query_idx in range(query_len):
            # 对每个查询位置单独处理
            v_for_query = rearrange(v, 'b nh h w dh -> (b nh) dh h w')  # (B*heads, dim_head, H, W)
            grids_for_query = sampling_grids[query_idx*B*self.heads:(query_idx+1)*B*self.heads]  # (B*heads, n_points, 1, 2)
            
            # 执行双线性插值采样
            # grid_sample需要 align_corners=True 以匹配[-1,1]的归一化坐标
            sampled = torch.nn.functional.grid_sample(
                v_for_query,  # (B*heads, dim_head, H, W)
                grids_for_query,  # (B*heads, n_points, 1, 2)
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )  # (B*heads, dim_head, n_points, 1)
            
            sampled = sampled.squeeze(-1)  # (B*heads, dim_head, n_points)
            v_sampled_list.append(sampled)
        
        # 合并所有查询位置的结果
        v_sampled = torch.stack(v_sampled_list, dim=1)  # (B*heads, query_len, dim_head, n_points)
        v_sampled = rearrange(v_sampled, '(b h) n d p -> b n h p d', b=B, h=self.heads)
        # (B, query_len, heads, n_points, dim_head)
        
        # === 步骤8：加权求和 ===
        # 使用预测的注意力权重对采样特征进行加权
        # attention_weights: (B, query_len, heads, n_points)
        # v_sampled: (B, query_len, heads, n_points, dim_head)
        attention_weights = attention_weights.unsqueeze(-1)  # (B, query_len, heads, n_points, 1)
        out = (attention_weights * v_sampled).sum(dim=3)  # (B, query_len, heads, dim_head)
        
        # === 步骤9：合并多头并输出投影 ===
        out = rearrange(out, 'b n h d -> b n (h d)')  # (B, query_len, inner_dim)
        return self.to_out(out)


# ===== Transformer块组件 =====

class Transformer(nn.Module):
    """
    标准Transformer编码器
    
    由多层自注意力+前馈网络组成的Transformer编码器。
    每层结构：PreNorm + Self-Attention + Residual + PreNorm + FFN + Residual
    
    特点：
    - 使用Pre-Norm结构提升训练稳定性
    - 支持多种归一化方式
    - 纯自注意力，适用于序列内部信息交互
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
    ):
        """
        Args:
            dim: 特征维度
            depth: Transformer层数
            heads: 注意力头数
            dim_head: 每个头的维度
            mlp_dim: 前馈网络隐藏层维度
            dropout: Dropout概率
            norm: 归一化类型
            norm_cond_dim: 条件归一化维度
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        
        # 构建depth层Transformer块
        for _ in range(depth):
            # 自注意力层
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            # 前馈网络层
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            
            # 用PreNorm包装，形成：LayerNorm -> Function -> Residual
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),  # 注意力块
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),  # 前馈块
                    ]
                )
            )

    def forward(self, x: torch.Tensor, *args):
        """
        前向传播：逐层应用自注意力和前馈网络
        
        Args:
            x: 输入序列 (batch_size, seq_len, dim)
            *args: 传递给归一化层的额外参数
            
        Returns:
            编码后的序列 (batch_size, seq_len, dim)
        """
        for attn, ff in self.layers:
            # 自注意力 + 残差连接
            x = attn(x, *args) + x
            # 前馈网络 + 残差连接
            x = ff(x, *args) + x
        return x


class TransformerCrossAttn(nn.Module):
    """
    交叉注意力Transformer
    
    这是HSMR模型的核心组件，结合了自注意力和交叉注意力。
    每层结构：PreNorm + Self-Attention + Residual + PreNorm + Cross-Attention + Residual + PreNorm + FFN + Residual
    
    工作流程：
    1. 自注意力：让查询token内部进行信息交互
    2. 交叉注意力：让查询token从图像特征中提取相关信息
    3. 前馈网络：对特征进行非线性变换
    
    在HSMR中的作用：
    - 输入：查询token（姿态查询）+ 上下文（ViT图像特征）
    - 输出：融合了图像信息的姿态特征
    
    更新说明：
    - 新增支持Deformable Cross-Attention
    - 可通过use_deformable_attn参数选择使用标准或可变形注意力
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        use_deformable_attn: bool = False,  # 新增：是否使用Deformable Attention
        deformable_attn_n_points: int = 8,  # 新增：Deformable Attention采样点数
    ):
        """
        Args:
            dim: 查询序列特征维度
            depth: Transformer层数
            heads: 注意力头数
            dim_head: 每个头的维度
            mlp_dim: 前馈网络隐藏层维度
            dropout: Dropout概率
            norm: 归一化类型
            norm_cond_dim: 条件归一化维度
            context_dim: 上下文序列特征维度
            use_deformable_attn: 是否使用Deformable Cross-Attention（默认False使用标准CrossAttention）
            deformable_attn_n_points: Deformable Attention每个头的采样点数（默认8）
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        self.use_deformable_attn = use_deformable_attn  # 记录使用的注意力类型
        
        # 构建depth层交叉注意力Transformer块
        for _ in range(depth):
            # 三个核心组件
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)  # 自注意力
            
            # === 交叉注意力层选择 ===
            # 根据配置选择使用标准CrossAttention或DeformableCrossAttention
            if use_deformable_attn:
                # 新方案：使用Deformable Cross-Attention
                # 优势：更高效、更聚焦于关键区域、可学习的采样位置
                ca = DeformableCrossAttention(
                    dim, 
                    context_dim=context_dim, 
                    heads=heads, 
                    dim_head=dim_head, 
                    dropout=dropout,
                    n_points=deformable_attn_n_points,
                )
            else:
                # 原方案：使用标准Cross-Attention（已注释但保留）
                # 计算所有空间位置的注意力权重
                ca = CrossAttention(
                    dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout
                )
            
            ff = FeedForward(dim, mlp_dim, dropout=dropout)                       # 前馈网络
            
            # 用PreNorm包装
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),  # 自注意力块
                        PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),  # 交叉注意力块
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),  # 前馈块
                    ]
                )
            )

    def forward(self, x: torch.Tensor, *args, context=None, context_list=None, spatial_shapes=None):
        """
        前向传播：逐层应用自注意力、交叉注意力和前馈网络
        
        Args:
            x: 查询序列 (batch_size, query_len, dim)
            *args: 传递给归一化层的额外参数
            context: 上下文序列 (batch_size, context_len, context_dim)
            context_list: 每层使用不同上下文的列表，None时所有层使用相同上下文
            spatial_shapes: 可选的元组 (H, W)，用于Deformable Attention指定特征图空间尺寸
            
        Returns:
            变换后的查询序列 (batch_size, query_len, dim)
        """
        # 处理上下文列表：如果未提供，所有层使用相同上下文
        if context_list is None:
            context_list = [context] * len(self.layers)
        if len(context_list) != len(self.layers):
            raise ValueError(f"len(context_list) != len(self.layers) ({len(context_list)} != {len(self.layers)})")

        # 逐层处理
        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            # 1. 自注意力：查询token内部交互
            x = self_attn(x, *args) + x
            
            # 2. 交叉注意力：从图像特征中提取信息
            # 根据是否使用Deformable Attention传递不同参数
            if self.use_deformable_attn:
                # Deformable Attention需要spatial_shapes参数
                x = cross_attn(x, *args, context=context_list[i], spatial_shapes=spatial_shapes) + x
            else:
                # 标准Cross-Attention（原方案）
                x = cross_attn(x, *args, context=context_list[i]) + x
            
            # 3. 前馈网络：非线性变换
            x = ff(x, *args) + x
        return x


# ===== 特殊Dropout策略 =====

class DropTokenDropout(nn.Module):
    """
    丢弃Token的Dropout策略
    
    与标准Dropout不同，这种策略直接丢弃整个token，而不是将部分特征置零。
    在训练时随机移除序列中的一些token，可以提升模型的鲁棒性。
    
    适用场景：
    - 序列长度可变的情况
    - 希望模型对token缺失有鲁棒性
    """
    
    def __init__(self, p: float = 0.1):
        """
        Args:
            p: 丢弃token的概率，范围[0,1]
        """
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        """
        前向传播：随机丢弃token
        
        Args:
            x: 输入序列 (batch_size, seq_len, dim)
            
        Returns:
            丢弃部分token后的序列 (batch_size, new_seq_len, dim)
        """
        # 仅在训练时应用dropout
        if self.training and self.p > 0:
            # 为每个位置生成丢弃掩码（第一个batch的掩码应用到所有batch）
            zero_mask = torch.full_like(x[0, :, 0], self.p).bernoulli().bool()
            # TODO: 可以为每个batch使用不同的排列索引
            if zero_mask.any():
                # 保留未被掩码的token
                x = x[:, ~zero_mask, :]
        return x


class ZeroTokenDropout(nn.Module):
    """
    零化Token的Dropout策略
    
    与DropTokenDropout不同，这种策略将选中的token置零而不是移除。
    保持序列长度不变，但部分token的信息被完全清除。
    
    适用场景：
    - 需要保持固定序列长度
    - 模拟token信息丢失的情况
    """
    
    def __init__(self, p: float = 0.1):
        """
        Args:
            p: 零化token的概率，范围[0,1]
        """
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        """
        前向传播：随机将token置零
        
        Args:
            x: 输入序列 (batch_size, seq_len, dim)
            
        Returns:
            部分token被置零的序列 (batch_size, seq_len, dim)
        """
        # 仅在训练时应用dropout
        if self.training and self.p > 0:
            # 为每个batch的每个位置生成零化掩码
            zero_mask = torch.full_like(x[:, :, 0], self.p).bernoulli().bool()
            # 将选中的token完全置零
            x[zero_mask, :] = 0
        return x


# ===== 完整的编码器和解码器 =====

class TransformerEncoder(nn.Module):
    """
    完整的Transformer编码器
    
    包含token嵌入、位置编码、dropout和多层Transformer块的完整编码器。
    支持频域位置编码和多种dropout策略。
    
    架构流程：
    Input -> Token嵌入 -> 位置编码 -> Dropout -> Transformer层 -> Output
    
    特性：
    - 支持频域位置编码（Fourier特征）
    - 灵活的dropout位置选择
    - 多种token dropout策略
    """
    
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = "drop",
        emb_dropout_loc: str = "token",
        norm: str = "layer",
        norm_cond_dim: int = -1,
        token_pe_numfreq: int = -1,
    ):
        """
        Args:
            num_tokens: token数量
            token_dim: 输入token维度
            dim: Transformer内部特征维度
            depth: Transformer层数
            heads: 注意力头数
            mlp_dim: 前馈网络隐藏层维度
            dim_head: 每个注意力头的维度
            dropout: Transformer内部dropout概率
            emb_dropout: 嵌入层dropout概率
            emb_dropout_type: 嵌入dropout类型 ("drop"/"zero")
            emb_dropout_loc: dropout应用位置 ("input"/"token"/"token_afterpos")
            norm: 归一化类型
            norm_cond_dim: 条件归一化维度
            token_pe_numfreq: 位置编码频率数量，-1表示不使用频域编码
        """
        super().__init__()
        
        # Token嵌入层：支持频域位置编码
        if token_pe_numfreq > 0:
            # 使用频域位置编码（Fourier特征）
            token_dim_new = token_dim * (2 * token_pe_numfreq + 1)
            self.to_token_embedding = nn.Sequential(
                Rearrange("b n d -> (b n) d", n=num_tokens, d=token_dim),
                FrequencyEmbedder(token_pe_numfreq, token_pe_numfreq - 1),
                Rearrange("(b n) d -> b n d", n=num_tokens, d=token_dim_new),
                nn.Linear(token_dim_new, dim),
            )
        else:
            # 标准线性嵌入
            self.to_token_embedding = nn.Linear(token_dim, dim)
            
        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        
        # 嵌入层dropout策略
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)      # 丢弃token
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)      # 零化token
        else:
            raise ValueError(f"Unknown emb_dropout_type: {emb_dropout_type}")
        self.emb_dropout_loc = emb_dropout_loc

        # 核心Transformer编码器
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, norm=norm, norm_cond_dim=norm_cond_dim
        )

    def forward(self, inp: torch.Tensor, *args, **kwargs):
        """
        前向传播
        
        Args:
            inp: 输入token (batch_size, num_tokens, token_dim)
            *args, **kwargs: 传递给Transformer的额外参数
            
        Returns:
            编码后的特征 (batch_size, num_tokens, dim)
        """
        x = inp

        # 根据配置在不同位置应用dropout
        if self.emb_dropout_loc == "input":
            x = self.dropout(x)
            
        # Token嵌入
        x = self.to_token_embedding(x)

        if self.emb_dropout_loc == "token":
            x = self.dropout(x)
            
        # 添加位置编码
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]

        if self.emb_dropout_loc == "token_afterpos":
            x = self.dropout(x)
            
        # 通过Transformer编码器
        x = self.transformer(x, *args)
        return x


class TransformerDecoder(nn.Module):
    """
    交叉注意力Transformer解码器
    
    这是HSMR模型的核心组件，实现了完整的交叉注意力解码器。
    通过交叉注意力机制让查询token从上下文（ViT特征）中提取相关信息。
    
    架构流程：
    Query Token -> Token嵌入 -> 位置编码 -> Dropout -> 交叉注意力Transformer -> Output
                                                    ↑
                                            Context (ViT特征)
    
    在HSMR中的作用：
    - Query: 姿态查询token（可学习参数）
    - Context: ViT提取的图像特征
    - Output: 融合了图像信息的姿态特征，用于预测人体参数
    
    特性：
    - 支持跳过token嵌入（当输入已经是正确维度时）
    - 多种dropout策略
    - 灵活的上下文处理
    - 支持标准Cross-Attention和Deformable Cross-Attention两种模式
    
    更新说明（Deformable Attention支持）：
    - 新增use_deformable_attn参数：选择使用标准或可变形注意力
    - 新增deformable_attn_n_points参数：控制采样点数量
    - forward方法新增spatial_shapes参数：传递特征图的空间尺寸(H,W)
    """
    
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = 'drop',
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        skip_token_embedding: bool = False,
        use_deformable_attn: bool = False,  # 新增：是否使用Deformable Attention
        deformable_attn_n_points: int = 8,  # 新增：Deformable Attention采样点数
    ):
        """
        Args:
            num_tokens: 查询token数量（HSMR中通常为1）
            token_dim: 输入token维度
            dim: Transformer内部特征维度
            depth: 交叉注意力层数
            heads: 注意力头数
            mlp_dim: 前馈网络隐藏层维度
            dim_head: 每个注意力头的维度
            dropout: Transformer内部dropout概率
            emb_dropout: 嵌入层dropout概率
            emb_dropout_type: 嵌入dropout类型 ("drop"/"zero"/"normal")
            norm: 归一化类型
            norm_cond_dim: 条件归一化维度
            context_dim: 上下文特征维度（ViT特征维度）
            skip_token_embedding: 是否跳过token嵌入层
            use_deformable_attn: 是否使用Deformable Cross-Attention（默认False使用标准CrossAttention）
            deformable_attn_n_points: Deformable Attention每个头的采样点数（默认8）
        """
        super().__init__()
        
        # Token嵌入层（可选择跳过）
        if not skip_token_embedding:
            self.to_token_embedding = nn.Linear(token_dim, dim)
        else:
            self.to_token_embedding = nn.Identity()
            if token_dim != dim:
                raise ValueError(
                    f"token_dim ({token_dim}) != dim ({dim}) when skip_token_embedding is True"
                )

        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        
        # 嵌入层dropout策略
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)      # 丢弃token
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)      # 零化token
        elif emb_dropout_type == "normal":
            self.dropout = nn.Dropout(emb_dropout)            # 标准dropout
        else:
            raise ValueError(f"Unknown emb_dropout_type: {emb_dropout_type}")

        # 核心交叉注意力Transformer（支持Deformable Attention）
        self.transformer = TransformerCrossAttn(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            norm=norm,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
            use_deformable_attn=use_deformable_attn,
            deformable_attn_n_points=deformable_attn_n_points,
        )

    def forward(self, inp: torch.Tensor, *args, context=None, context_list=None, spatial_shapes=None):
        """
        前向传播：交叉注意力解码
        
        Args:
            inp: 查询token (batch_size, num_tokens, token_dim)
            *args: 传递给Transformer的额外参数
            context: 上下文序列 (batch_size, context_len, context_dim)
                    在HSMR中是ViT特征
            context_list: 每层使用不同上下文的列表
            spatial_shapes: 可选的元组 (H, W)，用于Deformable Attention指定特征图空间尺寸
            
        Returns:
            解码后的特征 (batch_size, num_tokens, dim)
            在HSMR中用于预测人体姿态、形状和相机参数
        """
        # Token嵌入
        x = self.to_token_embedding(inp)
        b, n, _ = x.shape

        # 应用dropout
        x = self.dropout(x)
        
        # 添加位置编码
        x += self.pos_embedding[:, :n]

        # 通过交叉注意力Transformer
        # 这里是关键：查询token通过交叉注意力从图像特征中提取信息
        # 如果使用Deformable Attention，需要传递spatial_shapes参数
        x = self.transformer(x, *args, context=context, context_list=context_list, spatial_shapes=spatial_shapes)
        return x

