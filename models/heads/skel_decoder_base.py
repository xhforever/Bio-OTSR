import torch
import torch.nn as nn
import numpy as np

from omegaconf import OmegaConf

from body_models.skel_utils.transforms import params_q2rep, params_rep2q
from util.constants import SKEL_MEAN_PARAMS
from util.pylogger import get_pylogger
from .utils.pose_transformer import SKELTransformerDecoder

# [NEW] 导入 Bio-OTSR Solver (确保第2步的文件已创建)
from lib.body_models.skel.otsr_solver import BioOTSRSolver

logger = get_pylogger(__name__)

class SKELTransformerDecoderHeadBase(nn.Module):
    """
        Bio-OTSR Modified Decoder Head
        1. Predicts Geometric Features (XYZ, Ortho, Scalar) instead of Poses directly.
        2. Uses BioOTSRSolver to convert features to SKEL Poses.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 定义维度常数 (需与 kin_skel.py 中的定义对应)
        # Type A (4个关节) + Type C (2个关节) = 6个 Ortho 向量
        self.n_ortho_vecs = 6 
        # Type B (10个关节) + Type D (22个参数) = 32 个标量
        self.n_scalars = 32 
        self.n_joints = 24

        self.num_decoder_layers = 6 
        n_betas = 10
        n_cam = 3

        # Build transformer decoder.
        transformer_args = {
                'num_tokens' : 1,
                'dim'        : cfg.hub.skel_head.transformer_decoder.context_dim,
                'token_dim'  : cfg.hub.skel_head.transformer_decoder.context_dim,
                'skip_token_embedding' : True,
            }
        transformer_args.update(OmegaConf.to_container(cfg.hub.skel_head.transformer_decoder, resolve=True))  
          
        self.transformer = SKELTransformerDecoder(**transformer_args)
        
        dim = transformer_args['dim']

        # --- [NEW] Bio-OTSR 专用解码分支 ---
        
        # 1. Swing Head: 预测 24 个关节的 3D 坐标 (相对 Root 或 Patch 中心)
        self.xyz_decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, self.n_joints * 3), 
        )

        # 2. Twist Head: 预测 Type A/C 关节的正交方向向量
        self.ortho_decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, self.n_ortho_vecs * 3), 
        )

        # 3. Scalar Head: 预测 Type B/D 关节的标量参数
        self.scalar_decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, self.n_scalars),
        )

        # 实例化数学解算器 (不可训练)
        self.solver = BioOTSRSolver()

        # ----------------------------------

        # 保持原有的 Cam 和 Betas 解码器
        self.cam_decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, n_cam),
        )

        self.betas_decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, n_betas),
        )

        # Embedding projection layers
        # 注意: 输入维度可能需要根据新的 token 组合调整，这里暂时保持兼容性
        # 如果 Transformer 输入需要几何特征 embedding，可以在这里修改，
        # 但为了最小化改动，我们暂且保留原有的 embedding 逻辑或略作调整。
        self.to_pose_embedding = nn.Linear(24 * 6 + 3, dim) # 这里的 24*6 可能不再适用，见下方 forward 说明
        self.to_betas_embedding = nn.Linear(10 + 3, dim)
        self.to_cam_embedding = nn.Linear(3 + 3, dim)

        logger.info("Bio-OTSR Decoder initialized with Geometric Heads!")

        # Mean Params Loading (主要用于 Betas 和 Cam，Pose 初始化改为零或其他)
        mean_params = np.load(SKEL_MEAN_PARAMS, allow_pickle=True)
        # mean_poses = torch.from_numpy(mean_params['poses']).unsqueeze(0) # 原有的 mean pose 是角度，这里暂时不用
        mean_betas = torch.from_numpy(mean_params['betas']).unsqueeze(0)
        mean_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)

        self.register_buffer('mean_betas', mean_betas)
        self.register_buffer('mean_cam', mean_cam)

        
    @property
    def layers(self):
        return self.transformer.layers
    
    @property
    def dropout(self):
        return self.transformer.dropout
    
    @property
    def pos_embedding(self):
        return self.transformer.pos_embedding


    def forward(self, x, **kwargs):
        # x: (B, dim) Global Feature Token
        # x_norm_patch: (B, N, dim) Patch Features
        
        x_cls = x 
        x_norm_patch = kwargs.get('x_norm_patch', None)
        bbox_info = kwargs.get('bbox_info', None)

        assert x_norm_patch is not None and bbox_info is not None, 'x_norm_patch and bbox_info are required'
        B = x.shape[0]

        # --- 1. Initialization (Iterative Refinement Start) ---
        
        # Betas & Cam 初始化 (保持原样)
        betas_init = self.betas_decoder(x_cls) + self.mean_betas
        cam_init = self.cam_decoder(x_cls) + self.mean_cam
        
        # [NEW] Geometric Features 初始化
        # 不再使用 mean_poses，而是从 x_cls 初始化几何猜测
        xyz_i = self.xyz_decoder(x_cls)       # (B, 24*3)
        ortho_i = self.ortho_decoder(x_cls)   # (B, 6*3)
        scalar_i = self.scalar_decoder(x_cls) # (B, 32)
        
        # 为了 Transformer Token 的构建，我们需要一个 Pose 的 Embedding 表示。
        # 原代码使用 6D 旋转。为了兼容 self.to_pose_embedding，
        # 我们可以临时解算一次 Pose，或者用零初始化一个 dummy token。
        # 简单起见，我们在这里解算一次作为初始 token 状态。
        with torch.no_grad(): # 初始化阶段不传梯度，节省显存
            pose_params_init = self.solver(
                xyz_i.reshape(B, 24, 3), 
                ortho_i.reshape(B, self.n_ortho_vecs, 3), 
                scalar_i
            ) # (B, 46)
            # 将 46维 Euler 转为 6D (24*6=144) 以适配 Embedding 层
            # 注意: 这里需要一个 utility 将 46维转 144维，或者修改 self.to_pose_embedding
            # 方案 B: 修改 to_pose_embedding 的输入维度为 46+3 (更简单) -> 见下方注
        
        # 暂且保留原逻辑：假设 pose_params_init 是某种表示。
        # 我们可以用 zeros 替代初始 pose token，让网络自己学。
        pose_token_input = torch.zeros(B, 1, 24*6 + 3, device=x.device, dtype=x.dtype) 
        # (为了不大幅改动 Transformer 输入结构，这里做个简化占位，或者你需要修改 self.to_pose_embedding)

        betas_i = betas_init
        cam_i = cam_init
     
        # --- 2. Transformer Loop ---
        context_list = [x_norm_patch] * len(self.layers)        

        # 构建 Tokens (简化版，复用原逻辑)
        # 注意：这里如果 pose_token_input 不包含真实信息，Transformer 第一层只能靠 Patch Context
        poses_token = self.to_pose_embedding(pose_token_input) 
        betas_token = torch.cat([betas_init, bbox_info], dim=-1)[:, None,:]
        betas_token = self.to_betas_embedding(betas_token)
        cam_token = torch.cat([cam_init, bbox_info], dim=-1)[:, None,:]
        cam_token = self.to_cam_embedding(cam_token)

        # B, 3, dim
        token = torch.cat([poses_token, betas_token, cam_token], dim=1)
        b, n, _ = token.shape
        token = self.dropout(token)
        token = token + self.pos_embedding[:, :n]
        
        per_layer_params = []
        
        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            # 1. Update Tokens
            token = self_attn(token) + token
            token = cross_attn(token, context=context_list[i]) + token
            token = ff(token) + token

            # 2. [NEW] Update Geometric Features (Residual Learning)
            # 预测残差并叠加
            xyz_i    = xyz_i    + self.xyz_decoder(token[:, 0])
            ortho_i  = ortho_i  + self.ortho_decoder(token[:, 0]) # 复用 token[0] (Pose Token)
            scalar_i = scalar_i + self.scalar_decoder(token[:, 0])
            
            # Betas & Cam
            betas_i = betas_i + self.betas_decoder(token[:, 1])
            cam_i   = cam_i   + self.cam_decoder(token[:, 2])

            # 3. Solve Pose for this layer (Deep Supervision)
            # 将扁平的向量 reshape 传给 solver
            cur_kp3d = xyz_i.reshape(B, 24, 3)
            cur_ortho = ortho_i.reshape(B, self.n_ortho_vecs, 3)
            
            # 调用 Solver 解算
            solved_poses = self.solver(cur_kp3d, cur_ortho, scalar_i) # (B, 46)

            per_layer_params.append({
                f'pd_poses_{i}' : solved_poses, # 用于 L_param
                f'pd_betas_{i}' : betas_i,
                f'pd_cam_{i}'   : cam_i,
                # [Optional] 如果想对每一层加 L_swing/L_twist，可在此添加 raw 输出
                f'raw_kp3d_{i}' : cur_kp3d,
                f'raw_ortho_{i}': cur_ortho
            })

            # 4. Update Token for next layer (Optional)
            # 如果希望 Transformer 感知到当前的 Pose 变化，应更新 token
            # 这里简化处理，保持 token 的隐式更新
        
        # --- 3. Final Output ---
        
        # Final Solve
        final_kp3d = xyz_i.reshape(B, 24, 3)
        final_ortho = ortho_i.reshape(B, self.n_ortho_vecs, 3)
        final_solved_poses = self.solver(final_kp3d, final_ortho, scalar_i)

        # 整理输出字典
        return {
            # 编码器初始猜测 (这里用第一层的解算结果代替，或保留 init)
            'pd_enc_poses' : per_layer_params[0]['pd_poses_0'], 
            'pd_enc_betas' : betas_init,
            'pd_enc_cam'   : cam_init,
            
            # 解码器最终输出 (驱动 SKEL)
            'pd_dec_poses' : final_solved_poses,
            'pd_dec_betas' : betas_i,
            'pd_dec_cam'   : cam_i, 
            
            # [NEW] 原始几何特征 (用于 Loss: L_swing, L_twist)
            'raw_kp3d'     : final_kp3d,
            'raw_ortho'    : final_ortho,
            'raw_scalar'   : scalar_i,
            
            'per_layer_params' : per_layer_params
        }