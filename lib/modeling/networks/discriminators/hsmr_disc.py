from lib.kits.basic import *


class HSMRDiscriminator(nn.Module):
    """
    HSMR判别器网络
    
    这是一个基于HMR论文提出的姿态+形状判别器，用于判断输入的人体姿态和形状参数是否真实。
    判别器包含三个分支：
    1. poses_alone: 单独对每个关节姿态进行判别
    2. betas: 对形状参数进行判别  
    3. poses_joint: 对所有关节姿态联合进行判别
    """

    def __init__(self):
        '''
        初始化HSMR判别器
        
        构建三个判别分支：
        - 姿态单独判别分支：使用1D卷积处理每个关节的旋转矩阵
        - 形状参数判别分支：使用全连接层处理SMPL的beta参数
        - 姿态联合判别分支：使用全连接层处理所有关节的联合特征
        '''
        super(HSMRDiscriminator, self).__init__()

        # SMPL身体姿态关节数量（不包括全局旋转）
        self.n_poses = 23
        
        # ==================== 姿态单独判别分支 ====================
        # 第一层1D卷积：将9维旋转矩阵特征映射到32维
        self.D_conv1 = nn.Conv2d(9, 32, kernel_size=1)
        nn.init.xavier_uniform_(self.D_conv1.weight)  # Xavier初始化权重
        nn.init.zeros_(self.D_conv1.bias)            # 偏置初始化为0
        
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        
        # 第二层1D卷积：保持32维特征
        self.D_conv2 = nn.Conv2d(32, 32, kernel_size=1)
        nn.init.xavier_uniform_(self.D_conv2.weight)
        nn.init.zeros_(self.D_conv2.bias)
        
        # 为每个关节创建独立的输出层（23个关节，每个输出1维判别结果）
        pose_out = []
        for i in range(self.n_poses):
            pose_out_temp = nn.Linear(32, 1)  # 32维特征 -> 1维判别结果
            nn.init.xavier_uniform_(pose_out_temp.weight)
            nn.init.zeros_(pose_out_temp.bias)
            pose_out.append(pose_out_temp)
        self.pose_out = nn.ModuleList(pose_out)  # 将所有输出层组织成ModuleList

        # ==================== 形状参数判别分支 ====================
        # SMPL形状参数beta的判别网络（beta通常为10维）
        self.betas_fc1 = nn.Linear(10, 10)  # 第一层全连接：10维 -> 10维
        nn.init.xavier_uniform_(self.betas_fc1.weight)
        nn.init.zeros_(self.betas_fc1.bias)
        
        self.betas_fc2 = nn.Linear(10, 5)   # 第二层全连接：10维 -> 5维
        nn.init.xavier_uniform_(self.betas_fc2.weight)
        nn.init.zeros_(self.betas_fc2.bias)
        
        self.betas_out = nn.Linear(5, 1)    # 输出层：5维 -> 1维判别结果
        nn.init.xavier_uniform_(self.betas_out.weight)
        nn.init.zeros_(self.betas_out.bias)

        # ==================== 姿态联合判别分支 ====================
        # 将所有关节的特征联合起来进行判别（32维特征 × 23个关节 = 736维）
        self.D_alljoints_fc1 = nn.Linear(32*self.n_poses, 1024)  # 第一层：736维 -> 1024维
        nn.init.xavier_uniform_(self.D_alljoints_fc1.weight)
        nn.init.zeros_(self.D_alljoints_fc1.bias)
        
        self.D_alljoints_fc2 = nn.Linear(1024, 1024)  # 第二层：1024维 -> 1024维
        nn.init.xavier_uniform_(self.D_alljoints_fc2.weight)
        nn.init.zeros_(self.D_alljoints_fc2.bias)
        
        self.D_alljoints_out = nn.Linear(1024, 1)     # 输出层：1024维 -> 1维判别结果
        nn.init.xavier_uniform_(self.D_alljoints_out.weight)
        nn.init.zeros_(self.D_alljoints_out.bias)


    def forward(self, poses_body: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        '''
        判别器前向传播过程
        
        ### 输入参数
        - poses_body: torch.Tensor, 形状为 (B, 23, 9) 
            - SMPL身体姿态参数的旋转矩阵表示（不包括全局旋转）
            - B: 批次大小, 23: 关节数量, 9: 3x3旋转矩阵展平
        - betas: torch.Tensor, 形状为 (B, 10)
            - SMPL形状参数，控制人体体型
            
        ### 返回值
        - torch.Tensor, 形状为 (B, 25)
            - 判别结果：23个关节独立判别 + 1个形状判别 + 1个联合姿态判别 = 25维
        '''
        # 重塑姿态张量以适应卷积层输入格式
        poses_body = poses_body.reshape(-1, self.n_poses, 1, 9)  # (B, 23, 1, 9)
        B = poses_body.shape[0]  # 获取批次大小
        poses_body = poses_body.permute(0, 3, 1, 2).contiguous()  # (B, 9, 23, 1) - 通道维度放在第二位

        # ==================== 姿态单独判别分支处理 ====================
        # 通过两层1D卷积提取每个关节的特征
        poses_body = self.D_conv1(poses_body)  # (B, 32, 23, 1)
        poses_body = self.relu(poses_body)
        poses_body = self.D_conv2(poses_body)  # (B, 32, 23, 1)
        poses_body = self.relu(poses_body)

        # 对每个关节独立进行判别
        poses_out = []
        for i in range(self.n_poses):
            # 提取第i个关节的32维特征并通过对应的输出层
            poses_out_i = self.pose_out[i](poses_body[:, :, i, 0])  # (B, 1)
            poses_out.append(poses_out_i)
        poses_out = torch.cat(poses_out, dim=1)  # (B, 23) - 23个关节的判别结果

        # ==================== 形状参数判别分支处理 ====================
        # 通过三层全连接网络处理SMPL形状参数
        betas = self.betas_fc1(betas)    # (B, 10) -> (B, 10)
        betas = self.relu(betas)
        betas = self.betas_fc2(betas)    # (B, 10) -> (B, 5)
        betas = self.relu(betas)
        betas_out = self.betas_out(betas)  # (B, 5) -> (B, 1) - 形状参数判别结果

        # ==================== 姿态联合判别分支处理 ====================
        # 将所有关节的特征展平并联合判别
        poses_body = poses_body.reshape(B, -1)  # (B, 32*23) = (B, 736) - 展平所有关节特征
        poses_all = self.D_alljoints_fc1(poses_body)  # (B, 736) -> (B, 1024)
        poses_all = self.relu(poses_all)
        poses_all = self.D_alljoints_fc2(poses_all)   # (B, 1024) -> (B, 1024)
        poses_all = self.relu(poses_all)
        poses_all_out = self.D_alljoints_out(poses_all)  # (B, 1024) -> (B, 1) - 联合姿态判别结果

        # ==================== 合并所有判别结果 ====================
        # 拼接三个分支的输出：23个关节判别 + 1个形状判别 + 1个联合姿态判别 = 25维
        disc_out = torch.cat((poses_out, betas_out, poses_all_out), dim=1)  # (B, 25)
        return disc_out
