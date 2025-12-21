# SKEL-CF 训练模型流程图

## 📊 完整训练流程

```mermaid
flowchart TB
    Start([开始训练]) --> LoadConfig[加载配置文件 Hydra]
    LoadConfig --> InitDist[初始化分布式环境 DDP]
    InitDist --> BuildModel[构建模型 SKELViT]
    
    BuildModel --> ModelComponents{模型组件初始化}
    
    ModelComponents --> Backbone[ViT-H Backbone<br/>ViTPose预训练]
    ModelComponents --> Decoder[SKEL Transformer Decoder<br/>6层解码器]
    ModelComponents --> CamModel[Camera Model FLNet<br/>冻结参数]
    ModelComponents --> SKELModel[SKEL Body Model<br/>参数化人体模型]
    
    Backbone --> ModelReady[模型准备完成]
    Decoder --> ModelReady
    CamModel --> ModelReady
    SKELModel --> ModelReady
    
    ModelReady --> BuildDataset[构建训练数据集]
    BuildDataset --> DataLoader[创建 DataLoader<br/>分布式采样器]
    
    DataLoader --> BuildOptim[构建优化器 AdamW]
    BuildOptim --> BuildLoss[构建损失函数 HPE_Loss]
    BuildLoss --> BuildEMA[构建 EMA 模型]
    
    BuildEMA --> TrainLoop{训练循环<br/>Epoch Loop}
    
    TrainLoop --> |每个Epoch| LoadBatch[加载 Batch 数据]
    
    LoadBatch --> Forward[前向传播]
    
    Forward --> SubForward{详细前向过程}
    
    style SubForward fill:#e1f5ff
```

## 🔄 前向传播详细流程

```mermaid
flowchart TB
    Input["输入图像 Batch
    B×3×256×256"] --> Crop["裁剪图像
    B×3×256×192"]
    
    Crop --> ViTBackbone[ViT Backbone 特征提取]
    
    ViTBackbone --> PatchEmbed["Patch Embedding
    切分为 patches"]
    PatchEmbed --> AddPos["添加位置编码
    包含 CLS token"]
    AddPos --> TransLayers["12层 Transformer
    Self-Attention"]
    TransLayers --> BackboneOut["输出特征图
    B×192×1280"]
    
    BackboneOut --> SplitFeatures{特征分离}
    
    SplitFeatures --> GlobalFeat["全局特征 x_cls
    mean pooling
    B×1280"]
    SplitFeatures --> SpatialFeat["空间特征 x_norm_patch
    保留所有 tokens
    B×192×1280"]
    
    GlobalFeat --> InitPredict[初始参数预测]
    InitPredict --> InitPoses["初始姿态
    poses_init: B×144"]
    InitPredict --> InitBetas["初始形状
    betas_init: B×10"]
    InitPredict --> InitCam["初始相机
    cam_init: B×3"]
    
    InitPoses --> DecoderInput[Decoder 输入准备]
    InitBetas --> DecoderInput
    InitCam --> DecoderInput
    SpatialFeat --> DecoderContext["Context for
    Cross-Attention"]
    
    DecoderInput --> TokenEmbed["Token Embedding
    poses + betas + cam
    + bbox_info"]
    
    TokenEmbed --> TransDecoder{"Transformer Decoder
    6层迭代精化"}
    DecoderContext --> TransDecoder
    
    TransDecoder --> Layer1[Layer 1: Self-Attn + Cross-Attn + FFN]
    Layer1 --> Update1["更新参数
    poses/betas/cam"]
    Update1 --> Layer2[Layer 2: 继续精化]
    Layer2 --> Update2[更新参数]
    Update2 --> Layer3[Layer 3-6: 持续精化...]
    Layer3 --> FinalParams[最终参数输出]
    
    FinalParams --> EncOutput["Encoder 输出
    pd_enc_poses/betas/cam"]
    FinalParams --> DecOutput["Decoder 输出
    pd_dec_poses/betas/cam"]
    
    EncOutput --> SKELWrapper1[SKEL Wrapper]
    DecOutput --> SKELWrapper2[SKEL Wrapper]
    
    SKELWrapper1 --> Enc3D["Encoder 预测
    3D关键点/2D投影/顶点"]
    SKELWrapper2 --> Dec3D["Decoder 预测
    3D关键点/2D投影/顶点"]
    
    Enc3D --> LossCalc[损失计算]
    Dec3D --> LossCalc
    
    style TransDecoder fill:#ffe1e1
    style SKELWrapper1 fill:#e1ffe1
    style SKELWrapper2 fill:#e1ffe1
```

## 🎯 损失计算流程

```mermaid
flowchart TB
    Predictions[模型预测输出] --> EncPred["Encoder 预测
    kp2d/kp3d/poses/betas"]
    Predictions --> DecPred["Decoder 预测
    kp2d/kp3d/poses/betas"]
    
    GroundTruth["Ground Truth 标签
    从数据集"] --> GTData["GT数据
    kp2d/kp3d/poses/betas"]
    
    EncPred --> LossEnc{Encoder 损失}
    DecPred --> LossDec{Decoder 损失}
    GTData --> LossEnc
    GTData --> LossDec
    
    LossEnc --> L2D_Enc["2D关键点损失
    Keypoint2DLoss"]
    LossEnc --> L3D_Enc["3D关键点损失
    Keypoint3DLoss"]
    LossEnc --> LBetas_Enc["形状参数损失
    ParameterLoss"]
    LossEnc --> LPoses_Enc["姿态损失
    Body + Orient"]
    
    LossDec --> L2D_Dec[2D关键点损失]
    LossDec --> L3D_Dec[3D关键点损失]
    LossDec --> LBetas_Dec[形状参数损失]
    LossDec --> LPoses_Dec[姿态损失]
    
    L2D_Enc --> WeightEnc["加权求和
    ×loss_weights"]
    L3D_Enc --> WeightEnc
    LBetas_Enc --> WeightEnc
    LPoses_Enc --> WeightEnc
    
    L2D_Dec --> WeightDec["加权求和
    ×loss_weights"]
    L3D_Dec --> WeightDec
    LBetas_Dec --> WeightDec
    LPoses_Dec --> WeightDec
    
    WeightEnc --> TotalEnc[Total Encoder Loss]
    WeightDec --> TotalDec[Total Decoder Loss]
    
    TotalEnc --> Combine["组合损失
    λ×Loss_enc + Loss_dec"]
    TotalDec --> Combine
    
    LayerOutputs[每层中间输出] --> AuxLoss["辅助损失
    Auxiliary Loss"]
    AuxLoss --> Combine
    
    Combine --> FinalLoss["最终总损失
    Total Loss"]
    
    style LossEnc fill:#fff4e1
    style LossDec fill:#fff4e1
    style FinalLoss fill:#ff9999
```

## ⚙️ 反向传播与优化流程

```mermaid
flowchart TB
    TotalLoss[总损失 Total Loss] --> AMP["混合精度训练
    AMP Scaler"]
    
    AMP --> ScaleLoss["缩放损失
    scaler.scale"]
    ScaleLoss --> Backward["反向传播
    loss.backward"]
    
    Backward --> Unscale["取消缩放
    scaler.unscale_"]
    Unscale --> ClipGrad["梯度裁剪
    max_norm=1.0"]
    
    ClipGrad --> OptimStep["优化器更新
    AdamW.step"]
    OptimStep --> ScalerUpdate["更新 Scaler
    scaler.update"]
    
    ScalerUpdate --> LRSchedule["学习率调度
    Warmup + Constant"]
    LRSchedule --> EMAUpdate["更新 EMA 模型
    指数移动平均"]
    
    EMAUpdate --> CheckLog{"是否记录日志?
    每 N 步"}
    
    CheckLog --> |是| SaveCheckpoint["保存检查点
    last_step.pth"]
    CheckLog --> |是| TensorBoard["记录到 TensorBoard
    损失/学习率"]
    CheckLog --> |否| NextBatch{下一个 Batch?}
    
    SaveCheckpoint --> NextBatch
    TensorBoard --> NextBatch
    
    NextBatch --> |继续| LoadNextBatch[加载下一批数据]
    NextBatch --> |Epoch结束| Evaluate[评估阶段]
    
    LoadNextBatch --> Forward[前向传播]
    
    style AMP fill:#e1f5ff
    style OptimStep fill:#ffe1e1
    style EMAUpdate fill:#e1ffe1
```

## 📊 评估流程

```mermaid
flowchart TB
    EpochEnd[Epoch 结束] --> EvalStart[开始评估]
    
    EvalStart --> EvalModel[评估主模型]
    EvalStart --> EvalEMA[评估 EMA 模型]
    
    EvalModel --> EvalCOCO1["HMR2 评估
    COCO数据集"]
    EvalModel --> EvalMOYO1[MOYO-HARD 评估]
    
    EvalEMA --> EvalCOCO2["HMR2 评估
    COCO数据集"]
    EvalEMA --> EvalMOYO2[MOYO-HARD 评估]
    
    EvalMOYO1 --> CalcMetrics1{计算指标}
    EvalMOYO2 --> CalcMetrics2{计算指标}
    
    CalcMetrics1 --> MPJPE1["MPJPE
    平均关节位置误差"]
    CalcMetrics1 --> PAMPJPE1["PA-MPJPE
    对齐后误差"]
    CalcMetrics1 --> PVE1["PVE
    顶点误差"]
    
    CalcMetrics2 --> MPJPE2[MPJPE]
    CalcMetrics2 --> PAMPJPE2[PA-MPJPE]
    CalcMetrics2 --> PVE2[PVE]
    
    MPJPE1 --> CompareMetrics[比较指标]
    PAMPJPE1 --> CompareMetrics
    PVE1 --> CompareMetrics
    MPJPE2 --> CompareMetrics
    PAMPJPE2 --> CompareMetrics
    PVE2 --> CompareMetrics
    
    CompareMetrics --> CheckBest{"是否最佳?
    PVE < best_pve"}
    
    CheckBest --> |是| SaveBest["保存最佳模型
    best.pth"]
    CheckBest --> |否| LogMetrics[记录指标]
    
    SaveBest --> LogMetrics
    LogMetrics --> NextEpoch{继续训练?}
    
    NextEpoch --> |是| NewEpoch[新 Epoch]
    NextEpoch --> |否| TrainEnd([训练结束])
    
    style EvalMOYO1 fill:#e1f5ff
    style EvalMOYO2 fill:#e1f5ff
    style SaveBest fill:#e1ffe1
```

## 🏗️ 模型架构详图

```mermaid
flowchart TB
    subgraph backbone["ViT-H Backbone"]
        Input1["输入图像: B×3×256×192"]
        PE1["Patch Embedding: 16×16 patches"]
        ViTLayers1["12层 ViT Blocks"]
        FeatOut1["特征输出: B×192×1280"]
        Input1 --> PE1 --> ViTLayers1 --> FeatOut1
    end
    
    subgraph decoder["SKEL Transformer Decoder"]
        InitToken1["初始 Tokens"]
        SelfAttn1["Self-Attention"]
        CrossAttn1["Cross-Attention"]
        FFN1["Feed Forward"]
        Layer2["6层迭代精化"]
        FinalOut1["最终参数"]
        InitToken1 --> SelfAttn1 --> CrossAttn1 --> FFN1 --> Layer2 --> FinalOut1
    end
    
    subgraph cammodel["Camera Model"]
        CamInput1["图像输入"]
        HRNet1["HRNet Backbone"]
        FocalLength1["预测焦距 Frozen"]
        CamInput1 --> HRNet1 --> FocalLength1
    end
    
    subgraph skelbody["SKEL Body Model"]
        Params1["poses + betas"]
        LBS1["Linear Blend Skinning"]
        Joints1["关节: 44个关节点"]
        Verts1["顶点: 6890个顶点"]
        Params1 --> LBS1
        LBS1 --> Joints1
        LBS1 --> Verts1
    end
    
    FeatOut1 --> decoder
    FocalLength1 --> decoder
    FinalOut1 --> skelbody
    
    style backbone fill:#e3f2fd
    style decoder fill:#fff3e0
    style cammodel fill:#f3e5f5
    style skelbody fill:#e8f5e9
```

## 🔢 数据流维度变化

```mermaid
flowchart LR
    A["图像
    B×3×256×256"] --> B["裁剪
    B×3×256×192"]
    B --> C["Patches
    B×192×1280"]
    C --> D["全局特征
    B×1280"]
    C --> E["空间特征
    B×192×1280"]
    
    D --> F["初始参数
    poses: B×144
    betas: B×10
    cam: B×3"]
    
    F --> G["Token
    B×3×1024"]
    E --> G
    
    G --> H["Decoder输出
    poses: B×46
    betas: B×10
    cam: B×3"]
    
    H --> I["SKEL输出
    joints: B×44×3
    verts: B×6890×3
    kp2d: B×44×2"]
    
    style A fill:#ffebee
    style C fill:#e3f2fd
    style F fill:#fff3e0
    style I fill:#e8f5e9
```

## 📈 训练参数配置

```mermaid
flowchart LR
    subgraph optimizer["优化器配置"]
        opt1["AdamW"]
        opt2["学习率: 1e-4"]
        opt3["权重衰减: 1e-4"]
    end
    
    subgraph scheduler["学习率调度"]
        sch1["Warmup Epochs"]
        sch2["Constant LR"]
    end
    
    subgraph amp["混合精度训练"]
        amp1["AMP"]
        amp2["GradScaler"]
    end
    
    subgraph dist["分布式训练"]
        dist1["DDP"]
        dist2["多GPU支持"]
    end
    
    subgraph loss["损失权重"]
        loss1["kp2d: 5.0"]
        loss2["kp3d: 5.0"]
        loss3["betas: 0.01"]
        loss4["poses: 1.0"]
    end
    
    subgraph augment["数据增强"]
        aug1["Random Flip"]
        aug2["Color Jitter"]
        aug3["Random Crop"]
    end
    
    subgraph batch["批处理"]
        bat1["Batch Size: 32"]
        bat2["Num Workers: 8"]
    end
    
    subgraph ema["EMA模型"]
        ema1["Decay: 0.999"]
        ema2["指数移动平均"]
    end
```

## 🎓 关键设计理念

```mermaid
flowchart TD
    Design[SKEL-CF 设计理念]
    
    Design --> Enc["编码器-解码器
    Encoder-Decoder"]
    Design --> Iter["迭代精化
    Iterative Refinement"]
    Design --> Multi["多尺度特征
    Multi-scale Features"]
    Design --> Aux["辅助监督
    Auxiliary Supervision"]
    
    Enc --> EncDesc["ViT编码全局语义
    Decoder迭代优化"]
    Iter --> IterDesc["6层逐步精化参数
    每层残差更新"]
    Multi --> MultiDesc["全局特征初始化
    空间特征提供细节"]
    Aux --> AuxDesc["每层输出都计算损失
    加速收敛"]
    
    style Design fill:#ff9999
    style Enc fill:#99ccff
    style Iter fill:#99ff99
    style Multi fill:#ffcc99
    style Aux fill:#cc99ff
```

---

## 📝 说明

以上流程图完整展示了 SKEL-CF 训练模型的：
- ✅ 完整训练循环
- ✅ 前向传播细节
- ✅ 损失计算机制
- ✅ 反向传播与优化
- ✅ 评估流程
- ✅ 模型架构
- ✅ 数据维度变化
- ✅ 核心设计理念

可以使用支持 Mermaid 的 Markdown 查看器（如 Typora、VS Code、GitHub）来渲染这些流程图。
