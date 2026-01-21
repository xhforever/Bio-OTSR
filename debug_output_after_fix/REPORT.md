# Bio-OTSR修复测试报告

## 测试信息
- 测试时间: 2026-01-20 10:16:19
- 使用模型: data_outputs/exp/2gpu-freeze-encoder-5/checkpoints/best.pth
- 修复内容: BUG #1 (basis_matrix索引) + BUG #2 (TYPE_A关节映射)

## 修复详情

### BUG #1: Basis Matrix索引
- 文件: lib/body_models/skel/otsr_solver.py:240
- 修复: a_parent_idx → a_child_idx

### BUG #2: TYPE_A关节索引
- 文件: lib/body_models/skel/kin_skel.py:217-220
- 修复: 
  - femur_r: (2,1) → (1,0)  # pelvis → femur_r
  - femur_l: (7,6) → (6,0)  # pelvis → femur_l
  - humerus_r: (16,15) → (15,12)  # thorax → humerus_r
  - humerus_l: (21,20) → (20,12)  # thorax → humerus_l

## 测试结果

### 定性评估
- [ ] 骨架是否与人体轮廓对齐？
- [ ] 四肢关节角度是否自然？
- [ ] 左右对称部位是否一致？

### 定量评估（如适用）
- MPJPE: ___ mm
- PA-MPJPE: ___ mm
- 与修复前对比: ___

## 下一步建议

根据上述结果，选择以下方案之一：

### ✅ 方案1: 效果良好，直接使用
如果骨架已经基本对齐，无需额外训练。

### 🔄 方案2: 效果改善但不完美，微调
运行以下命令进行微调（5-10 epochs）：
```bash
python run_train.py \
    --resume data_outputs/exp/2gpu-freeze-encoder-5/checkpoints/best.pth \
    --lr 1e-5 \
    --epochs 10 \
    --output_dir data_outputs/exp/bioOTSR-fixed-finetune
```

### 🔁 方案3: 效果不理想，完全重训练
运行以下命令重新训练：
```bash
python run_train.py \
    --config config/your_config.yaml \
    --output_dir data_outputs/exp/bioOTSR-fixed-retrain
```

## 备注
- 原文件已备份为 .backup
- 如需回滚，删除修改并重命名备份文件
