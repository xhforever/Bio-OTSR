#!/bin/bash
# 只提取特征图的demo版本（不需要OpenGL/渲染库）
# 可以在CPU或GPU上运行

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skelvit

# 如果要强制使用CPU，取消下面这行的注释
# export CUDA_VISIBLE_DEVICES=""

PYTHONPATH=. python vis/run_demo_features_only.py \
              misc.image_folder='demo_images' \
              misc.output_folder='demo_output/features-only/2gpu-freeze-encoder-5' \
              trainer.ckpt_path='/data/yangxianghao/SKEL-CF/data_outputs/exp/2gpu-freeze-encoder-5/checkpoints/best.pth'

