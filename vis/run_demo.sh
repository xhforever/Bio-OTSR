#!/bin/bash
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/data/yangxianghao/SKEL-CF/10_nvidia.json

CUDA_DEVICE=3
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
export EGL_DEVICE_ID=$CUDA_DEVICE

PYTHONPATH=. python vis/run_demo.py \
              trainer.ckpt_path='/data/yangxianghao/SKEL-CF/data_outputs/exp/SKEL-CF.pth' \
              misc.image_folder='demo_images'\
              misc.output_folder='demo_output'











    