set -euo pipefail
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2,3
export OMP_NUM_THREADS=4

# 自动选择可用端口（如已设置 MASTER_PORT 则沿用）
MASTER_PORT=${MASTER_PORT:-$(python - <<'PY'
import socket
s=socket.socket(); s.bind(('',0))
port=s.getsockname()[1]; s.close()
print(port)
PY
)}

# 四卡P40训练配置：冻结encoder，只训练decoder
torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} run_train.py \
            trainer.freeze_encoder=True \
            trainer.batch_size=320 \
            trainer.test_batch_size=64 \
            exp_name=2gpu-freeze-encoder-5 \
            trainer.max_lr=5e-5 \
            trainer.min_lr=5e-6 \
            trainer.total_epochs=30 \
            general.num_workers=4 \
            logger.logger_interval=50 \
            trainer.ema_decaybase=0.9995 \
            hub.skel_head.transformer_decoder.context_dim=1280 \
            trainer.ema_tau=4000 \
            policy.img_patch_size=256 \
            trainer.lamda=1. \
            misc.aux_loss_weight=0.1\
            +resume_training=False \
            pretrained_ckpt="/data/yangxianghao/SKEL-CF/data_outputs/exp/SKEL-CF.pth"
            