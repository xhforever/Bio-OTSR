set -euo pipefail
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2,3
export OMP_NUM_THREADS=4

# 自动选择可用端口
MASTER_PORT=${MASTER_PORT:-$(python - <<'PY'
import socket
s=socket.socket(); s.bind(('',0))
port=s.getsockname()[1]; s.close()
print(port)
PY
)}

# [重新开始] 从best.pth重新训练
# 用更合适的学习率，避免2D loss爆炸
torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} run_train.py \
            trainer.freeze_encoder=True \
            trainer.batch_size=320 \
            trainer.test_batch_size=64 \
            exp_name=2gpu-freeze-encoder-5-restart \
            trainer.max_lr=1e-5 \
            trainer.min_lr=1e-6 \
            trainer.warmup_epochs=2 \
            trainer.total_epochs=30 \
            general.num_workers=4 \
            logger.logger_interval=50 \
            trainer.ema_decaybase=0.9995 \
            hub.skel_head.transformer_decoder.context_dim=1280 \
            trainer.ema_tau=4000 \
            policy.img_patch_size=256 \
            trainer.lamda=1. \
            misc.aux_loss_weight=0.1 \
            +resume_training=False \
            pretrained_ckpt="/data/yangxianghao/SKEL-CF/data_outputs/exp/2gpu-freeze-encoder-5/checkpoints/best.pth"

