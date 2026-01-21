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

# [续训配置] 从last_step.pth恢复训练
# 关键修改：
# 1. resume_training=True - 恢复optimizer和scheduler状态
# 2. max_lr降低到1e-5 - 解决2D loss过高问题
# 3. warmup_epochs=2 - 更平滑的训练
torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} run_train.py \
            trainer.freeze_encoder=True \
            trainer.batch_size=320 \
            trainer.test_batch_size=64 \
            exp_name=2gpu-freeze-encoder-5 \
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
            +resume_training=True \
            +resume_scheduler=True \
            pretrained_ckpt="/data/yangxianghao/SKEL-CF/data_outputs/exp/2gpu-freeze-encoder-5/checkpoints/last_step.pth"

