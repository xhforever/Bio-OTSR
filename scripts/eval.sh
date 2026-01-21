# export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2,3

# 自动寻找空闲端口
MASTER_PORT=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)
export MASTER_PORT
echo "Using MASTER_PORT=${MASTER_PORT}"

# 多卡评测使用 torchrun
torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} run_test.py \
                            general.num_workers=8 \
                            hub.skel_head.transformer_decoder.context_dim=1280 \
                            policy.img_patch_size=256 \
                            trainer.test_batch_size=128 \
                            DATASETS.VAL_DATASETS=3dpw-smpl \
                            exp_name='freeze-encoder-3' \
                            trainer.ckpt_path='/data/yangxianghao/SKEL-CF/data_outputs/exp/4gpu-freeze-encoder-3/checkpoints/last_step.pth' 
                                        


                                        