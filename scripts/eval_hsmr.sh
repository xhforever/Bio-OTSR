# export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
PYTHONPATH=. torchrun   run_hsmr_test.py \
                        trainer.test_batch_size=5 \
                        general.num_workers=8 \
                        hub.skel_head.transformer_decoder.context_dim=1280 \
                        policy.img_patch_size=256 \
                        trainer.ckpt_path='data_outputs/exp/1226-TRAIN-p40-4gpu-finetune-h36m-coco/checkpoints/last_step.pth'\
                        exp_name='1226-TRAIN-p40-4gpu-finetune-h36m-coco'



                                        