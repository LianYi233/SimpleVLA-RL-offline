#!/bin/bash
set -x

GOLBAL_STEP=19
model_path=/root/autodl-tmp/checkpoints/VLA-RL/openvla_oft_libero10_rl_2gpu_mix_policy_resnet_align-gamma0.03

bash examples/overwrite_vla_ckpt_utils.sh "$model_path/actor/global_step_$GOLBAL_STEP"

python -m scripts.model_merger merge \
    --backend fsdp \
    --local_dir "$model_path/actor/global_step_$GOLBAL_STEP" \
    --target_dir "$model_path/hf_model/global_step_$GOLBAL_STEP"