set -x

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
export LIBERO_DATASETS_ROOT="${LIBERO_DATASETS_ROOT:-/root/autodl-tmp/datasets/LIBERO}"
export LIBERO_ROOT="${LIBERO_ROOT:-/root/2604-VLA_RL_offline/SimpleVLA_RL_Offline-main/LIBERO}"
export PYTHONPATH="/root/2604-VLA_RL_offline/SimpleVLA_RL_Offline-main:${LIBERO_ROOT}:${PYTHONPATH:-}"
export ROBOT_PLATFORM=LIBERO
export RUNTIME_ENV_PATH="${RUNTIME_ENV_PATH:-src/.runtime_env.libero.json}"

PROJECT_NAME="${PROJECT_NAME:-Offline-VLA-RL}"               # 项目名
EXPERIMENT_NAME="${EXPERIMENT_NAME:-openvla_oft_libero_10_rl_2gpu_grpo}"    # 实验名
# For openvla-oft Libero-Long traj1 SFT or traj all SFT models can be find in https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86
SFT_MODEL_PATH="${SFT_MODEL_PATH:-/root/autodl-tmp/models/simplevla/Openvla-oft-SFT-libero10-trajall}"   # 模型路径
CKPT_PATH="${CKPT_PATH:-/root/autodl-tmp/chkpt/2604-VLA_RL_offline}"              # ckpt保存路径
# DATASET_NAME can be libero_10 (libero_Long), libero_90, libero_spatial, libero_object, libero_goal
DATASET_NAME="${DATASET_NAME:-libero_10}"        # libero数据集选择
VLA_NAME="${VLA_NAME:-openvla-oft}"                  # 所依赖VLA模型名称
NUM_GPUS="${NUM_GPUS:-2}"                            # 使用的GPU数量
# If you want to use multi-node RL, set NUM_NODES accordingly.
NUM_NODES="${NUM_NODES:-1}"                          # 使用的节点数量
WANDB_MODE="${WANDB_MODE:-disabled}"                 # wandb模式，disabled表示不使用wandb
ALIGN_PATH=".${ALIGN_PATH:-/align.json}"

# Configure LIBERO dataset/code paths before training.
python3 scripts/setup_libero_config.py \
    --datasets-root "$LIBERO_DATASETS_ROOT" \
    --libero-repo "$LIBERO_ROOT" \
    --config-path "$LIBERO_CONFIG_PATH"

bash examples/overwrite_vla_ckpt_utils.sh $SFT_MODEL_PATH

export TENSORBOARD_DIR="tensorboard_log/$PROJECT_NAME/$EXPERIMENT_NAME"

mkdir -p "$CKPT_PATH"

HYDRA_FULL_ERROR=1 python -u -m src.main_ppo \
    actor_rollout_ref.rollout.rollout_w_demo=False \
    data.n_samples=10 \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=50 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=250 \
    data.val_batch_size=64 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=500 \
    actor_rollout_ref.actor.ppo_micro_batch_size=20 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=8 \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=5 \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.model_family=openvla-oft \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$RUNTIME_ENV_PATH \
    trainer.wandb_mode=$WANDB_MODE \
    trainer.val_before_train=False \
