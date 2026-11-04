#!/usr/bin/env bash
# Fine-tune the ReCogDrive IL checkpoint on ReactTrain (IL + RT, paper section 5): the planner's stage-2
# denoising objective on the kinematic targets, VLM frozen, 3 epochs = 276 updates of 384 samples
# (24 per GPU x GPUS x 16/GPUS accumulation), lr 3e-5 with 25 warm-up updates, seed 0. The paper's run took
# 1.5 h on one A100; its checkpoint is epoch=2-step=276.ckpt.
#   GPUS=1 scripts/reacttrain/train.sh [extra Hydra overrides]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EXP="$REACTTRAIN_WORK"; VIEW="$EXP/recogdrive_kinematic_sft/navtrain_selected_t1234_pool35226_v1"
GPUS="${GPUS:-1}"; (( 16 % GPUS == 0 )) || { echo "GPUS must divide 16" >&2; exit 2; }
IL="$CHECKPOINTS/recogdrive/IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt"
OUT="${OUTPUT_DIR:-$EXP/training/il_rt_seed0}"
export NAVSIM_EXP_ROOT="$EXP"
exec torchrun --nproc_per_node="$GPUS" --master_port="${MASTER_PORT:-29523}" \
    "$FORK/navsim/planning/script/run_training_recogdrive_rl.py" \
    agent=recogdrive_agent agent.lr=3e-5 agent.velacc_dropout_probability=0.0 agent.joint_motion_dropout_probability=0.0 \
    agent.bc_coeff=0 agent.planner_initialization=stage2 agent.training_comfort_weight=2.0 agent.training_ep_weight=10.0 \
    agent.training_reward_mode=executed agent.training_raw_safety_weight=0.2 agent.training_raw_contact_speed_weight=0.05 \
    agent.training_raw_safety_inputs=null seed=0 agent.lr_scheduler_epochs=25 "agent.vlm_path=$CHECKPOINTS/recogdrive/VLM" \
    agent.cam_type=single agent.grpo=false agent.cache_hidden_state=true agent.vlm_type=internvl \
    "agent.checkpoint_path=$IL" agent.dit_type=small agent.vlm_size=small agent.sampling_method=ddim \
    "agent.metric_cache_path=$VIEW/metrics" agent.pdm_reward_workers=1 "agent.reference_policy_checkpoint=$IL" \
    trainer.params.max_epochs=3 trainer.params.num_nodes=1 "trainer.params.devices=$GPUS" \
    "trainer.params.accumulate_grad_batches=$((16 / GPUS))" trainer.params.limit_train_batches=1.0 \
    +trainer.params.log_every_n_steps=5 dataloader.params.batch_size=24 +dataloader.params.drop_last=false \
    experiment_name=reacttrain "output_dir=$OUT" train_test_split=navtrain "cache_path=$VIEW/features" \
    use_cache_without_dataset=true force_cache_computation=false +expected_train_samples=58710 \
    +expected_validation_samples=18179 "mixed_training_manifest='$VIEW/manifest.json'" +trainer.params.max_steps=276 \
    +expected_optimizer_steps_per_epoch=92 +expected_total_optimizer_steps=276 lr_warmup_updates=25 \
    agent.per_sample_loss=true hazard_sft_originals_per_epoch=11742 run_label=reacttrain checkpoint_selection=all_epochs \
    fixed_update_checkpoint_interval=0 "$@"
