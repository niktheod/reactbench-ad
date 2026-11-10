#!/usr/bin/env bash
# Fine-tune the ReCogDrive IL checkpoint on ReactTrain (IL + RT, paper section 5): the planner's stage-2
# denoising objective on the kinematic targets, VLM frozen, 3 epochs = 276 updates of 384 samples
# (24 per GPU x GPUS x 16/GPUS accumulation), lr 3e-5 with 25 warm-up updates, seed 0. The paper's
# checkpoint is epoch=2-step=276.ckpt.
#   GPUS=1 scripts/reacttrain/train.sh [extra Hydra overrides]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EXP="$REACTTRAIN_WORK"; VIEW="$EXP/recogdrive_kinematic_sft/navtrain_selected_t1234_pool35226_v1"
GPUS="${GPUS:-1}"; (( 16 % GPUS == 0 )) || { echo "GPUS must divide 16" >&2; exit 2; }
IL="$CHECKPOINTS/recogdrive/IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt"
OUT="${OUTPUT_DIR:-$EXP/training/il_rt_seed0}"
export NAVSIM_EXP_ROOT="$EXP"
exec "$PYTHON" -m torch.distributed.run --nproc_per_node="$GPUS" --master_port="${MASTER_PORT:-29523}" \
    "$FORK/navsim/planning/script/run_training_recogdrive_rl.py" \
    agent=recogdrive_agent agent.lr=3e-5 seed=0 "agent.vlm_path=$CHECKPOINTS/recogdrive/VLM" \
    agent.cam_type=single agent.grpo=false agent.cache_hidden_state=true agent.vlm_type=internvl \
    "agent.checkpoint_path=$IL" agent.dit_type=small agent.vlm_size=small agent.sampling_method=ddim \
    "agent.metric_cache_path=$VIEW/metrics" \
    trainer.params.max_epochs=3 trainer.params.num_nodes=1 "trainer.params.devices=$GPUS" \
    "trainer.params.accumulate_grad_batches=$((16 / GPUS))" trainer.params.limit_train_batches=1.0 \
    +trainer.params.log_every_n_steps=5 dataloader.params.batch_size=24 +dataloader.params.drop_last=false \
    experiment_name=reacttrain "output_dir=$OUT" train_test_split=navtrain "cache_path=$VIEW/features" \
    use_cache_without_dataset=true force_cache_computation=false \
    "mixed_training_manifest='$VIEW/manifest.json'" +trainer.params.max_steps=276 lr_warmup_updates=25 \
    agent.per_sample_loss=true kinematic_sft_originals_per_epoch=11742 run_label=reacttrain "$@"
