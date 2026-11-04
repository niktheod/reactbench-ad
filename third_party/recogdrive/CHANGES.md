# Changes from upstream ReCogDrive

Base: upstream commit `37cd7b2`. Files not listed below are unchanged.

## Removed
Everything the release does not use: the other NAVSIM agents (TransFuser, constant velocity, ego MLP,
human, InternVL/Qwen text agents), `internvl_chat/` (VLM pre-training), submission and visualization
scripts, dataset-generation and download scripts, shell/Slurm launchers (the release has its own under
`scripts/`), docs and tests. Configs are reduced to the agent, splits, workers, metric caching, PDM
scoring and training configs that the release runs.

## Modified
- `navsim/agents/recogdrive/`: agent, backbone, DiT, features and diffusion planner. Per-token seeded
  diffusion sampling, so that the original and edited scene of a ReactBench pair share all diffusion
  noise; without a seed, sampling draws from the global RNG as upstream does. Stage-2 planner
  initialization from the IL checkpoint and per-sample stage-2 losses for the ReactTrain fine-tuning.
  Development options that the paper does not use remain available and are off by default: GRPO
  reward variants (`raw`, `minimum_jerk`, speed siblings), velocity/acceleration and joint motion
  dropout, and KV-cache conditioning of the planner. The raw-safety reward variant was removed.
- `navsim/planning/training/agent_lightning_module.py`, `navsim/planning/script/run_training_recogdrive_rl.py`
  and `config/training/{training_base,recogdrive_stage3}.yaml`: the training entry point of the
  ReactTrain fine-tuning (mixed original/edited training views, the kinematic stop targets, an
  epoch-wise original sampler, linear LR warm-up before the per-update cosine schedule, every-epoch
  checkpoints, optimizer-clock checks). One validation forward per batch (upstream ran a discarded
  extra one). The paired edited-scene monitor used during development is optional and off by default;
  it runs on its own RNG state and does not change training.
- `navsim/evaluate/pdm_score.py`: `pdm_scores`, the PDMS of several candidate trajectories of one scene
  in one batch (equal to repeated `pdm_score` calls), used by the training rewards and the probe.
- `navsim/common/dataloader.py`: optional node-local staging of metric-cache files.
- `run_pdm_score_recogdrive.py`: `+inference_seed`, a per-token seed for the navtest evaluation.
- `config/pdm_scoring/default_run_pdm_score.yaml`, `config/training/default_training.yaml` and
  `config/common/agent/recogdrive_agent.yaml`: ReCogDrive as the default agent and the agent options above.

## Added
- ReactTrain fine-tuning: `navsim/planning/training/recogdrive_{kinematic_sft,mixed_data,mixed_training,hazard_sft}.py`
  (stop targets, training views, samplers), `scripts/cache_dataset/` (VLM features and metric caches of the
  edited samples, cache validators, targets, views) and `navsim/planning/metric_caching/edited_scenario.py`
  (metric caches with the inserted actor as a tracked object).
- `navsim/evaluate/minimum_jerk_trajectory.py`: path reconstruction shared by the stop targets.
- `navsim/evaluate/{minimum_jerk_pdm_score,raw_pdm_score,sibling_targets}.py`: the reward variants above.
- `navsim/planning/training/recogdrive_{validation,paired_validation,triplet_validation,checkpoint_selection}.py`
  and `callbacks/recogdrive_validation_callback.py`: the optional training monitors.
- `metadata/vlm_artifacts.sha256`: SHA-256 of the released VLM files, checked before caching features.
