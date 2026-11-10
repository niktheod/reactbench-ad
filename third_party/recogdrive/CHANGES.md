# Changes from upstream ReCogDrive

Base: upstream commit `37cd7b2`. Files not listed below are unchanged.

## Removed
Everything the release does not use: the other NAVSIM agents (TransFuser, constant velocity, ego MLP,
human, InternVL/Qwen text agents), `internvl_chat/` (VLM pre-training), submission and visualization
scripts, dataset-generation and download scripts, shell/Slurm launchers (the release has its own under
`scripts/`), docs and tests. Configs are reduced to the agent, splits, workers, metric caching, PDM
scoring and training configs that the release runs.

## Modified
Each entry names the paper result that needs it. Diffusion sampling is unchanged: it draws from the global RNG,
which the ReactBench runner seeds per token (paired seeds, `reactbench/models/seeds.py`).
- `navsim/agents/recogdrive/recogdrive_agent.py`, `recogdrive_diffusion_planner.py`:
  - ReactTrain fine-tuning (IL + RT): `per_sample_loss`, the unchanged stage-2 objective returned per sample.
  - All ReCogDrive results: the checkpoint loader `planner_state_from_checkpoint` accepts the IL/RL checkpoints,
    which lack the planner's (unused) dropout null states; `inference_precision` (default `32-true`, as upstream).
- `navsim/planning/training/agent_lightning_module.py`, `navsim/planning/script/run_training_recogdrive_rl.py`
  and `config/training/{training_base,recogdrive_stage3}.yaml`: ReactTrain fine-tuning. A kinematic-SFT
  training view (`mixed_training_manifest`) trains the stage-2 objective on the kinematic stop targets with an
  epoch-wise original sampler, linear LR warm-up before a per-update cosine schedule, an update limit and a
  checkpoint after every epoch (the paper's checkpoint is `epoch=2-step=276.ckpt`). One validation forward per
  batch (upstream ran a discarded extra one, whose RNG draws changed later epochs).
- `navsim/evaluate/pdm_score.py`: `pdm_scores`, the PDMS of several candidate trajectories of one scene in one
  batch (equal to repeated `pdm_score` calls), used by the GRPO exploration probe (appendix).
- `run_pdm_score_recogdrive.py`: `+inference_seed`, a per-token seed for the navtest PDMS (Table 3).
- `config/pdm_scoring/default_run_pdm_score.yaml`, `config/training/default_training.yaml` and
  `config/common/agent/recogdrive_agent.yaml`: ReCogDrive as the default agent and the agent options.
- `navsim/planning/training/dataset.py`: the unused `Dataset_For_Traj`/`Dataset_For_Pipeline` (and with them
  `navsim/visualization/`) are removed; so is the unused `AgentLightningModule` (`agent_lightning_module.py`
  keeps `AgentLightningDiT`).

Dormant development options, off by default and not used in the paper (`grpo=false` in ReactTrain): the GRPO
reward variants (`raw`, `minimum_jerk`, speed siblings; the raw-safety variant is not included and its mode
raises `NotImplementedError`), stage-3 planner initialization and BC options, velocity/acceleration and
joint motion dropout, and KV-cache conditioning of the planner (`conditioning_mode=vlm_kv`, `mask_vlm_padding`;
`recogdrive_backbone.py`, `recogdrive_dit.py` and `recogdrive_features.py` change only for it).

## Added
- ReactTrain fine-tuning: `navsim/planning/training/recogdrive_{kinematic_sft,mixed_data,mixed_training}.py`
  (stop targets, training views, sampler, schedule), `scripts/cache_dataset/` (VLM features and metric caches
  of the edited samples, cache validators, targets, views) and `navsim/planning/metric_caching/edited_scenario.py`
  (metric caches with the inserted actor as a tracked object).
- `navsim/evaluate/minimum_jerk_trajectory.py`: the GT path of the stop targets (and of the `minimum_jerk` reward).
- Dormant options above: `navsim/evaluate/{minimum_jerk_pdm_score,raw_pdm_score,sibling_targets}.py` (reward
  variants), `navsim/agents/recogdrive/recogdrive_motion_inputs.py` (joint dropout),
  `navsim/agents/recogdrive/{recogdrive_conditioning.py,blocks/kv_cross_attention.py}` (KV-cache conditioning).
- `metadata/vlm_artifacts.sha256`: SHA-256 of the released VLM files, checked before caching features.
