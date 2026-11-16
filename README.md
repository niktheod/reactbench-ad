# ReactBench

**ReactBench-AD: Do Driving VLAs and World-Action Models React to Imminent Hazards in Their Path?** · <Authors> · [Paper](<arXiv link>) · [Project page](<project page>)

ReactBench measures how driving models react to objects that appear in their path. It edits real NAVSIM
(**ReactNAV**, 3,228 navtest scenes) and PhysicalAI-Autonomous-Vehicles (**ReactPAI**, 1,045 clips) samples:
a crossing pedestrian, a standing pedestrian or a stationary car is composited into the camera images so that
the logged ego trajectory reaches it after a look-ahead time τ ∈ {0.5, 1.0, 1.5, 2.0} s. A model is run on the
original and on the edited scene of each sample, and both predictions are scored in the edited scene.
**ReactTrain** (paper section 5) fine-tunes ReCogDrive on edited navtrain scenes.

**Main result** (ΔP in percentage points, higher is better; * the paired 95% bootstrap CI includes zero):

| ReactNAV | Cross. τ=0.5 | 1.0 | 1.5 | 2.0 | Stand. τ=0.5 | 1.0 | 1.5 | 2.0 | Car τ=0.5 | 1.0 | 1.5 | 2.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Alpamayo 1.5 | +0.25 | +3.10 | +7.16 | +8.71 | +0.09* | +3.66 | +11.93 | +17.63 | +0.12 | +3.84 | +15.92 | +28.81 |
| Alpamayo 2 | +0.06* | +2.11 | +6.75 | +8.36 | +0.03* | +3.25 | +9.42 | +12.33 | +0.12 | +5.36 | +19.39 | +22.43 |
| AutoVLA | +0.00* | -0.03* | +0.53 | +0.90 | +0.00* | +0.16 | +0.34 | +0.96 | +0.03* | +0.87 | +2.14 | +3.26 |
| ReCogDrive | +0.00* | -0.03* | +0.22 | +0.40 | +0.00* | +0.00* | +0.12 | +0.40 | +0.00* | +0.00* | +0.28 | +1.89 |
| DriveWAM | +0.00* | +0.06* | +0.12 | +0.09* | +0.00* | +0.12 | +0.56 | +2.08 | +0.00* | +1.77 | +3.62 | +5.61 |
| SimWAM | +0.00* | +0.06* | +0.12 | +0.22 | +0.00* | +0.00* | +0.15 | +0.71 | +0.00* | +0.06* | +0.99 | +2.26 |
| **ReactPAI** | | | | | | | | | | | | |
| Alpamayo 1.5 | +0.10* | +5.55 | +25.17 | +35.12 | +0.19* | +5.07 | +21.63 | +35.69 | +0.67 | +12.92 | +20.86 | +25.65 |
| Alpamayo 2 | +0.00* | +3.44 | +22.68 | +34.07 | +0.00* | +0.57 | +8.61 | +15.89 | +0.10* | +5.55 | +16.65 | +25.17 |

## The benchmark

### Samples

A dataset has 12 runs, one per inserted object and look-ahead time, named `<object>_timestep_<k>`
(`crossing_pedestrian`, `standing_pedestrian`, `car`; τ = k × frame interval). A sample is identified by its
token; the original scene is the unedited source sample with the same token.

- **ReactNAV** (k = 1-4, frames 0.5 s apart): `final_images/<token>/{CAM_L0,CAM_F0,CAM_R0}/{0,1,2,3}.{png,jpg}`,
  the front-left, front and front-right cameras over four history frames (3 is the current frame). The ego
  history, velocity, acceleration and driving command are those of the NAVSIM navtest sample, available as
  `<outputs>/annotations/reactnav.json` (below), together with the paths of the original images.
- **ReactPAI** (k = 5, 10, 15, 20, frames 0.1 s apart): `final_images/<clip>/<camera>/{0,1,2,3}.{png,jpg}` for
  `camera_cross_left_120fov`, `camera_front_wide_120fov`, `camera_cross_right_120fov` and
  `camera_front_tele_30fov`. The original images, the ego history (16 poses at 10 Hz) and the vehicle geometry
  come from the PAI-AV download; `reactbench.datasets.paiav.PAIAVCompactRun` loads them
  (`edited_camera_paths`, `materialize_original_images`, `ego_history`, `ego_vehicle_geometry`). There is no
  navigation command.
- Both: `metadata/<token>.json` describes the inserted object (asset, placement, motion); the scorer reads it.

### Output format

Save one prediction per sample and scene:

    <outputs>/predictions/<model>/<reactnav|reactpai>/original/trajectories/<token>.npz   # original scene
    <outputs>/predictions/<model>/<reactnav|reactpai>/<run>/trajectories/<token>.npz      # edited scene

Each `.npz` holds `trajectory`, a float array (N, 3) of rear-axle poses (x, y, heading) in the ego frame at
the current frame (x forward, y left, heading in radians), where pose i is at time (i + 1) · dt, and
`interval_length_s` = dt. It must cover at least 4 s; only the first 4 s are scored. The original and the edited
prediction of a sample should use the same random numbers (`reactbench/models/seeds.py`).

### Metrics

Both predictions of a sample are evaluated in the edited scene with at-fault collision checks at 100 Hz on the
piecewise-linear trajectory (paper section 3.2 and appendix). P = NC × DAC on ReactNAV and P = NC on ReactPAI;
ΔP = P(edited prediction) − P(original prediction). ΔCS is the change in contact speed over the samples where
both predictions collide, and TIR the share of samples whose two predictions differ by less than ε (ADE).

## Getting the data

**ReactNAV** is released; it builds on your copy of NAVSIM navtest (OpenScene logs, sensor blobs and maps).

1. Download the edited images (`<release link: ReactNAV>`, CC BY-NC-SA 4.0) to `datasets/reactnav/`.
2. Set `navsim_data` in `configs/paths.yaml` (`cp configs/paths.example.yaml configs/paths.yaml`).
3. `python -m reactbench.datasets.navsim_annotations data/reactnav/reactnav.yaml` (NAVSIM devkit; writes
   `<outputs>/annotations/reactnav.json`).
4. `python -m reactbench.datasets.reactnav link`: the release holds only the frames the object changes; this adds
   the unchanged frames as links to your navtest images.
5. Download the lane polygons (`<release link: lanes>`) to `outputs/work/navsim/lanes/` (`scoring_work`); the
   ReactNAV scorer needs them. Regenerating them needs a NAVSIM v2 navtest metric cache:
   `python -m reactbench.scoring.navsim_lanes --tokens-from outputs/annotations/reactnav.json
   data/real_scenes/navtest_v2.json --metric-cache <cache>` with the NAVSIM v2 devkit on `PYTHONPATH`.

**ReactPAI** cannot be redistributed (PAI-AV licence) and is regenerated from the original clips:

1. Accept the dataset licence of `nvidia/PhysicalAI-Autonomous-Vehicles` on Hugging Face and `hf auth login`.
2. `python -m reactbench.datasets.download_reactpai` (`paiav_root` in `configs/paths.yaml`) downloads only the
   1,045 clips' cameras, LiDAR, egomotion, calibration and obstacle labels (about 470 GB, against ~10 TB for the
   whole archives); `--dry-run` previews it.
3. Composite the 12 runs as in [Benchmark construction](#benchmark-construction). This needs the compositing setup
   (Blender, Depth Anything, the assets, two environments) and GPUs for rendering and harmonization.

## Evaluating your model

1. Save its predictions in the [output format](#output-format) for the original scenes and the 12 runs. The six
   runners in `reactbench/models/` are worked examples: subclassing `reactbench.models.runner.Runner` gives the
   run layout, paired seeds, failure handling and scoring (`models/drivewam/eval.py` is about 50 lines).
2. Score them (CPU; Python with the NAVSIM devkit for ReactNAV, `scoring_python` in `configs/paths.yaml`):

       python -m reactbench.scoring outputs/predictions/<model>/reactnav [--workers 8]
       python -m reactbench.scoring outputs/predictions/<model>/reactpai

   Each run gets `metrics/*_at_fault_{per_sample,summary,manifest}.json`.
3. Summarize, alone or next to the paper's models:

       python -m reactbench.summarize --models <model> --tables main tir
       python -m reactbench.summarize --models alpamayo1_5 recogdrive <model> --tables main tir --by-speed

## Reproducing the paper

### Tables from the released scores (CPU, minutes)

    pip install -e .                      # numpy, scipy, shapely, pyyaml (Python >= 3.9)
    REACTBENCH_OUTPUTS=/abs/path/to/scores python -m reactbench.summarize

with the per-sample scores of every run (`<release link: scores>`) unpacked at `/abs/path/to/scores` (a relative
`REACTBENCH_OUTPUTS` resolves against the repository root). It prints and writes to `<scores>/summaries/` the
tables `main` (main and detailed results), `tir`, `side_cameras` (AutoVLA with CAM_L1/R1 or CAM_L0/R0),
`training_stages`, `fine_tuning` (ReactNAV and navtest PDMS), `generalization` (unseen objects, real scenes),
`real_scene_control` (ReCogDrive, DriveWAM and SimWAM on the real scenes), `speed_distribution`, `feasibility`
(τ = 0.5 s, unavoidable samples), `cot` (CoT swap, fixed CoTs), `knockout` and `exploration_probe`; select some
with `--tables`. Every cell uses the model's own valid original/edited pairs.
`--tir-curves` and `--by-speed` also write the data of the TIR and ΔP-by-speed figures as CSV. All intervals are
95% percentile bootstrap intervals over scenes with 10,000 resamples. The released scores contain no CoT texts or trajectories, so
the CoT summary figure (`python -m reactbench.analysis.cot.summary`) and the original-scene plan comparison of IL
and IL + RT (section 5, printed under `fine_tuning`) need the model outputs.

### Setup for running the models and the benchmark construction

1. `third_party/setup.sh` fetches Alpamayo 1.5, Alpamayo 2, AutoVLA, DriveWAM and SimWAM at the evaluated commits,
   applies `third_party/patches/` and installs Blender 4.5.11 LTS. ReCogDrive and its NAVSIM devkit are in
   `third_party/recogdrive` (`CHANGES.md`).
2. Environments (`conda env create -f envs/<name>.yml`; the Alpamayo ones are uv environments, see the head of
   `envs/alpamayo*.txt`):

   | Environment | Used for |
   |---|---|
   | `recogdrive` | ReCogDrive, scoring (`scoring_python`), annotations, ReactTrain, navtest PDMS, real scenes, probe, tests |
   | `autovla` | AutoVLA; PAI-AV compositing (`PYTHON=` for `scripts/composite_paiav.sh`) |
   | `compositing-navsim` | NAVSIM compositing (`PYTHON=` for `scripts/composite_navsim.sh`), assets |
   | `pctnet` | PCTNet harmonization (`PCTNET_PYTHON=`) |
   | `drivewam`, `simwam` | DriveWAM, SimWAM |
   | `alpamayo1_5`, `alpamayo2` | Alpamayo 1.5, Alpamayo 2 |

   Run commands from the repository root with `PYTHONPATH=third_party/recogdrive:.` (`scripts/env.sh`).
3. `configs/paths.yaml`: the NAVSIM data, `paiav_root`, checkpoints, outputs and `scoring_python` (any key can
   also be set as `REACTBENCH_<KEY>`).
4. `scripts/download_checkpoints.sh` downloads ReCogDrive, AutoVLA, DriveWAM and SimWAM (then run
   `python -m reactbench.models.simwam.prompt_embedding` in the `simwam` environment). The Alpamayo models load from
   the Hugging Face cache (accept their licences and log in). The ReCogDrive IL + RT checkpoint
   (`<release link: checkpoint>`) goes to `checkpoints/recogdrive/IL_RT/recogdrive_il_rt.ckpt`.

### Evaluation (section 4)

Every runner takes a benchmark run, first caches the original-scene predictions (`--original`, once per model and
dataset), then predicts and scores each run; outputs go to `<outputs>/predictions/<model>/<dataset>/<run>/`.

```bash
python -m reactbench.datasets.navsim_annotations data/reactnav/reactnav.yaml          # once (recogdrive env)
for model in recogdrive drivewam autovla simwam alpamayo1_5 alpamayo2; do             # ReactNAV
  python -m reactbench.models.$model.eval datasets/reactnav/car_timestep_1 --original
  for run in datasets/reactnav/{crossing_pedestrian,standing_pedestrian,car}_timestep_?; do
    python -m reactbench.models.$model.eval $run
  done
done
for model in alpamayo1_5 alpamayo2; do                                                # ReactPAI
  python -m reactbench.models.$model.eval datasets/reactpai/car_timestep_5 --dataset reactpai --original
  for run in datasets/reactpai/*_timestep_*; do python -m reactbench.models.$model.eval $run --dataset reactpai; done
done
```

Each runs in its model's environment. `--config` selects a configuration (ReCogDrive: `rl` (default), `il`,
`vlm_sft`, `il_rt`; SimWAM: `rl` (default), `il`; AutoVLA: `config` (default), `l0r0`): the training-stage,
fine-tuning and side-camera tables use the other ones, each with its own `--original` pass. `--max-samples N` runs a smoke test, `--overwrite` replaces an existing output.
AutoVLA reads its own preprocessed navtest samples (`autovla_inputs`: AutoVLA's `run_nuplan_preprocessing.sh` on
navtest, without CoT). As in AutoVLA's released code, its front-left and front-right views are CAM_L1/R1, which are
not rendered: they stay original, and the runner skips the 1, 4 and 9 crossing-pedestrian samples (τ = 1.0, 1.5,
2.0 s) whose pedestrian would enter them (`data/reactnav/autovla_side_camera_fov.json`). `--config l0r0`
(`autovla_l0r0`) feeds the rendered CAM_L0/R0 instead, on every sample (appendix). `reactbench/models/runner.py` holds what the runners share; `models/drivewam/eval.py` is the
shortest example of a runner. **Paired seeds:** the original and the edited scene of a pair share all random
numbers; every runner derives its seeds from the token (`reactbench/models/seeds.py`: the first 63 bits of SHA-256
of `"<namespace>:<base seed>:<token>"`). GPU numerics differ slightly between GPU models; the paper's runs used
NVIDIA A100 and H100 GPUs.

The CoT analysis (section 4.2, appendix) runs on the ReactPAI runs at τ = 2.0 s and reuses the predictions above:

```bash
for model in alpamayo1_5 alpamayo2; do
  for object in crossing_pedestrian standing_pedestrian car; do
    python -m reactbench.analysis.cot.run_$model datasets/reactpai/${object}_timestep_20 --cot-swap
    python -m reactbench.analysis.cot.run_$model datasets/reactpai/${object}_timestep_20 --fixed-cots
  done
done
python -m reactbench.analysis.cot.summary                  # CoT summary figure (all look-ahead times)
python -m reactbench.analysis.knockout.run_alpamayo1_5     # attention knockout: the 300 clips of data/knockout/tokens.json
python -m reactbench.analysis.knockout.run_alpamayo2
python -m reactbench.analysis.knockout.analyze
```

### ReactTrain fine-tuning (section 5)

```bash
scripts/reacttrain/composite.sh            # 24 runs on 2,000 navtrain scenes -> 1,957 sources
scripts/reacttrain/caches.sh original      # navtrain VLM features and metric caches (GPU)
scripts/reacttrain/caches.sh edited        # the same for the edited samples
scripts/reacttrain/view.sh                 # kinematic stop targets and the training view (CPU)
GPUS=1 scripts/reacttrain/train.sh         # 3 epochs, 276 updates
cp <reacttrain_work>/training/il_rt_seed0/checkpoints/epoch=2-step=276.ckpt checkpoints/recogdrive/IL_RT/recogdrive_il_rt.ckpt
scripts/navtest_pdms.sh --metric-cache     # navtest PDMS (GPU), once
scripts/navtest_pdms.sh recogdrive_il checkpoints/recogdrive/IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt
scripts/navtest_pdms.sh recogdrive_il_rt checkpoints/recogdrive/IL_RT/recogdrive_il_rt.ckpt
```

and the ReactNAV runs of `--config il` and `--config il_rt` as above. The ReactTrain scenes are fixed by
`data/reacttrain/` (the 8,734 training-log scenes at >= 5 km/h whose hazardous and benign variants of all three
objects can be placed at every look-ahead time); `composite.sh` renders 2,000 of them and keeps the 1,957 rendered
in all 24 runs (`python -m reactbench.selection.reacttrain intersect`).

**Generalization.** Unseen objects: `scripts/assets.sh animals <Rocketbox clone>`, then
`scripts/composite_navsim.sh --animal --max-samples 300 standing_animal_timestep_$t $t` for t = 1-4, evaluated
with `--config il` and `il_rt`. Real scenes (`data/real_scenes/navtest_v2.json`; its selection:
`python -m reactbench.selection.real_scenes --output outputs/real_scenes/cohort.json` and
`python -m reactbench.models.recogdrive.real_scenes build --cohort outputs/real_scenes/cohort.json`):

    python -m reactbench.models.recogdrive.real_scenes infer --name recogdrive_il --config il
    python -m reactbench.models.recogdrive.real_scenes infer --name recogdrive_il_rt --config il_rt
    # appendix control: the released single-frame models (SimWAM and DriveWAM in their own environments)
    python -m reactbench.models.recogdrive.real_scenes infer --name recogdrive --config rl
    python -m reactbench.models.recogdrive.real_scenes annotations
    python -m reactbench.models.simwam.real_scenes infer --name simwam
    python -m reactbench.models.drivewam.real_scenes infer --name drivewam
    python -m reactbench.models.recogdrive.real_scenes summarize

**GRPO exploration probe** (appendix): `python -m reactbench.analysis.exploration_probe` (after `view.sh`).

### Benchmark construction (section 3)

```bash
scripts/download_checkpoints.sh depth                  # Depth Anything V2 Small, run offline by both compositors
scripts/assets.sh humans <Microsoft-Rocketbox clone>   # Rocketbox 0943055
scripts/assets.sh cars <openx-assets clone>            # OpenX 463ea3d
for t in 1 2 3 4; do                                   # ReactNAV: tau = t x 0.5 s
  scripts/composite_navsim.sh crossing_pedestrian_timestep_$t $t
  scripts/composite_navsim.sh --standing standing_pedestrian_timestep_$t $t
  scripts/composite_navsim.sh --car car_timestep_$t $t
done
for k in 5 10 15 20; do                                # ReactPAI: tau = k x 0.1 s
  scripts/composite_paiav.sh crossing_pedestrian_timestep_$k $k
  scripts/composite_paiav.sh --standing standing_pedestrian_timestep_$k $k
  scripts/composite_paiav.sh --car car_timestep_$k $k
done
```

The scenes and crossing sides are fixed by `data/`: ReactNAV is the 3,228 navtest scenes with an ego speed of at
least 5 km/h (all end up at 5-40 km/h) in which all three objects can be placed at every look-ahead time and every
run rendered successfully; the selection scripts are
not released. The ReactPAI clips are listed in `data/reactpai/selection_manifest.json`.

### Unavoidable collisions and tests

`python -m reactbench.scoring.feasibility --reactnav outputs/predictions/recogdrive/reactnav --reactpai
outputs/predictions/alpamayo1_5/reactpai` writes the per-sample braking-feasibility verdicts read by the
`feasibility` table (NAVSIM logs and PAI-AV data needed; the verdicts are part of the released scores).
`python -m pytest tests` (`recogdrive` environment) checks the at-fault scoring, the output format and the
real-scenes selection.

## Layout

| Path | Contents |
|---|---|
| `reactbench/datasets/` | ReactNAV (`reactnav.py`: link/pack) and ReactPAI (`paiav.py`, `download_reactpai.py`) data, NAVSIM annotations |
| `reactbench/scoring/` | At-fault scoring of ReactNAV and ReactPAI (`python -m reactbench.scoring`), braking feasibility |
| `reactbench/summarize/` | All tables (`python -m reactbench.summarize`); `reactbench/export.py` writes the released scores |
| `reactbench/models/` | Runners: Alpamayo 1.5, Alpamayo 2, AutoVLA, ReCogDrive, DriveWAM, SimWAM (the last three also on the real scenes) |
| `reactbench/analysis/` | CoT swap and fixed CoTs, CoT summary, attention knockout, GRPO exploration probe |
| `reactbench/compositing/` | NAVSIM and PAI-AV compositors (Blender), PCTNet harmonization (libcom 0.1.7) |
| `reactbench/assets/`, `reactbench/selection/` | Builders of the inserted objects; real-scenes selection, ReactTrain intersection |
| `third_party/` | Modified ReCogDrive with its NAVSIM devkit; `setup.sh` and `patches/` for the other models |
| `data/` | Scene and clip lists, crossing and benign sides, AutoVLA's side-camera exclusions, real-scenes scenes, knockout clips |
| `scripts/`, `envs/`, `configs/`, `tests/` | Launchers, environments, `paths.example.yaml`, unit tests |

## Citation

```bibtex
<BibTeX>
```

## Licences and acknowledgements

Code: Apache-2.0 (`LICENSE`). Data: `DATA_LICENSE.md`. Third-party code and assets: `THIRD_PARTY.md`.
<Acknowledgements>
