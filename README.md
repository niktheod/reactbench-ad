# ReactBench

Code and data lists for the paper's benchmark of how driving VLAs and world-action models react to objects
inserted into their path. ReactBench edits real NAVSIM (**ReactNAV**, 3,228 navtest scenes) and PAI-AV
(**ReactPAI**, 1,045 clips) samples: a crossing pedestrian, a standing pedestrian or a stationary car is
composited into the camera images so that the logged ego trajectory reaches it after a look-ahead time
τ ∈ {0.5, 1.0, 1.5, 2.0} s. Each model is run on the original and the edited scene with the same random
seeds, and the pair is scored with at-fault collision checks (P = NC × DAC on ReactNAV, NC on ReactPAI).
**ReactTrain** (section 5) fine-tunes ReCogDrive on edited navtrain scenes with kinematic stop targets.

The repository reproduces every table and number of the paper. `python -m reactbench.summarize` prints them
from scored runs, or from the released per-sample scores without running any model.

## Layout

| Path | Contents |
|---|---|
| `reactbench/compositing/` | NAVSIM and PAI-AV compositors (Blender), PCTNet harmonization (`harmonization/`, libcom 0.1.7) |
| `reactbench/assets/` | Builders of the inserted objects: Rocketbox humans and animals, OpenX cars |
| `reactbench/selection/` | ReactNAV, ReactTrain and real-scenes selection |
| `reactbench/models/` | Runners: Alpamayo 1.5, Alpamayo 2, AutoVLA, ReCogDrive (+ real scenes), DriveWAM, SimWAM |
| `reactbench/scoring/` | At-fault scoring of ReactNAV and ReactPAI pairs, braking feasibility |
| `reactbench/analysis/` | CoT swap, fixed CoTs, CoT summary, attention knockout, GRPO exploration probe |
| `reactbench/summarize/` | All tables (`python -m reactbench.summarize`) |
| `third_party/recogdrive/` | Modified ReCogDrive with its NAVSIM devkit (inference, scoring devkit, ReactTrain fine-tuning; `CHANGES.md`) |
| `third_party/setup.sh` | Fetches the other models' code at the paper's commits and applies `third_party/patches/` |
| `data/` | ReactNAV and ReactTrain scene lists, ReactPAI clip list, crossing sides, real-scenes scenes, knockout clips |
| `scripts/` | Launchers: compositing, assets, checkpoints, ReactTrain, navtest PDMS |
| `envs/` | One environment per stack |

## Setup

1. Environments: `conda env create -f envs/<name>.yml` for `recogdrive` (ReCogDrive, scoring, summaries),
   `autovla` (AutoVLA, PAI-AV compositing), `compositing-navsim`, `pctnet`, `drivewam`, `simwam`; the Alpamayo
   environments are uv environments (`envs/alpamayo1_5.txt`, `envs/alpamayo2.txt`). Run commands from the
   repository root with `PYTHONPATH=third_party/recogdrive:.` (`scripts/env.sh` sets it).
2. `third_party/setup.sh` fetches Alpamayo 1.5, Alpamayo 2, AutoVLA, DriveWAM and SimWAM at the evaluated
   commits, applies the patches and installs Blender 4.5.11 LTS.
3. `cp configs/paths.example.yaml configs/paths.yaml` and set the locations (any key can also be set as
   `REACTBENCH_<KEY>`): the NAVSIM data (`navsim_data`, the OpenScene layout), the PAI-AV download
   (`paiav_root`), checkpoints, outputs, and the Python of the scoring environment (`scoring_python`).
4. `scripts/download_checkpoints.sh` downloads ReCogDrive, AutoVLA, DriveWAM and SimWAM; the Alpamayo
   models load from the Hugging Face cache. The ReCogDrive IL + RT checkpoint of section 5 is released with
   the data (below).
5. Data that the scorer and the real-scenes set read: the lane polygons of the navtest scenes
   (`outputs/work/navsim/lanes/`, released with the data, or
   `python -m reactbench.scoring.navsim_lanes --tokens-from outputs/annotations/reactnav.json --metric-cache <dir>`
   on a NAVSIM v2 navtest metric cache) and the PAI-AV obstacle labels
   (`python -m reactbench.datasets.paiav_obstacles --download`).

## Released data

- ReactNAV edited images (CC BY-NC-SA 4.0, like nuPlan/OpenScene) and the ReCogDrive IL + RT checkpoint:
  `<release link>`. Unpack the images to `datasets/reactnav/`.
- ReactPAI cannot be redistributed (PAI-AV licence). `data/reactpai/selection_manifest.json` lists the 1,045
  clips; `python -m reactbench.datasets.download_reactpai` downloads their PAI-AV files and
  `scripts/composite_paiav.sh` regenerates the 12 runs (below).
- Per-sample scores of every run: `<release link>`; `REACTBENCH_OUTPUTS=<scores> python -m reactbench.summarize`
  prints all tables from them. `python -m reactbench.export DIR` writes such an export from your own outputs.

## Reproducing the paper

### ReactBench construction (section 3)

```bash
scripts/assets.sh humans <Microsoft-Rocketbox clone>   # Rocketbox 0943055
scripts/assets.sh cars <openx-assets clone>            # OpenX 463ea3d
for t in 1 2 3 4; do                                   # ReactNAV: 12 runs, tau = t x 0.5 s
  scripts/composite_navsim.sh crossing_pedestrian_timestep_$t $t
  scripts/composite_navsim.sh --standing standing_pedestrian_timestep_$t $t
  scripts/composite_navsim.sh --car car_timestep_$t $t
done
for k in 5 10 15 20; do                                # ReactPAI: 12 runs, tau = k x 0.1 s
  scripts/composite_paiav.sh crossing_pedestrian_timestep_$k $k
  scripts/composite_paiav.sh --standing standing_pedestrian_timestep_$k $k
  scripts/composite_paiav.sh --car car_timestep_$k $k
done
```

The scene lists are fixed by `data/`;
`python -m reactbench.selection.reactnav` documents and re-derives the ReactNAV list (the 3,696 placeable
navtest scenes from `reactbench/compositing/navsim/classify_samples.py`, the 3,253 at 5-40 km/h, and the 3,228
without a rendering failure in any run).

### Evaluation (section 4)

Each runner caches the original-scene predictions once (`--original`) and then runs and scores each edited
run; predictions and scores go to `outputs/predictions/<model>/<dataset>/<run>/`.

```bash
python -m reactbench.datasets.navsim_annotations data/reactnav/reactnav.yaml
for run in datasets/reactnav/*_timestep_*; do
  python -m reactbench.models.recogdrive.eval $run --config rl          # also: il, vlm_sft (training stages)
  python -m reactbench.models.drivewam.eval $run
  python -m reactbench.models.autovla.eval $run                         # inputs: AutoVLA's run_nuplan_preprocessing.sh
  python -m reactbench.models.simwam.eval $run --config rl              # also: il
  python -m reactbench.models.alpamayo1_5.eval_navsim $run
  python -m reactbench.models.alpamayo2.eval_navsim $run
done
for run in datasets/reactpai/*_timestep_*; do
  python -m reactbench.models.alpamayo1_5.eval_paiav $run
  python -m reactbench.models.alpamayo2.eval_paiav $run
done
```

(add one `--original` call per model and dataset first; `scripts/` has no batch launcher because GPU queues
differ). The CoT analysis runs on the ReactPAI runs at τ = 2.0 s: `--cot-swap` and `--fixed-cots` of the two
Alpamayo ReactPAI runners, `python -m reactbench.analysis.cot.summary` and
`python -m reactbench.analysis.cot.text_numbers` (numbers in the text). The attention knockout:

```bash
python -m reactbench.analysis.knockout.run_alpamayo1_5     # 300 clips of data/knockout/tokens.json
python -m reactbench.analysis.knockout.run_alpamayo2
python -m reactbench.analysis.knockout.analyze
```

### ReactTrain fine-tuning (section 5)

```bash
scripts/reacttrain/composite.sh                    # 24 runs on 2,000 navtrain scenes -> 1,957 sources
scripts/reacttrain/caches.sh original               # navtrain VLM features and metric caches
scripts/reacttrain/caches.sh edited                 # the same for the edited samples
scripts/reacttrain/view.sh                          # kinematic stop targets and the training view
GPUS=1 scripts/reacttrain/train.sh                  # 3 epochs, 276 updates; checkpoint epoch=2-step=276.ckpt
python -m reactbench.models.recogdrive.eval <run> --config il_rt   # ReactNAV (and the animal runs)
scripts/navtest_pdms.sh --metric-cache && scripts/navtest_pdms.sh recogdrive_il_rt <checkpoint>
```

Generalization: the unseen objects are standing Rocketbox animals (`scripts/assets.sh animals ...`, then
`scripts/composite_navsim.sh --animal --max-samples 300 standing_animal_timestep_$t $t`) evaluated like
ReactNAV; the real scenes (`data/real_scenes/navtest_v2.json`) run with
`python -m reactbench.models.recogdrive.real_scenes infer --name recogdrive_il --config il` and
`--name recogdrive_il_rt --config il_rt`,
then `... real_scenes summarize`. The GRPO exploration probe (appendix) is
`python -m reactbench.analysis.exploration_probe --manifest <ReactTrain view manifest>`.

### Tables

```bash
python -m reactbench.scoring.feasibility --reactnav outputs/predictions/recogdrive/reactnav \
    --reactpai outputs/predictions/alpamayo1_5/reactpai         # unavoidable samples (appendix)
python -m reactbench.summarize [--tables ...] [--tir-curves] [--by-speed]
```

prints the main and detailed results, trajectory invariance rates, training stages, fine-tuning,
generalization, sample statistics, unavoidable collisions, CoT swap, fixed CoTs, knockout and exploration
probe tables, and writes them to `outputs/summaries/`. All intervals are 95% percentile bootstrap intervals
over scenes with 10,000 resamples. GPU numerics differ slightly between GPU models; the paper's runs used
NVIDIA A100 and H100 GPUs.

## Licences

Code: Apache-2.0 (`LICENSE`). Third-party code and assets: `THIRD_PARTY.md`. Data: `DATA_LICENSE.md`.
