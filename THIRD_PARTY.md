# Third-party code and assets

| Component | Version | Licence | Where | Use |
|---|---|---|---|---|
| ReCogDrive (with its NAVSIM devkit) | 37cd7b2, modified | Apache-2.0 | `third_party/recogdrive` (`CHANGES.md`) | ReCogDrive runs, NAVSIM devkit for scoring, ReactTrain fine-tuning |
| libcom (PCTNet image harmonization) | 0.1.7, subset, unmodified | Apache-2.0 | `reactbench/compositing/harmonization` | Harmonization of the composited object (PCTNet only; the copy also contains libcom's LBM model, unused, which `image_harmonization.py` imports at load time, hence `diffusers` in `envs/pctnet.yml`) |
| Alpamayo 1.5 | f42e594 + `patches/alpamayo1.5.patch` | see upstream | fetched by `third_party/setup.sh` | Evaluated model. Patch: optional `diffusion_seed` for a dedicated generator of the initial diffusion noise (`flow_matching.py`; paired seeds) |
| Alpamayo 2 | 5e7975f | see upstream | fetched | Evaluated model |
| AutoVLA | ba34eed + `patches/AutoVLA.patch` | see upstream | fetched | Evaluated model. Patch: Qwen2.5-VL loaded from the Hugging Face id, `attn_implementation="flash_attention_2"`, a check of the number of generated action tokens; `setup.sh` points the action codebook path at the checkout |
| DriveWAM | abbefb4 + `patches/DriveWAM.patch` | see upstream | fetched | Evaluated model. Patch: nuplan-devkit pinned to its git tag (`environment.yml`); `src/modules/model.py` runs without `flash_attn` (optional import) and disables the cuDNN attention backend in the math-only SDPA context |
| SimWAM | 1433ee7 | see upstream | fetched | Evaluated model |
| Blender | 4.5.11 LTS | GPL-3.0-or-later | fetched | Rendering of the inserted objects |
| Microsoft Rocketbox avatars and animals | 0943055 | MIT | built by `scripts/assets.sh` | Pedestrians and standing animals |
| OpenX vehicle assets | 463ea3d | MPL-2.0 AND CC-BY-4.0 | built by `scripts/assets.sh` | Cars (models below; per-asset attribution also in each `.xoma`) |
| Mercedes-Benz GLS (X167) 2019 3D model by Merc_TV (Sketchfab) | hood mesh only | CC-BY-4.0 | `reactbench/compositing/paiav/assets/ego_vehicle/gls_x167_hood.npz` | Ego-vehicle hood mask of the PAI-AV front camera |
| Depth Anything V2 Small | 5426e4f (Hugging Face) | Apache-2.0 | `scripts/download_checkpoints.sh depth` | Depth for occlusion handling |

The model weights are downloaded from their publishers (`scripts/download_checkpoints.sh`) under their own
licences.

**Vehicle models.** The seven cars, which appear in the edited ReactNAV, ReactTrain and ReactPAI images, are OpenX
Assets adaptations of Sketchfab models licensed under CC BY 4.0: Audi TT Roadster (2014), Dacia Duster (2010),
Hyundai Tucson (2015) and Mini Countryman (2016) by razor24; BMW X1 (2016) by mk2design; Volvo EX30 (2024) by
Mehdi Lagzouli (LagzDesign); and Volvo V60 Polestar (2013) by myedsu.
