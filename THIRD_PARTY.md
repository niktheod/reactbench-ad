# Third-party code and assets

| Component | Version | Licence | Where | Use |
|---|---|---|---|---|
| ReCogDrive (with its NAVSIM devkit) | 37cd7b2, modified | Apache-2.0 | `third_party/recogdrive` (`CHANGES.md`) | ReCogDrive runs, NAVSIM devkit for scoring, ReactTrain fine-tuning |
| libcom (PCTNet image harmonization) | 0.1.7, subset | Apache-2.0 | `reactbench/compositing/harmonization` | Harmonization of the composited object |
| Alpamayo 1.5 | f42e594 + `patches/alpamayo1.5.patch` | see upstream | fetched by `third_party/setup.sh` | Evaluated model |
| Alpamayo 2 | 5e7975f | see upstream | fetched | Evaluated model |
| AutoVLA | ba34eed + `patches/AutoVLA.patch` | see upstream | fetched | Evaluated model |
| DriveWAM | abbefb4 + `patches/DriveWAM.patch` | see upstream | fetched | Evaluated model |
| SimWAM | 1433ee7 | see upstream | fetched | Evaluated model |
| Blender | 4.5.11 LTS | GPL-3.0-or-later | fetched | Rendering of the inserted objects |
| Microsoft Rocketbox avatars and animals | 0943055 | MIT | built by `scripts/assets.sh` | Pedestrians and standing animals |
| OpenX vehicle assets | 463ea3d | MPL-2.0 AND CC-BY-4.0 | built by `scripts/assets.sh` | Cars (per-asset attribution in each `.xoma`) |
| Mercedes-Benz GLS (X167) 2019 3D model by Merc_TV (Sketchfab) | hood mesh only | CC-BY-4.0 | `reactbench/compositing/paiav/assets/ego_vehicle/gls_x167_hood.npz` | Ego-vehicle hood mask of the PAI-AV front camera |
| Depth Anything V2 Small | Hugging Face | Apache-2.0 | downloaded on first use | Depth for occlusion handling |

The model weights are downloaded from their publishers (`scripts/download_checkpoints.sh`) under their own
licences.
