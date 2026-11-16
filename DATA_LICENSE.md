# Data licences

- **ReactNAV, ReactTrain and the standing-animal set** are edits of nuPlan/OpenScene camera images and are
  released under the licence of those datasets, CC BY-NC-SA 4.0. The scene lists, crossing sides and benign
  sides in `data/reactnav/` and `data/reacttrain/`, the real-scenes file `data/real_scenes/navtest_v2.json`
  and the navtest lane polygons are derived from NAVSIM/nuPlan and fall under the same licence.
- **ReactPAI** is not distributed: the NVIDIA PAI-AV dataset licence does not allow sharing modified data.
  `data/reactpai/` holds only clip identifiers, the archive layout of the PAI-AV release they are read from,
  and the crossing side per clip; the edited clips are regenerated locally from the original data.
- The inserted cars are CC BY 4.0 Sketchfab models adapted by OpenX Assets; their attributions are in
  `THIRD_PARTY.md` ("Vehicle models") and apply to every edited image that shows them.
