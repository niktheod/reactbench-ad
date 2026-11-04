#!/usr/bin/env python
"""Extract the released RecogDrive VLM's features for selected training edits."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path

from edited_training_cache import (
    TRAIN_ROOT, argument_parser, atomic_write, code_identity, completed_record,
    digest, discover, ensure_contract, finish, freeze_manifest, load_features, locked,
    record_result, release_paths, save_features, scene_from_source, sha256, verify_file, verify_sample,
)


def feature_contract(vlm_path):
    """Pin the same released VLM/tokenizer inventory as the original training cache."""
    inventory = TRAIN_ROOT / "metadata/vlm_artifacts.sha256"
    files = {}
    for line in inventory.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if name.startswith("VLM/"):
            path = vlm_path / Path(name).name
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"VLM differs from the released cache weights/tokenizer: {path}")
            files[path.name] = actual
    if "model.safetensors" not in files:
        raise ValueError("Released VLM inventory is incomplete")
    root = TRAIN_ROOT / "navsim/agents/recogdrive"
    return {"protocol": "recogdrive-edited-training-features-v1", "vlm": str(vlm_path), "files": files,
            "camera": "current CAM_F0", "hidden_width": 1536,
            "code": code_identity([Path(__file__), Path(__file__).with_name("edited_training_cache.py"),
                root / "recogdrive_features.py", root / "recogdrive_backbone.py",
                root / "utils/internvl_preprocess.py", root / "utils/conversation.py",
                TRAIN_ROOT / "navsim/common/dataclasses.py"]),
            "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "Pillow")}}


def validate_features(features, original, targets):
    import torch
    for name, shape in (("history_trajectory", (4, 3)), ("high_command_one_hot", (4,)), ("status_feature", (8,))):
        value = features[name]
        if value.shape != shape or value.dtype != torch.float32 or value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError(f"Invalid edited {name}")
        if not torch.equal(value, original[name]):
            raise ValueError(f"Edited {name} differs from original feature cache")
    hidden = features["last_hidden_state"]
    if (hidden.ndim != 2 or hidden.shape[0] <= 0 or hidden.shape[1] != 1536
            or hidden.dtype != torch.float32 or hidden.device.type != "cpu" or not torch.isfinite(hidden).all()):
        raise ValueError("Invalid VLM last_hidden_state; expected finite CPU float32 [L,1536]")
    trajectory = targets["trajectory"]
    if trajectory.shape != (8, 3) or not torch.isfinite(trajectory).all():
        raise ValueError("Invalid recorded original trajectory target")


class FeatureExtractor:
    def __init__(self, vlm_path):
        self.vlm_path = vlm_path
        self.builder = None
        self.original_key = None

    def __call__(self, sample, source, feature_path, target_path):
        import torch
        from PIL import Image

        image = verify_file(sample["image"])
        with Image.open(image) as handle:
            handle.verify()
        if self.builder is None:
            from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
            if not torch.cuda.is_available():
                raise RuntimeError("Feature extraction requires a GPU; use the Slurm launcher")
            self.builder = ReCogDriveFeatureBuilder(cache_hidden_state=True, cache_mode=True,
                model_type="internvl", checkpoint_path=str(self.vlm_path), device="cuda:0")
            self.builder.backbone.requires_grad_(False).eval()
        key = digest({"features": source["features"], "targets": source["targets"]})
        if key != self.original_key:
            self.original = load_features(verify_file(source["features"]))
            self.target_bytes = verify_file(source["targets"]).read_bytes()
            self.targets = load_features(source["targets"]["path"])
            self.original_key = key
        scene = scene_from_source(source, future_frames=8)
        agent_input = scene.get_agent_input()
        agent_input.cameras[-1].cam_f0.image = image
        with torch.inference_mode():
            features = self.builder.compute_features(agent_input)
        validate_features(features, self.original, self.targets)
        save_features(feature_path, features)
        atomic_write(target_path, lambda handle: handle.write(self.target_bytes))
        validate_features(load_features(feature_path), self.original, load_features(target_path))


def prepare_features(args, manifest, contract, extractor):
    with locked(args.output / ".features.lock"):
        contract_id = ensure_contract(args.output, "features", contract)
        records = []
        for index, sample in enumerate(manifest["samples"], 1):
            identity = digest([manifest["identity"], contract_id, sample["sample_id"]])
            source = manifest["sources"][sample["source_token"]]
            try:
                verify_sample(sample, source)
                verify_file(source["features"])
                verify_file(source["targets"])
                record = completed_record(args.output, "features", sample, identity)
                if record is None:
                    paths = [args.output / sample[name] for name in ("features", "targets")]
                    extractor(sample, source, *paths)
                    verify_sample(sample, source)
                    record = record_result(args.output, "features", sample, identity, paths)
            except Exception as exc:
                record = record_result(args.output, "features", sample, identity, error=exc)
                print(f"FAILED {sample['sample_id']}: {exc}", flush=True)
                # A broken GPU/model must not cause thousands of identical attempts.
                if isinstance(exc, (RuntimeError, MemoryError, ImportError)):
                    finish(args.output, "features", manifest, contract_id, records + [record])
            records.append(record)
            if index % 100 == 0:
                print(f"Features: {index}/{len(manifest['samples'])}", flush=True)
        finish(args.output, "features", manifest, contract_id, records)


def main():
    parser = argument_parser(__doc__)
    parser.add_argument("--vlm-path", type=Path, default=release_paths.get("checkpoints", "checkpoints") / "recogdrive/VLM")
    args = parser.parse_args()
    manifest = discover(args)
    manifest = freeze_manifest(args, manifest)
    if args.dry_run:
        return
    vlm_path = args.vlm_path.expanduser().resolve()
    contract = feature_contract(vlm_path)
    prepare_features(args, manifest, contract, FeatureExtractor(vlm_path))


if __name__ == "__main__":
    main()
