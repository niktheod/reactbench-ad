#!/usr/bin/env python
"""Build links and indexes for all original training data plus all prepared edits."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from edited_training_cache import EXP, atomic_write, write_json
from navsim.planning.training.recogdrive_mixed_data import (
    FEATURE_NAMES, PROTOCOL, EDITED_ONLY_PROTOCOL, HAZARD_SFT_PROTOCOL, ORIGINAL_POOL_SELECTION, check_file, digest,
    load_manifest, metric_index, sample_counts, select_original_pool, stamp, validate_edited_only, validate_hazard_sft,
)
from navsim.planning.training.recogdrive_mixed_data import (
    KINEMATIC_SFT_KINDS, KINEMATIC_SFT_PROTOCOL, KINEMATIC_SFT_TIMESTEPS, SFT_ACTORS, TIMESTEP_PAIR_SELECTION,
    TIMESTEP_PAIRS, edit_key, kinematic_sft_edits, validate_kinematic_sft,
)


def original_samples(validation_path, metric_root):
    from omegaconf import OmegaConf

    value = json.loads(validation_path.read_text())
    total = value["actual_token_count"]
    if (value.get("schema_version") != 1 or value.get("max_scenes") is not None or total <= 0
            or any(value.get(k) != total for k in (
                "expected_token_count", "validated_feature_file_count", "validated_target_file_count"))):
        raise ValueError("Original features require a complete hidden-state validation manifest")
    references = [stamp(validation_path, content=True)]
    for path_key, hash_key in (("train_val_log_split", "train_val_log_split_sha256"),
                               ("navtrain_filter", "navtrain_filter_sha256")):
        reference = stamp(value[path_key], content=True)
        if reference["sha256"] != value[hash_key]:
            raise ValueError(f"Original split/filter changed: {reference['path']}")
        references.append(reference)
    split = OmegaConf.load(value["train_val_log_split"])
    train_logs, val_logs = set(split["train_logs"]), set(split["val_logs"])
    if train_logs & val_logs:
        raise ValueError("Original training/validation logs overlap")
    allowed_tokens = set(OmegaConf.load(value["navtrain_filter"])["tokens"])
    metrics, index = metric_index(metric_root)
    references.append(stamp(index, content=True))
    root, samples = Path(value["cache_path"]).resolve(), {}
    def inspect_log(log):
        directory = root / log
        if not directory.is_dir():
            return []
        records = []
        for feature in sorted(directory.iterdir()):
            token = feature.name
            if token not in allowed_tokens or token not in metrics:
                raise ValueError(f"Unexpected/orphan original sample: {token}")
            files = [stamp(feature / name) for name in FEATURE_NAMES]
            files.append(stamp(metrics[token]))
            records.append((token, {"source_token": token, "kind": "original", "log_name": log,
                              "split": "train" if log in train_logs else "validation",
                              "feature_path": str(feature.resolve()), "metric_path": metrics[token], "files": files}))
        return records
    with ThreadPoolExecutor(max_workers=8) as pool:
        for records in pool.map(inspect_log, sorted(train_logs | val_logs)):
            for token, sample in records:
                if token in samples:
                    raise ValueError(f"Duplicate original sample: {token}")
                samples[token] = sample
                if len(samples) % 10000 == 0:
                    print(f"Original cache inventory: {len(samples)}/{total}", flush=True)
    counts = sample_counts(samples)
    if (len(samples) != total or counts["train"] != value["train_token_count"]
            or counts["validation"] != value["validation_token_count"]):
        raise ValueError("Original cache coverage differs from its validation manifest")
    return samples, references


def check_prepared_sample(root, kind, sample, source, manifest_identity, contract_identity):
    record = json.loads((root / "preparation" / kind / f"{sample['sample_id']}.json").read_text())
    expected = [root / sample[k] for k in (("features", "targets") if kind == "features" else ("metric_cache",))]
    if (record.get("error") or record.get("sample_id") != sample["sample_id"]
            or record.get("identity") != digest([manifest_identity, contract_identity, sample["sample_id"]])
            or [str(Path(r["path"]).resolve()) for r in record["files"]] != [str(p.resolve()) for p in expected]):
        raise ValueError(f"Invalid {kind} completion record for {sample['sample_id']}")
    current = [stamp(p) for p in expected]
    if any(now["size"] != old["size"] for now, old in zip(current, record["files"])):
        raise ValueError(f"Changed {kind} file size for {sample['sample_id']}")
    if any(now["mtime_ns"] != old["mtime_ns"] for now, old in zip(current, record["files"])):
        if kind != "features":
            raise ValueError(f"Changed metric cache for {sample['sample_id']}")
        # Some completed GPU outputs acquired later filesystem timestamps.
        # Revalidate those tensors/targets before recording their current stamps;
        # do not relax the preparation scripts' own restart checks.
        from edited_training_cache import load_features, verify_file
        from prepare_edited_training_features import validate_features
        original = load_features(verify_file(source["features"]))
        validate_features(load_features(expected[0]), original, load_features(expected[1]))
        if expected[1].read_bytes() != verify_file(source["targets"]).read_bytes():
            raise ValueError(f"Edited target differs from original for {sample['sample_id']}")
    return current


def build_manifest(original_validation, original_metrics, edited_root, output):
    samples, sources = original_samples(original_validation, original_metrics)
    path = edited_root / "manifest.json"
    edited = json.loads(path.read_text())
    if edited.get("identity") != digest({k: v for k, v in edited.items() if k != "identity"}):
        raise ValueError("Edited manifest checksum mismatch")
    sources.append(stamp(path, content=True))
    contract_ids = {}
    for kind in ("features", "metrics"):
        status_path, contract_path = edited_root / f"{kind}_status.json", edited_root / f"{kind}_contract.json"
        status = json.loads(status_path.read_text())
        contract_ids[kind] = digest(json.loads(contract_path.read_text()))
        if (status.get("complete") is not True or status.get("failures") != 0
                or status.get("samples") != len(edited["samples"]) or status.get("expected_samples") != len(edited["samples"])
                or status.get("manifest_identity") != edited["identity"] or status.get("contract") != contract_ids[kind]):
            raise ValueError(f"Edited {kind} preparation is incomplete or inconsistent")
        sources.extend(stamp(p, content=True) for p in (status_path, contract_path))
    metrics, index = metric_index(edited_root / "metrics")
    sources.append(stamp(index, content=True))
    if set(metrics) != {s["sample_id"] for s in edited["samples"]}:
        raise ValueError("Edited metric index differs from its manifest")
    for index, sample in enumerate(edited["samples"], 1):
        token, source_token = sample["sample_id"], sample["source_token"]
        original = samples.get(source_token)
        if token in samples or original is None or original["kind"] != "original" or original["split"] != "train":
            raise ValueError(f"Duplicate edited sample or non-training source: {token}")
        source = edited["sources"][source_token]
        if Path(source["features"]["path"]).parent.resolve() != Path(original["feature_path"]):
            raise ValueError(f"Edited source feature cache differs for {token}")
        feature = (edited_root / sample["features"]).resolve()
        if (feature.parent.name != token or feature.parent.parent.name != original["log_name"]
                or metrics[token] != str((edited_root / sample["metric_cache"]).absolute())):
            raise ValueError(f"Edited feature/metric/log identity mismatch: {token}")
        files = []
        for kind in ("features", "metrics"):
            files.extend(check_prepared_sample(edited_root, kind, sample, source, edited["identity"], contract_ids[kind]))
        check_file(sample["image"])
        samples[token] = {"split": "train", "kind": edited["variants"][sample["variant"]]["kind"],
                          "source_token": source_token, "log_name": original["log_name"],
                          "feature_path": str(feature.parent), "metric_path": metrics[token], "files": files,
                          "image": sample["image"], "actor_motion": sample["actor_motion"]}
        if index % 2000 == 0:
            print(f"Edited cache inventory: {index}/{len(edited['samples'])}", flush=True)
    value = {"protocol": PROTOCOL, "feature_cache": str(output / "features"),
             "metric_cache": str(output / "metrics"), "counts": sample_counts(samples),
             "samples": samples, "sources": sources, "edited_manifest_identity": edited["identity"]}
    for reference in sources:
        check_file(reference, content=True)
    return {**value, "artifact_sha256": digest(value)}


def edited_only_manifest(mixed, output, holdout_count=100, split_seed=0):
    """Create a source-grouped view without copying or recomputing any caches."""
    if mixed["protocol"] != PROTOCOL:
        raise ValueError("Edited-only preparation requires the complete mixed population")
    edits = {t: s for t, s in mixed["samples"].items() if s["kind"] != "original"}
    source_tokens = sorted({s["source_token"] for s in edits.values()})
    if type(holdout_count) is not int or not 0 < holdout_count < len(source_tokens):
        raise ValueError("holdout_count must leave both training and held-out sources")
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a nonnegative integer")
    # Hash ordering is independent of Python/NumPy RNG versions and training seed.
    held = sorted(sorted(source_tokens, key=lambda t: (digest([split_seed, t]), t))[:holdout_count])
    sources = {t: mixed["samples"][t] for t in source_tokens}
    samples = {t: s for t, s in mixed["samples"].items() if s["split"] == "validation"}
    samples.update({t: s for t, s in edits.items() if s["source_token"] not in held})
    value = {"protocol": EDITED_ONLY_PROTOCOL, "feature_cache": str(output / "features"),
             "metric_cache": str(output / "metrics"), "samples": samples, "original_sources": sources,
             "heldout_samples": {t: s for t, s in edits.items() if s["source_token"] in held},
             "holdout": {"seed": split_seed, "count": holdout_count, "source_tokens": held,
                         "selection": "sha256(seed,source)-ascending-v1"},
             "triplets": [{"source_token": t, "variant": f"{actor}_t{step}",
                           "benign": f"{t}__benign_{actor}_t{step}",
                           "hazardous": f"{t}__hazardous_{actor}_t{step}"}
                          for t in held for actor in ("car", "standing", "walking") for step in (3, 4)],
             "sources": mixed["sources"], "parent_identity": mixed["artifact_sha256"],
             "edited_manifest_identity": mixed["edited_manifest_identity"]}
    value["counts"] = sample_counts(samples, sources)
    validate_edited_only(value)
    def verify_sample(sample):
        for reference in sample["files"]:
            check_file(reference)
    inventory = [*samples.values(), *value["heldout_samples"].values(), *sources.values()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, _ in enumerate(pool.map(verify_sample, inventory), 1):
            if index % 10000 == 0:
                print(f"Edited-only cache inventory: {index}/{len(inventory)}", flush=True)
    return {**value, "artifact_sha256": digest(value)}


def hazard_sft_manifest(mixed, edited_only, targets_path, output, original_pool=None, pool_seed=0):
    """The edited-only split plus the original training samples outside its held-out sources.

    Clean validation, held-out sources/triplets and edits come unchanged from the edited-only
    manifest; the frozen hazardous target artifact is signed as a source. By default every such
    original is included; ``original_pool`` keeps a fixed seeded subset of that many, recorded
    in the manifest, so the view and its checks cover only the originals a run can draw.
    """
    from navsim.planning.training.recogdrive_hazard_sft import load_targets
    if mixed["protocol"] != PROTOCOL or edited_only["protocol"] != EDITED_ONLY_PROTOCOL:
        raise ValueError("Hazard SFT preparation requires the complete mixed and the edited-only manifests")
    if edited_only["parent_identity"] != mixed["artifact_sha256"]:
        raise ValueError("The edited-only manifest was not derived from this mixed manifest")
    held = set(edited_only["holdout"]["source_tokens"])
    candidates = [token for token, sample in mixed["samples"].items()
                  if sample["kind"] == "original" and sample["split"] == "train" and token not in held]
    chosen = candidates
    if original_pool is not None:
        selected = set(select_original_pool(candidates, original_pool, pool_seed))
        chosen = [token for token in candidates if token in selected]
    samples = dict(edited_only["samples"])
    for token in chosen:
        if token in samples:
            raise ValueError(f"Duplicate hazard SFT sample: {token}")
        samples[token] = mixed["samples"][token]
    targets_path = Path(targets_path).expanduser().resolve()
    target_value = json.loads(targets_path.read_text())
    value = {"protocol": HAZARD_SFT_PROTOCOL, "feature_cache": str(output / "features"),
             "metric_cache": str(output / "metrics"), "samples": samples,
             "original_sources": edited_only["original_sources"], "heldout_samples": edited_only["heldout_samples"],
             "holdout": edited_only["holdout"], "triplets": edited_only["triplets"],
             "sources": [*edited_only["sources"], stamp(targets_path, content=True)],
             "hazard_targets": {"path": str(targets_path), "artifact_sha256": target_value.get("artifact_sha256"),
                                "ranking": target_value.get("ranking")},
             "parent_identity": edited_only["artifact_sha256"], "mixed_identity": mixed["artifact_sha256"],
             "edited_manifest_identity": edited_only["edited_manifest_identity"]}
    if original_pool is not None:
        value["original_pool"] = {"selection": ORIGINAL_POOL_SELECTION, "seed": pool_seed, "count": original_pool,
                                  "candidates": len(candidates), "candidates_sha256": digest(sorted(candidates)),
                                  "tokens_sha256": digest(sorted(chosen))}
    value["counts"] = sample_counts(samples, value["original_sources"])
    validate_hazard_sft(value)
    load_targets(value)  # artifact checksum, split identity and exact hazardous coverage
    def verify_sample(sample):
        for reference in sample["files"]:
            check_file(reference)
    inventory = [*samples.values(), *value["heldout_samples"].values(), *value["original_sources"].values()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, _ in enumerate(pool.map(verify_sample, inventory), 1):
            if index % 10000 == 0:
                print(f"Hazard SFT cache inventory: {index}/{len(inventory)}", flush=True)
    return {**value, "artifact_sha256": digest(value)}


def edited_root_samples(edited_root, originals):
    """The training samples of one more prepared edited cache root, checked as ``build_manifest`` checks
    its root (ReactTrain fine-tuning adds the t1/t2 caches to the t3/t4 edits of a complete mixed manifest).

    ``originals`` are the complete mixed manifest's samples. Returns (samples, signed sources, edited
    manifest identity); nothing is copied or recomputed.
    """
    edited_root = Path(edited_root).expanduser().resolve()
    path = edited_root / "manifest.json"
    edited = json.loads(path.read_text())
    if edited.get("identity") != digest({k: v for k, v in edited.items() if k != "identity"}):
        raise ValueError("Edited manifest checksum mismatch")
    sources = [stamp(path, content=True)]
    contract_ids = {}
    for kind in ("features", "metrics"):
        status_path, contract_path = edited_root / f"{kind}_status.json", edited_root / f"{kind}_contract.json"
        status = json.loads(status_path.read_text())
        contract_ids[kind] = digest(json.loads(contract_path.read_text()))
        if (status.get("complete") is not True or status.get("failures") != 0
                or status.get("samples") != len(edited["samples"]) or status.get("expected_samples") != len(edited["samples"])
                or status.get("manifest_identity") != edited["identity"] or status.get("contract") != contract_ids[kind]):
            raise ValueError(f"Edited {kind} preparation is incomplete or inconsistent: {edited_root}")
        sources.extend(stamp(p, content=True) for p in (status_path, contract_path))
    metrics, index = metric_index(edited_root / "metrics")
    sources.append(stamp(index, content=True))
    if set(metrics) != {s["sample_id"] for s in edited["samples"]}:
        raise ValueError("Edited metric index differs from its manifest")
    samples = {}
    for position, sample in enumerate(edited["samples"], 1):
        token, source_token = sample["sample_id"], sample["source_token"]
        original = originals.get(source_token)
        if (token in originals or token in samples or original is None or original["kind"] != "original"
                or original["split"] != "train"):
            raise ValueError(f"Duplicate edited sample or non-training source: {token}")
        source = edited["sources"][source_token]
        if Path(source["features"]["path"]).parent.resolve() != Path(original["feature_path"]):
            raise ValueError(f"Edited source feature cache differs for {token}")
        feature = (edited_root / sample["features"]).resolve()
        if (feature.parent.name != token or feature.parent.parent.name != original["log_name"]
                or metrics[token] != str((edited_root / sample["metric_cache"]).absolute())):
            raise ValueError(f"Edited feature/metric/log identity mismatch: {token}")
        files = []
        for kind in ("features", "metrics"):
            files.extend(check_prepared_sample(edited_root, kind, sample, source, edited["identity"], contract_ids[kind]))
        check_file(sample["image"])
        samples[token] = {"split": "train", "kind": edited["variants"][sample["variant"]]["kind"],
                          "source_token": source_token, "log_name": original["log_name"],
                          "feature_path": str(feature.parent), "metric_path": metrics[token], "files": files,
                          "image": sample["image"], "actor_motion": sample["actor_motion"]}
        if position % 2000 == 0:
            print(f"Extra edited cache inventory: {position}/{len(edited['samples'])}", flush=True)
    for reference in sources:
        check_file(reference, content=True)
    return samples, sources, edited["identity"]


def kinematic_edits(mixed, extra_root):
    """(all t1-t4 edits, extra signed sources, extra edited identity) of a complete mixed manifest's
    t3/t4 edits plus the t1/t2 root; every source must have all 24 edits."""
    if mixed["protocol"] != PROTOCOL:
        raise ValueError("Kinematic SFT preparation requires the complete mixed manifest")
    edits = {t: s for t, s in mixed["samples"].items() if s["kind"] != "original"}
    extra, extra_sources, extra_identity = edited_root_samples(extra_root, mixed["samples"])
    if set(edits) & set(extra):
        raise ValueError("The extra edited root repeats edits of the mixed manifest")
    edits.update(extra)
    source_tokens = sorted({s["source_token"] for s in edits.values()})
    grid = {f"{source}__{kind}_{actor}_t{step}" for source in source_tokens for kind in ("hazardous", "benign")
            for actor in SFT_ACTORS for step in KINEMATIC_SFT_TIMESTEPS}
    if set(edits) != grid:
        raise ValueError("Kinematic SFT requires all 24 edits (hazardous/benign x 3 actors x t1-t4) of every source")
    for token, sample in edits.items():
        edit_key(token, sample, KINEMATIC_SFT_TIMESTEPS)
    return edits, extra_sources, extra_identity


def kinematic_sft_manifest(mixed, extra_root, targets_path, output, original_pool=None, pool_seed=0, variant_seed=0,
                           training_kinds=None):
    """ReactTrain view: clean validation, the selected t1-t4 edits of every source and originals.

    Every source keeps, per actor, the hazardous and benign edits of its two dealt timesteps; there
    are no held-out sources. By default every original training sample is included; ``original_pool``
    keeps a fixed seeded subset of that many. The frozen kinematic target artifact is signed as a source.
    ``training_kinds`` (ablations; default all three) keeps only those kinds, always with the hazardous
    edits; it is recorded sorted in the view.
    """
    from navsim.planning.training.recogdrive_kinematic_sft import load_targets
    kinds = KINEMATIC_SFT_KINDS if training_kinds is None else tuple(sorted(set(training_kinds)))
    if "hazardous" not in kinds or not set(kinds) <= set(KINEMATIC_SFT_KINDS) or (
            training_kinds is not None and kinds == KINEMATIC_SFT_KINDS):
        raise ValueError("Training kinds keep the hazardous edits and drop the benign edits and/or originals "
                         "(the default view keeps all three)")
    if "original" not in kinds and original_pool is not None:
        raise ValueError("A view without originals has no original pool")
    edits, extra_sources, extra_identity = kinematic_edits(mixed, extra_root)
    source_tokens = sorted({s["source_token"] for s in edits.values()})
    kept = kinematic_sft_edits(source_tokens, variant_seed, [kind for kind in ("hazardous", "benign") if kind in kinds])
    candidates = [token for token, sample in mixed["samples"].items()
                  if sample["kind"] == "original" and sample["split"] == "train"]
    chosen = candidates if "original" in kinds else []
    if original_pool is not None:
        selected = set(select_original_pool(candidates, original_pool, pool_seed))
        chosen = [token for token in candidates if token in selected]
    samples = {t: s for t, s in mixed["samples"].items() if s["split"] == "validation"}
    for token in [*kept, *chosen]:
        if token in samples:
            raise ValueError(f"Duplicate kinematic SFT sample: {token}")
        samples[token] = edits[token] if token in edits else mixed["samples"][token]
    targets_path = Path(targets_path).expanduser().resolve()
    target_value = json.loads(targets_path.read_text())
    value = {"protocol": KINEMATIC_SFT_PROTOCOL, "feature_cache": str(output / "features"),
             "metric_cache": str(output / "metrics"), "samples": samples,
             "original_sources": {token: mixed["samples"][token] for token in source_tokens},
             "variant_selection": {"selection": TIMESTEP_PAIR_SELECTION, "seed": variant_seed,
                                   "pairs": [list(pair) for pair in TIMESTEP_PAIRS], "edits_sha256": digest(kept)},
             "sources": [*mixed["sources"], *extra_sources, stamp(targets_path, content=True)],
             "kinematic_targets": {"path": str(targets_path), "artifact_sha256": target_value.get("artifact_sha256"),
                                   "settings": target_value.get("settings")},
             "parent_identity": mixed["artifact_sha256"], "edited_manifest_identity": mixed["edited_manifest_identity"],
             "extra_edited_manifest_identity": extra_identity}
    if training_kinds is not None:  # absent from the default view, which keeps its identity
        value["training_kinds"] = list(kinds)
    if original_pool is not None:
        value["original_pool"] = {"selection": ORIGINAL_POOL_SELECTION, "seed": pool_seed, "count": original_pool,
                                  "candidates": len(candidates), "candidates_sha256": digest(sorted(candidates)),
                                  "tokens_sha256": digest(sorted(chosen))}
    value["counts"] = sample_counts(samples, value["original_sources"])
    validate_kinematic_sft(value)
    load_targets(value)  # artifact checksum, cache identities and exact hazardous coverage

    def verify_sample(sample):
        for reference in sample["files"]:
            check_file(reference)
    inventory = [*samples.values(), *value["original_sources"].values()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, _ in enumerate(pool.map(verify_sample, inventory), 1):
            if index % 10000 == 0:
                print(f"Kinematic SFT cache inventory: {index}/{len(inventory)}", flush=True)
    return {**value, "artifact_sha256": digest(value)}


def publish(manifest, output):
    if output.exists():
        if not (output / "manifest.json").is_file() or load_manifest(output / "manifest.json") != manifest:
            raise FileExistsError("Output already exists without this complete manifest; use a new --output")
        for token, sample in manifest["samples"].items():
            link = output / "features" / sample["log_name"] / token
            if not link.is_symlink() or str(link.resolve()) != sample["feature_path"]:
                raise ValueError(f"Changed mixed feature link: {link}")
        metrics, _ = metric_index(output / "metrics")
        if metrics != {t: s["metric_path"] for t, s in manifest["samples"].items()}:
            raise ValueError("Changed mixed metric index")
        return
    output.mkdir(parents=True)
    for index, (token, sample) in enumerate(manifest["samples"].items(), 1):
        link = output / "features" / sample["log_name"] / token
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(sample["feature_path"], target_is_directory=True)
        if index % 20000 == 0:
            print(f"Linked cached samples: {index}/{len(manifest['samples'])}", flush=True)
    payload = "file_name\n" + "\n".join(s["metric_path"] for s in manifest["samples"].values()) + "\n"
    atomic_write(output / "metrics/metadata/metric_cache_metadata.csv", lambda f: f.write(payload.encode()))
    write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-validation", type=Path, default=EXP / "recogdrive_agent_cache_dir_train_validation.json")
    parser.add_argument("--original-metrics", type=Path, default=EXP / "metric_cache_train")
    parser.add_argument("--edited-root", type=Path, default=EXP / "recogdrive_edited_training/navtrain_selected_t34_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--population", choices=("mixed", "edited-only", "hazard-sft", "kinematic-sft"), default="mixed")
    parser.add_argument("--mixed-manifest", type=Path, help="Reuse a validated complete mixed manifest")
    parser.add_argument("--holdout-count", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--edited-only-manifest", type=Path, help="Hazard SFT: the edited-only split to extend")
    parser.add_argument("--hazard-targets", type=Path, help="Hazard SFT: prepare_hazard_sft_targets.py output")
    parser.add_argument("--sft-original-pool", type=int,
                        help="Hazard SFT: keep only this many originals, a fixed seeded subset (default: all)")
    parser.add_argument("--sft-pool-seed", type=int, default=0, help="Hazard SFT: seed of --sft-original-pool")
    parser.add_argument("--extra-edited-root", type=Path,
                        help="Kinematic SFT: the prepared t1/t2 edited cache root beside the mixed manifest's t3/t4 edits")
    parser.add_argument("--kinematic-targets", type=Path, help="Kinematic SFT: prepare_kinematic_sft_targets.py output")
    parser.add_argument("--variant-seed", type=int, default=0, help="Kinematic SFT: seed of the per-actor timestep pairs")
    parser.add_argument("--training-kinds", nargs="+", choices=KINEMATIC_SFT_KINDS,
                        help="Kinematic SFT ablations: keep only these training kinds, including hazardous (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Validate the full inventory without writing links or indexes")
    args = parser.parse_args()
    if args.population != "kinematic-sft" and (args.extra_edited_root or args.kinematic_targets or args.variant_seed):
        parser.error("--extra-edited-root/--kinematic-targets/--variant-seed are only supported with "
                     "--population kinematic-sft")
    if args.population != "kinematic-sft" and args.training_kinds:
        parser.error("--training-kinds is only supported with --population kinematic-sft")
    if args.population == "hazard-sft":
        if not (args.mixed_manifest and args.edited_only_manifest and args.hazard_targets):
            parser.error("--population hazard-sft requires --mixed-manifest, --edited-only-manifest and --hazard-targets")
        if (args.sft_original_pool is not None and args.sft_original_pool < 1) or args.sft_pool_seed < 0:
            parser.error("--sft-original-pool must be positive and --sft-pool-seed nonnegative")
    elif args.population == "kinematic-sft":
        if not (args.mixed_manifest and args.extra_edited_root and args.kinematic_targets):
            parser.error("--population kinematic-sft requires --mixed-manifest, --extra-edited-root and "
                         "--kinematic-targets")
        if args.edited_only_manifest or args.hazard_targets:
            parser.error("--edited-only-manifest/--hazard-targets are only supported with --population hazard-sft")
        if ((args.sft_original_pool is not None and args.sft_original_pool < 1) or args.sft_pool_seed < 0
                or args.variant_seed < 0):
            parser.error("--sft-original-pool must be positive and --sft-pool-seed/--variant-seed nonnegative")
        if args.training_kinds and ("hazardous" not in args.training_kinds
                                    or set(args.training_kinds) == set(KINEMATIC_SFT_KINDS)):
            parser.error("--training-kinds must include hazardous and drop benign and/or original")
        if args.training_kinds and "original" not in args.training_kinds and args.sft_original_pool is not None:
            parser.error("--sft-original-pool needs original in --training-kinds")
    elif args.edited_only_manifest or args.hazard_targets or args.sft_original_pool is not None:
        parser.error("--edited-only-manifest/--hazard-targets/--sft-original-pool are only supported with "
                     "--population hazard-sft")
    default_root = {"edited-only": "recogdrive_edited_only_training", "hazard-sft": "recogdrive_hazard_sft",
                    "kinematic-sft": "recogdrive_kinematic_sft"}.get(args.population, "recogdrive_mixed_training")
    default_name = "navtrain_selected_t34_v1"
    if args.population == "hazard-sft":
        ranking = "_pdms" if json.loads(args.hazard_targets.read_text()).get("ranking") == "pdms" else ""
        pool = "" if args.sft_original_pool is None else f"_pool{args.sft_original_pool}"
        if pool and args.sft_pool_seed:
            pool += f"_seed{args.sft_pool_seed}"
        default_name = f"navtrain_selected_t34{ranking}{pool}_v1"  # Each arm and original pool has its own view.
    elif args.population == "kinematic-sft":
        pool = "" if args.sft_original_pool is None else f"_pool{args.sft_original_pool}"
        if pool and args.sft_pool_seed:
            pool += f"_seed{args.sft_pool_seed}"
        variants = f"_variants{args.variant_seed}" if args.variant_seed else ""
        kinds = f"_kinds-{'-'.join(sorted(set(args.training_kinds)))}" if args.training_kinds else ""
        default_name = f"navtrain_selected_t1234{pool}{variants}{kinds}_v1"
    output = (args.output or EXP / default_root / default_name).expanduser().resolve()
    print("Checking original and edited cache coverage...", flush=True)
    manifest = (load_manifest(args.mixed_manifest) if args.mixed_manifest else
                build_manifest(args.original_validation.resolve(), args.original_metrics.resolve(), args.edited_root.resolve(), output))
    if args.population == "edited-only":
        manifest = edited_only_manifest(manifest, output, args.holdout_count, args.split_seed)
    elif args.population == "hazard-sft":
        manifest = hazard_sft_manifest(manifest, load_manifest(args.edited_only_manifest), args.hazard_targets, output,
                                       args.sft_original_pool, args.sft_pool_seed)
    elif args.population == "kinematic-sft":
        manifest = kinematic_sft_manifest(manifest, args.extra_edited_root, args.kinematic_targets, output,
                                          args.sft_original_pool, args.sft_pool_seed, args.variant_seed,
                                          training_kinds=args.training_kinds)
    elif args.mixed_manifest:
        parser.error("--mixed-manifest is only supported with --population edited-only or hazard-sft")
    if not args.dry_run:
        publish(manifest, output)
    print(json.dumps({"manifest": str(output / "manifest.json"), "dry_run": args.dry_run, **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
