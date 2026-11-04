"""Manifest and input checks for original + edited stage-3 training.

The original loaders and cache-preparation contracts remain unchanged. This
module describes a separate, immutable view of their already prepared files.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path


PROTOCOL = "recogdrive-original-plus-edits-v1"
EDITED_ONLY_PROTOCOL = "recogdrive-edited-only-braking-v1"
# Edited-only split plus every original training sample outside its held-out sources
#; hazardous targets come from a frozen artifact.
HAZARD_SFT_PROTOCOL = "recogdrive-hazard-sft-v1"
HOLDOUT_PROTOCOLS = (EDITED_ONLY_PROTOCOL, HAZARD_SFT_PROTOCOL)
# Criteria that rank each hazardous edit's GT speed variants: arms A-mj and A-pdms.
HAZARD_TARGET_RANKINGS = ("minimum_jerk", "pdms")
# Optional fixed subset of a hazard SFT view's originals, chosen once when the view is built.
ORIGINAL_POOL_SELECTION = "sha256(original-pool,seed,token)-ascending-v1"
ORIGINAL_POOL_KEYS = {"selection", "seed", "count", "candidates", "candidates_sha256", "tokens_sha256"}
# ReactTrain fine-tuning: clean validation, the selected t1-t4 edits of every
# edited source (no held-out sources) and originals; hazardous targets come from a frozen kinematic artifact.
KINEMATIC_SFT_PROTOCOL = "recogdrive-kinematic-sft-v1"
SFT_ACTORS = ("car", "standing", "walking")
KINEMATIC_SFT_TIMESTEPS = (1, 2, 3, 4)
# Every actor keeps two of the four placements per source: the six pairs dealt in turn over a seeded order.
TIMESTEP_PAIRS = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
TIMESTEP_PAIR_SELECTION = "sha256(timestep-pairs,seed,actor,source)-ascending-round-robin-v1"
VARIANT_SELECTION_KEYS = {"selection", "seed", "pairs", "edits_sha256"}
KINEMATIC_TARGET_KEYS = {"path", "artifact_sha256", "settings"}
# Training kinds of a kinematic SFT view. Ablation views record the sorted subset they keep as
# ``training_kinds`` (always hazardous); views without the record keep all three.
KINEMATIC_SFT_KINDS = ("benign", "hazardous", "original")
# Threads for the I/O-bound file checks of validate_dataset.
DATASET_CHECK_WORKERS = 32
FEATURE_NAMES = ("internvl_feature.gz", "trajectory_target.gz")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def stamp(path, *, content=False):
    # Preserve the named path: some valid edited images are symlinks to an
    # unchanged source image. Their preparation records identify that link path.
    path = Path(path).expanduser().absolute()
    stat = path.stat()
    if not path.is_file() or not stat.st_size:
        raise ValueError(f"Missing/empty mixed-data input: {path}")
    value = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content:
        value["sha256"] = file_hash(path)
    return value


def check_file(record, *, content=False):
    actual = stamp(record["path"], content=content)
    if any(record.get(k) != v for k, v in actual.items()):
        raise ValueError(f"Mixed-data input changed: {record['path']}")


def metric_index(root):
    """Require one unambiguous index and unique sample keys before merging."""
    paths = sorted((Path(root) / "metadata").glob("*.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one metric metadata CSV under {root}")
    result = {}
    with paths[0].open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["file_name"]:
            raise ValueError(f"Unexpected metric metadata header: {paths[0]}")
        for row in reader:
            path = Path(row["file_name"])
            token = path.parent.name
            if not path.is_absolute() or token in result:
                raise ValueError(f"Duplicate/nonabsolute metric cache: {path}")
            # The loader derives the sample key from this path's parent name.
            # Resolving a token-directory symlink here could change that key.
            result[token] = str(path.absolute())
    return result, paths[0]


def sample_counts(samples, original_sources=None):
    counts = Counter()
    for token, sample in samples.items():
        split, kind = sample["split"], sample["kind"]
        if split not in ("train", "validation") or kind not in ("original", "hazardous", "benign"):
            raise ValueError(f"Invalid split/kind for {token}")
        if kind != "original":
            source = (original_sources if original_sources is not None else samples).get(sample["source_token"], {})
            if split != "train" or source.get("kind") != "original" or source.get("split") != "train":
                raise ValueError(f"Edited sample has no original training source: {token}")
            if source["log_name"] != sample["log_name"] or token == sample["source_token"]:
                raise ValueError(f"Edited source/log identity mismatch: {token}")
        elif token != sample["source_token"]:
            raise ValueError(f"Original sample identity mismatch: {token}")
        counts[split] += 1
        if split == "train":
            counts[kind] += 1
    return {name: counts[name] for name in ("original", "hazardous", "benign", "train", "validation")}


def edit_key(token, sample, timesteps=(3, 4)):
    """Decode only the declared actor/timestep variants (twelve for the default timesteps 3/4)."""
    prefix = sample["source_token"] + "__" + sample["kind"] + "_"
    if not token.startswith(prefix):
        raise ValueError(f"Invalid edited identity: {token}")
    key = token[len(prefix):]
    allowed = {f"{actor}_t{step}" for actor in ("car", "standing", "walking") for step in timesteps}
    if key not in allowed:
        raise ValueError(f"Unknown edited variant: {token}")
    if (sample["actor_motion"].get("future_timestep") != int(key[-1]) or
            sample["actor_motion"].get("benign", False) != (sample["kind"] == "benign")):
        raise ValueError(f"Edited motion/variant mismatch: {token}")
    return key


def validate_edited_only(value):
    sources, heldout, split = value["original_sources"], value["heldout_samples"], value["holdout"]
    tokens = split["source_tokens"]
    if (type(split["seed"]) is not int or split["seed"] < 0 or type(split["count"]) is not int
            or not 0 < split["count"] < len(sources) or len(tokens) != split["count"]
            or tokens != sorted(set(tokens)) or not set(tokens) <= set(sources)):
        raise ValueError("Invalid edited-only source holdout")
    selected = sorted(sorted(sources, key=lambda t: (digest([split["seed"], t]), t))[:split["count"]])
    if split.get("selection") != "sha256(seed,source)-ascending-v1" or tokens != selected:
        raise ValueError("Edited-only holdout differs from its declared split seed")
    if any(s.get("source_token") != token or s.get("kind") != "original" or s.get("split") != "train"
           for token, s in sources.items()):
        raise ValueError("Invalid edited-only original source identity")
    train = {t: s for t, s in value["samples"].items() if s["split"] == "train"}
    if any(s["kind"] == "original" for s in train.values()) or set(train) & set(heldout):
        raise ValueError("Original/held-out inputs leaked into edited training")
    groups = {}
    for population, samples in (("train", train), ("holdout", heldout)):
        for token, sample in samples.items():
            source = sources.get(sample["source_token"], {})
            if (source.get("kind") != "original" or source.get("split") != "train"
                    or source.get("log_name") != sample["log_name"] or sample["split"] != "train"
                    or (sample["source_token"] in tokens) != (population == "holdout")):
                raise ValueError("Edited-only source/population mismatch")
            groups.setdefault(sample["source_token"], set()).add((sample["kind"], edit_key(token, sample)))
    expected = {(kind, f"{actor}_t{step}") for kind in ("hazardous", "benign")
                for actor in ("car", "standing", "walking") for step in (3, 4)}
    if set(groups) != set(sources) or any(keys != expected for keys in groups.values()):
        raise ValueError("Every edited-only source requires all twelve variants")
    expected_triplets = [{"source_token": source, "variant": f"{actor}_t{step}",
                         "benign": f"{source}__benign_{actor}_t{step}",
                         "hazardous": f"{source}__hazardous_{actor}_t{step}"}
                        for source in tokens for actor in ("car", "standing", "walking") for step in (3, 4)]
    if value["triplets"] != expected_triplets:
        raise ValueError("Edited-only triplet pairing mismatch")


def select_original_pool(candidates, count, seed):
    """The ``count`` candidate originals ranked first by sha256(original-pool, seed, token), sorted."""
    candidates = sorted(set(candidates))
    if type(count) is not int or type(seed) is not int or seed < 0 or not 0 < count <= len(candidates):
        raise ValueError("The original pool needs 0 < count <= candidates and a nonnegative integer seed")
    return sorted(sorted(candidates, key=lambda t: (digest(["original-pool", seed, t]), t))[:count])


def validate_hazard_sft(value):
    """The edited-only split plus original training samples outside its held-out sources.

    Without ``original_pool`` the view holds every such original; with it, the recorded subset.
    """
    held = set(value["holdout"]["source_tokens"])
    originals = {t for t, s in value["samples"].items() if s["split"] == "train" and s["kind"] == "original"}
    if not originals or any(t in held or value["samples"][t]["source_token"] != t for t in originals):
        raise ValueError("Hazard SFT originals must be training samples outside the held-out sources")
    pool = value.get("original_pool")
    if pool is not None and (
            not isinstance(pool, dict) or set(pool) != ORIGINAL_POOL_KEYS
            or pool["selection"] != ORIGINAL_POOL_SELECTION
            or any(type(pool[k]) is not int for k in ("seed", "count", "candidates")) or pool["seed"] < 0
            or not 0 < pool["count"] <= pool["candidates"] or pool["count"] != len(originals)
            or pool["tokens_sha256"] != digest(sorted(originals))):
        raise ValueError("Invalid hazard SFT original pool record")
    validate_edited_only({**value, "samples": {t: s for t, s in value["samples"].items() if t not in originals}})
    reference = value.get("hazard_targets")
    if (not isinstance(reference, dict) or set(reference) != {"path", "artifact_sha256", "ranking"}
            or reference["ranking"] not in HAZARD_TARGET_RANKINGS
            or not any(r.get("path") == reference["path"] and "sha256" in r for r in value["sources"])):
        raise ValueError("Hazard SFT manifests require their signed hazardous target artifact")


def select_timestep_pairs(sources, seed):
    """{actor: {source: (t_a, t_b)}}: per actor, the sources ranked by sha256(timestep-pairs, seed,
    actor, source) are dealt the six timestep pairs in turn, so every placement is kept for half of
    the sources (within one) of every actor."""
    sources = sorted(set(sources))
    if type(seed) is not int or seed < 0 or not sources:
        raise ValueError("Timestep pairs need sources and a nonnegative integer seed")
    selection = {}
    for actor in SFT_ACTORS:
        ranked = sorted(sources, key=lambda token: (digest(["timestep-pairs", seed, actor, token]), token))
        selection[actor] = {token: TIMESTEP_PAIRS[index % len(TIMESTEP_PAIRS)] for index, token in enumerate(ranked)}
    return selection


def kinematic_sft_edits(sources, seed, kinds=("hazardous", "benign")):
    """Sorted IDs of the selected edits: those of ``kinds`` (default hazardous and benign) of every
    kept actor/timestep."""
    return sorted(f"{source}__{kind}_{actor}_t{step}"
                  for actor, pairs in select_timestep_pairs(sources, seed).items()
                  for source, pair in pairs.items() for step in pair for kind in kinds)


def validate_kinematic_sft(value):
    """ReactTrain view: clean validation, the selected t1-t4 edits of every source and originals.

    There are no held-out sources. Without ``original_pool`` the view holds every original training
    sample; with it, the recorded subset. An ablation view's ``training_kinds`` drops the benign edits
    of the selection and/or every original training sample (and the pool record).
    """
    if any(key in value for key in ("heldout_samples", "holdout", "triplets")):
        raise ValueError("Kinematic SFT views have no held-out sources")
    kinds = list(KINEMATIC_SFT_KINDS)
    if "training_kinds" in value:
        kinds = value["training_kinds"]
        if (not isinstance(kinds, list) or not all(isinstance(kind, str) for kind in kinds)
                or kinds != sorted(set(kinds)) or "hazardous" not in kinds or not set(kinds) < set(KINEMATIC_SFT_KINDS)):
            raise ValueError("Invalid kinematic SFT training kinds")
    sources, samples = value["original_sources"], value["samples"]
    if not sources or any(s.get("source_token") != token or s.get("kind") != "original" or s.get("split") != "train"
                          for token, s in sources.items()):
        raise ValueError("Invalid kinematic SFT original source identity")
    selection = value.get("variant_selection")
    if (not isinstance(selection, dict) or set(selection) != VARIANT_SELECTION_KEYS
            or selection["selection"] != TIMESTEP_PAIR_SELECTION or type(selection["seed"]) is not int
            or selection["seed"] < 0 or selection["pairs"] != [list(pair) for pair in TIMESTEP_PAIRS]):
        raise ValueError("Invalid kinematic SFT variant selection record")
    expected = kinematic_sft_edits(sources, selection["seed"], [kind for kind in ("hazardous", "benign") if kind in kinds])
    edits = sorted(token for token, sample in samples.items() if sample["kind"] != "original")
    if edits != expected or selection["edits_sha256"] != digest(expected):
        raise ValueError("Kinematic SFT edits differ from the declared timestep-pair selection")
    for token in edits:
        sample = samples[token]
        edit_key(token, sample, KINEMATIC_SFT_TIMESTEPS)  # The ID's source and kind are the record's.
        if sample["split"] != "train" or sources[sample["source_token"]]["log_name"] != sample["log_name"]:
            raise ValueError(f"Kinematic SFT edit/source mismatch: {token}")
    originals = {t for t, s in samples.items() if s["split"] == "train" and s["kind"] == "original"}
    if "original" not in kinds:
        if originals or "original_pool" in value:
            raise ValueError("Kinematic SFT views without originals hold no original training sample or pool")
    else:
        if not originals or any(samples[t]["source_token"] != t for t in originals):
            raise ValueError("Kinematic SFT originals must be training samples")
        pool = value.get("original_pool")
        if pool is not None and (
                not isinstance(pool, dict) or set(pool) != ORIGINAL_POOL_KEYS
                or pool["selection"] != ORIGINAL_POOL_SELECTION
                or any(type(pool[k]) is not int for k in ("seed", "count", "candidates")) or pool["seed"] < 0
                or not 0 < pool["count"] <= pool["candidates"] or pool["count"] != len(originals)
                or pool["tokens_sha256"] != digest(sorted(originals))):
            raise ValueError("Invalid kinematic SFT original pool record")
    reference = value.get("kinematic_targets")
    if (not isinstance(reference, dict) or set(reference) != KINEMATIC_TARGET_KEYS
            or not isinstance(reference["settings"], dict)
            or not any(r.get("path") == reference["path"] and "sha256" in r for r in value["sources"])):
        raise ValueError("Kinematic SFT manifests require their signed kinematic target artifact")


def load_manifest(path, *, verify_sources=True):
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text())
    identity = value.pop("artifact_sha256", None)
    if identity != digest(value) or value.get("protocol") not in (PROTOCOL, *HOLDOUT_PROTOCOLS, KINEMATIC_SFT_PROTOCOL):
        raise ValueError("Mixed-data manifest checksum/protocol mismatch")
    edited_only = value["protocol"] == EDITED_ONLY_PROTOCOL
    if edited_only:
        validate_edited_only(value)
    elif value["protocol"] == HAZARD_SFT_PROTOCOL:
        validate_hazard_sft(value)
    elif value["protocol"] == KINEMATIC_SFT_PROTOCOL:
        validate_kinematic_sft(value)
    # These views keep their edits' source originals apart from the (sampled) training originals.
    grouped = value["protocol"] in (*HOLDOUT_PROTOCOLS, KINEMATIC_SFT_PROTOCOL)
    if value["counts"] != sample_counts(value["samples"], value["original_sources"] if grouped else None):
        raise ValueError("Mixed-data manifest counts do not match its samples")
    if value["protocol"] == KINEMATIC_SFT_PROTOCOL and "training_kinds" in value:
        # Ablation views (validated above): samples of every declared kind and, by the validation, of no other.
        if any(not value["counts"][kind] for kind in value["training_kinds"]):
            raise ValueError("Kinematic SFT ablation views require training samples of every declared kind")
    elif (not edited_only and not value["counts"]["original"]) or not value["counts"]["hazardous"] or not value["counts"]["benign"]:
        raise ValueError("Mixed data requires original, hazardous and benign training samples")
    if verify_sources:
        for reference in value["sources"]:
            check_file(reference, content=True)
    return {**value, "artifact_sha256": identity}


def validate_dataset(manifest, train_data, val_data, metric_paths, feature_root, metric_root):
    """Validate what the actual loaders resolved, including held-out validation."""
    if (str(Path(feature_root).resolve()) != manifest["feature_cache"] or
            str(Path(metric_root).resolve()) != manifest["metric_cache"]):
        raise ValueError("Mixed-data feature/metric cache roots differ from the manifest")

    def check(item):
        token, sample, data = item
        if str(data._valid_cache_paths[token].resolve()) != sample["feature_path"]:
            raise ValueError(f"Mixed feature path differs for {token}")
        if metric_paths.get(token) != sample["metric_path"]:
            raise ValueError(f"Mixed metric path differs for {token}")
        for reference in sample["files"]:
            check_file(reference)

    items = []
    for split, data in (("train", train_data), ("validation", val_data)):
        expected = {k: v for k, v in manifest["samples"].items() if v["split"] == split}
        if set(data.tokens) != set(expected) or len(data.tokens) != len(expected):
            raise ValueError(f"Mixed-data {split} coverage differs from the manifest")
        items += [(token, sample, data) for token, sample in expected.items()]
    # The same per-sample checks, run concurrently on the shared filesystem. Results are consumed
    # in order, so the first failing sample raises exactly as a sequential pass would.
    pool = ThreadPoolExecutor(max_workers=DATASET_CHECK_WORKERS)
    try:
        for _ in pool.map(check, items):
            pass
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def mixed_image_lookup(cache_paths, log_root, sensor_root, manifest):
    from navsim.agents.recogdrive.recogdrive_motion_inputs import build_image_lookup

    samples = manifest["samples"]
    expected = {k for k, v in samples.items() if v["split"] == "train"}
    if set(cache_paths) != expected:
        raise ValueError("Mixed joint-dropout image coverage differs from training")
    originals = {k: v for k, v in cache_paths.items() if samples[k]["kind"] == "original"}
    lookup, sources = build_image_lookup(originals, log_root, sensor_root)
    for token in sorted(expected - originals.keys()):
        record = samples[token]["image"]
        check_file(record)
        lookup[token] = record["path"]
    # Keep the original feature-contract code/file hashes intact for original
    # runs, while binding mixed runs to the images in their signed manifest.
    sources.append({"mixed_data_identity": manifest["artifact_sha256"]})
    return lookup, sources
