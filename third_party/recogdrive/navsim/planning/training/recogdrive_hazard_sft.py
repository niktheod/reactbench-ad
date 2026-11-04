"""Hazard SFT: frozen targets, training population and monitoring.

Each hazardous edit is supervised toward the best same-path speed sibling of its logged GT
(the experiment-15 family) when that sibling beats the GT by more than the family margin,
otherwise toward the GT itself. Arm A-mj ranks the family with the minimum-jerk training reward
V0, arm A-pdms with the unmodified NAVSIM PDMS; both score every member with both criteria and
record planned/executed contacts. Original and benign samples keep their GT. Training uses the
stage-2 objective; nothing here changes the stage-3 rewards or other experiments.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import math
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

from .recogdrive_mixed_data import HAZARD_SFT_PROTOCOL, HAZARD_TARGET_RANKINGS as RANKINGS, check_file, digest

TARGETS_PROTOCOL = "recogdrive-hazard-sft-targets-v2"
SAMPLER = "all-edits-plus-disjoint-original-chunks-of-seeded-pool-passes-v2"
POOL_ORDER_KEY = 1 << 32  # Seeds the passes' pool permutations apart from the per-epoch shuffles.
MONITORING_PROTOCOL = "recogdrive-hazard-sft-triplet-l1-minimum-jerk-pdms-v2"
COHORTS = ("original", "benign", "hazardous")
REWARD_COMPONENTS = ("nc", "dac", "ttc", "ep", "comfort", "beta", "beta_backward")
# (stored name, PDMResults field) of the stock PDMS subscores.
PDMS_FIELDS = (("nc", "no_at_fault_collisions"), ("dac", "drivable_area_compliance"), ("ep", "ego_progress"),
               ("ttc", "time_to_collision_within_bound"), ("comfort", "comfort"),
               ("ddc", "driving_direction_compliance"))
CONTACT_FIELDS = ("inserted", "logged_behind", "logged_other")
GAIN_BIN_EDGES = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)  # R_best - R_GT histogram; the last bin is open


def reward_settings():
    """V0: the stage-3 minimum-jerk training reward (EP/TTC/comfort 10/5/2) and frozen family defaults."""
    from navsim.evaluate.minimum_jerk_pdm_score import MinimumJerkRewardConfig
    from navsim.evaluate.sibling_targets import SiblingSettings
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorerConfig
    scorer = PDMScorerConfig(progress_weight=10.0, ttc_weight=5.0, comfortable_weight=2.0)
    return scorer, MinimumJerkRewardConfig(), SiblingSettings()


def stock_pdms():
    """Unmodified NAVSIM PDMS: 4 s at 10 Hz, PDM-Closed reference, LQR + bicycle execution, weights 5/5/2."""
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    return sampling, PDMSimulator(sampling), PDMScorer(sampling, PDMScorerConfig())


def scorer_weights(scorer):
    return {name: float(getattr(scorer, name)) for name in (
        "progress_weight", "ttc_weight", "comfortable_weight", "driving_direction_weight")}


def target_protocol(ranking):
    """Selection settings, conventions and code identity recorded in the target artifact."""
    from navsim.evaluate import minimum_jerk_pdm_score, pdm_score, sibling_targets
    from navsim.planning.simulation.planner.pdm_planner.scoring import pdm_scorer
    if ranking not in RANKINGS:
        raise ValueError(f"Unknown hazard target ranking: {ranking}")
    scorer, reward, siblings = reward_settings()
    _, _, stock = stock_pdms()
    code = {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (minimum_jerk_pdm_score, sibling_targets, pdm_score, pdm_scorer)}
    return {"version": 2, "ranking": ranking, "base": "logged-gt-poses-v1",
            "family": [[d, onset] for d, onset in siblings.family()], "margin": float(siblings.margin),
            "selection": "best-member-over-margin-else-gt-v2",
            "tie_order": "decelerations-first-then-smallest-offset-then-earlier-onset-v1",
            "heading": {"sibling": "path-tangent", "gt": "logged"},
            "scorer_weights": scorer_weights(scorer),
            "pdms": {"scorer_weights": scorer_weights(stock._config), "sampling": [40, 0.1],
                     "execution": "pdm-closed-reference-lqr-bicycle-each-member-separately-v1"},
            "contacts": {"planned": "minimum-jerk-reconstruction-10hz-v1", "executed": "pdm-simulator-10hz-v1",
                         "rule": "any-footprint-overlap-excluding-red-lights-and-initial-overlaps-v1",
                         "logged_behind": "nuplan-is-agent-behind-rear-axle-150deg-v1"},
            "minimum_jerk": minimum_jerk_pdm_score.protocol_metadata(reward),
            "family_generator": sibling_targets.protocol_metadata(siblings),
            "implementation": {**code, Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


def hazardous_samples(manifest):
    """(population, sample) per hazardous edit of the training and held-out sources."""
    samples = {t: ("train", s) for t, s in manifest["samples"].items() if s["kind"] == "hazardous"}
    samples.update({t: ("holdout", s) for t, s in manifest["heldout_samples"].items() if s["kind"] == "hazardous"})
    return samples


def sample_file(sample, name):
    """Frozen stamp of one prepared file: ``internvl_feature.gz``, ``trajectory_target.gz`` or ``metric``."""
    matches = [r for r in sample["files"]
               if (r["path"] == sample["metric_path"] if name == "metric" else Path(r["path"]).name == name)]
    if len(matches) != 1:
        raise ValueError(f"Expected one prepared {name} file for {sample['feature_path']}")
    return matches[0]


def load_gt(reference):
    """Logged GT poses (x, y, heading) at 0.5-4.0 s from a stamped ``trajectory_target.gz``."""
    check_file(reference)
    with gzip.open(reference["path"], "rb") as handle:
        gt = np.asarray(pickle.load(handle)["trajectory"], dtype=np.float64)
    if gt.shape != (8, 3) or not np.isfinite(gt).all():
        raise ValueError(f"Invalid GT trajectory: {reference['path']}")
    return gt


def load_metric_cache(reference):
    check_file(reference)
    with lzma.open(reference["path"], "rb") as handle:
        return pickle.load(handle)


def pdms_result(metric_cache, poses, stock=None):
    """Stock executed PDMS of one trajectory scored on its own (``pdm_score``, PDM-Closed reference)."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    sampling, simulator, scorer = stock_pdms() if stock is None else stock
    result = pdm_score(metric_cache, Trajectory(np.asarray(poses, dtype=np.float64)), sampling, simulator, scorer)
    values = {"score": float(result.score), **{name: float(getattr(result, field)) for name, field in PDMS_FIELDS}}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Nonfinite stock PDMS or subscore")
    return values


def executed_states(metric_cache, poses, stock=None):
    """The trajectory's LQR + bicycle states exactly as ``pdm_score`` simulates them (beside the reference)."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    sampling, simulator, _ = stock_pdms() if stock is None else stock
    initial = metric_cache.ego_state
    proposals = np.stack([
        get_trajectory_as_array(metric_cache.trajectory, sampling, initial.time_point),
        get_trajectory_as_array(transform_trajectory(Trajectory(np.asarray(poses, dtype=np.float64)), initial),
                                sampling, initial.time_point)])
    return simulator.simulate_proposals(proposals, initial)[1]


def planned_states(metric_cache, poses, reward):
    """The minimum-jerk reconstruction that V0 scores (10 Hz global states), or None when infeasible."""
    from navsim.evaluate.minimum_jerk_pdm_score import motion_states
    from navsim.evaluate.minimum_jerk_trajectory import reconstruct_motion
    initial = metric_cache.ego_state
    motion = reconstruct_motion(np.asarray(poses, dtype=np.float64)[:, :2],
                                (initial.rear_axle.x, initial.rear_axle.y, initial.rear_axle.heading),
                                initial.dynamic_car_state.rear_axle_velocity_2d.x, reward.reconstruction())
    return motion_states(motion) if motion.feasible else None


def contacts(states, observation, inserted_token):
    """Which tracks the ego footprint touches at 0-4 s, any fault: the inserted actor, and logged actors
    behind the ego (centroid within 30 deg of straight behind the rear axle) or elsewhere. Red lights and
    tracks already overlapping the ego at 0 s (stock initial collisions) are skipped. None without states."""
    if states is None:
        return None
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.observation.idm.utils import is_agent_behind
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        coords_array_to_polygon_array, state_array_to_coords_array)
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    polygons = coords_array_to_polygon_array(state_array_to_coords_array(states[None], get_pacifica_parameters()))[0]
    result = dict.fromkeys(CONTACT_FIELDS, False)
    seen = set(observation.collided_track_ids)
    for time_idx, polygon in enumerate(polygons):
        occupancy = observation[time_idx]
        for token in occupancy.intersects(polygon):
            if observation.red_light_token in token or token in seen:
                continue
            seen.add(token)
            if token == inserted_token:
                result["inserted"] = True
                continue
            centroid = occupancy[token].centroid
            behind = is_agent_behind(StateSE2(*states[time_idx, StateIndex.STATE_SE2]),
                                     StateSE2(centroid.x, centroid.y, 0.))
            result["logged_behind" if behind else "logged_other"] = True
    return result


def score_family(metric_cache, gt, inserted_token, settings=None):
    """Score the GT (logged heading) and its same-path speed siblings in one hazardous edit's world.

    Every member gets V0 (one batch, candidate-independent; without the backward-waypoint penalty, as
    in the experiment-15 family), the stock PDMS scored on its own, and the contacts of its planned
    (V0 reconstruction) and executed (LQR + bicycle) motion. Returns
    (poses, family) with per-member lists in family order, or (None, reason) for an infeasible GT
    fit (no siblings exist). Never mutates the metric cache.
    """
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.minimum_jerk_pdm_score import minimum_jerk_scores
    from navsim.evaluate.sibling_targets import reconstruct_base, sibling_waypoints
    scorer, reward, siblings = reward_settings() if settings is None else settings
    gt = np.asarray(gt, dtype=np.float64)
    speed = metric_cache.ego_state.dynamic_car_state.rear_axle_velocity_2d.x
    base, reason = reconstruct_base(gt[:, :2], speed, reward.reconstruction())
    if base is None:
        return None, reason
    members = siblings.family()
    poses = [gt] + [sibling_waypoints(base, d, onset, siblings) for d, onset in members[1:]]
    v0 = minimum_jerk_scores(metric_cache, [Trajectory(p) for p in poses], gt, scorer, reward, charge_backward=False)
    if v0[0].infeasible_fit:
        raise RuntimeError("Sibling and reward reconstructions disagree on the base")
    stock = stock_pdms()
    pdms = [pdms_result(metric_cache, p, stock) for p in poses]
    planned = [contacts(planned_states(metric_cache, p, reward), metric_cache.observation, inserted_token)
               for p in poses]
    executed = [contacts(executed_states(metric_cache, p, stock), metric_cache.observation, inserted_token)
                for p in poses]
    return poses, {
        "v0": [float(r.score) for r in v0], "v0_infeasible_fit": [bool(r.infeasible_fit) for r in v0],
        "pdms": {name: [m[name] for m in pdms] for name in ("score", *(n for n, _ in PDMS_FIELDS))},
        "contacts": {kind: {field: [None if c is None else c[field] for c in rows] for field in CONTACT_FIELDS}
                     for kind, rows in (("planned", planned), ("executed", executed))},
        "inserted_tracked": inserted_token in metric_cache.observation.unique_objects}


def ranking_scores(family, ranking):
    if ranking == "minimum_jerk":
        return family["v0"]
    if ranking == "pdms":
        return family["pdms"]["score"]
    raise ValueError(f"Unknown hazard target ranking: {ranking}")


def tie_order(members):
    """Decelerations (and the base) before speed-ups, then the smallest |offset|, then the earlier onset."""
    return sorted(range(len(members)), key=lambda i: (members[i][0] < 0, abs(members[i][0]), members[i][1]))


def select_member(members, scores, margin):
    """Index of the best member if it beats the base (index 0) by more than ``margin``; None keeps the GT."""
    best = max(tie_order(members), key=lambda i: scores[i])  # first maximum in tie order
    return best if scores[best] > scores[0] + margin else None


def select_hazard_target(metric_cache, gt, inserted_token, ranking="minimum_jerk", settings=None):
    """(target poses, record) for one hazardous edit; never mutates the metric cache.

    The GT is kept, with its logged heading, when its reconstruction is infeasible, the whole
    family scores 0, or no sibling beats it by more than the margin; sibling targets carry the
    path-tangent heading.
    """
    if ranking not in RANKINGS:
        raise ValueError(f"Unknown hazard target ranking: {ranking}")
    _, _, siblings = reward_settings() if settings is None else settings
    gt = np.asarray(gt, dtype=np.float64)
    poses, family = score_family(metric_cache, gt, inserted_token, settings)
    if poses is None:
        return gt, {"origin": "gt", "gt_kept_reason": "infeasible_fit", "infeasible_reason": family,
                    "selected": None, "family": None}
    scores = ranking_scores(family, ranking)
    selected = select_member(siblings.family(), scores, siblings.margin)
    if selected is None:
        reason = "all_zero_family" if all(score == 0 for score in scores) else "no_sibling_over_margin"
        return gt, {"origin": "gt", "gt_kept_reason": reason, "infeasible_reason": None, "selected": None,
                    "family": family}
    return np.asarray(poses[selected], dtype=np.float64), {
        "origin": "sibling", "gt_kept_reason": None, "infeasible_reason": None, "selected": selected,
        "family": family}


def _grid_positions(members):
    """Position of each member within its onset's sorted offsets (for 'within one step' agreement)."""
    positions = {}
    for onset in {o for _, o in members}:
        ordered = sorted((d, i) for i, (d, o) in enumerate(members) if o == onset)
        positions.update({i: position for position, (_, i) in enumerate(ordered)})
    return positions


def _group_summary(rows, ranking, members, margin):
    other = "pdms" if ranking == "minimum_jerk" else "minimum_jerk"
    positions = _grid_positions(members)
    feasible = [r for r in rows if r["family"] is not None]
    gains, ties, agree, near = [], 0, 0, 0
    contact_counts = {f"{who}_{kind}_{field}": 0 for who in ("gt", "target") for kind in ("planned", "executed")
                      for field in CONTACT_FIELDS}
    for row in feasible:
        scores = ranking_scores(row["family"], ranking)
        gains.append(max(scores) - scores[0])
        ties += sum(score == max(scores) for score in scores) > 1
        theirs = select_member(members, ranking_scores(row["family"], other), margin)
        mine = row["selected"]
        agree += mine == theirs
        near += (mine is not None and theirs is not None and members[mine][1] == members[theirs][1]
                 and abs(positions[mine] - positions[theirs]) <= 1)
        for who, index in (("gt", 0), ("target", 0 if mine is None else mine)):
            for kind in ("planned", "executed"):
                for field in CONTACT_FIELDS:
                    contact_counts[f"{who}_{kind}_{field}"] += bool(row["family"]["contacts"][kind][field][index])
    edges = (0., *GAIN_BIN_EDGES)
    histogram = {f"[{low},{high})": sum(low <= g < high for g in gains) for low, high in zip(edges, edges[1:])}
    histogram[f"[{edges[-1]},inf)"] = sum(g >= edges[-1] for g in gains)
    chosen = [members[r["selected"]] for r in rows if r["selected"] is not None]
    return {
        "samples": len(rows), "sibling_targets": len(chosen),
        "gt_kept": {reason: sum(r["gt_kept_reason"] == reason for r in rows)
                    for reason in ("infeasible_fit", "all_zero_family", "no_sibling_over_margin")},
        "inserted_not_tracked": sum(not r["family"]["inserted_tracked"] for r in feasible),
        "gain_histogram": histogram, "gain_median": float(np.median(gains)) if gains else None,
        "exact_ties": int(ties),
        "chosen_deceleration": sum(d > 0 for d, _ in chosen), "chosen_speed_up": sum(d < 0 for d, _ in chosen),
        "late_onset": sum(onset > 0 for _, onset in chosen),
        "mean_chosen_offset_mps2": float(np.mean([d for d, _ in chosen])) if chosen else None,
        "contacts": contact_counts,
        f"agreement_with_{other}": {"same_choice": int(agree), "same_onset_within_one_step": int(near),
                                    "families": len(feasible)}}


def target_summary(samples, ranking, settings=None):
    """Audit per population and per actor/timestep variant (feasible families only where scores are needed)."""
    _, _, siblings = reward_settings() if settings is None else settings
    members = siblings.family()
    groups = {}
    for record in samples.values():
        for key in (record["population"], f"{record['population']}/{record['variant']}"):
            groups.setdefault(key, []).append(record)
    return {key: _group_summary(rows, ranking, members, siblings.margin) for key, rows in sorted(groups.items())}


def load_targets(manifest):
    """Verified hazardous targets (sample ID -> float64 [8, 3]) for a hazard SFT manifest."""
    if manifest.get("protocol") != HAZARD_SFT_PROTOCOL:
        raise ValueError("Hazard targets require a hazard SFT manifest")
    reference = manifest["hazard_targets"]
    stamps = [r for r in manifest["sources"] if r.get("path") == reference["path"] and "sha256" in r]
    if len(stamps) != 1:
        raise ValueError("Hazard SFT manifest must sign its target artifact exactly once")
    data = Path(reference["path"]).read_bytes()
    if len(data) != stamps[0]["size"] or hashlib.sha256(data).hexdigest() != stamps[0]["sha256"]:
        raise ValueError(f"Hazard target artifact changed: {reference['path']}")
    value = json.loads(data)
    identity = value.pop("artifact_sha256", None)
    if (identity != digest(value) or identity != reference["artifact_sha256"]
            or value.get("protocol") != TARGETS_PROTOCOL or value.get("ranking") != reference["ranking"]):
        raise ValueError("Hazard target artifact checksum/protocol/ranking mismatch")
    if value["edited_only_manifest"] != manifest["parent_identity"]:
        raise ValueError("Hazard targets were built for a different edited-only split")
    expected = hazardous_samples(manifest)
    if set(value["samples"]) != set(expected):
        raise ValueError("Hazard targets must cover exactly the training and held-out hazardous edits")
    targets = {}
    for token, record in value["samples"].items():
        target = np.asarray(record["target"], dtype=np.float64)
        if (target.shape != (8, 3) or not np.isfinite(target).all()
                or record["population"] != expected[token][0]
                or record["source_token"] != expected[token][1]["source_token"]):
            raise ValueError(f"Invalid hazard target record: {token}")
        targets[token] = target
    return targets


class HazardTargetDataset(Dataset):
    """Training samples in sorted token order; hazardous targets replaced by the frozen SFT targets.

    The sorted order makes sampler indices independent of directory listing order on each rank.
    Original and benign samples keep their cached GT; replacements keep the GT tensor dtype.
    """

    def __init__(self, base, kinds, targets):
        self.base = base
        self.tokens = sorted(base.tokens)
        self._valid_cache_paths = base._valid_cache_paths
        self._base_index = {token: index for index, token in enumerate(base.tokens)}
        if len(self._base_index) != len(self.tokens):
            raise ValueError("Duplicate training tokens")
        self.kinds = {token: kinds[token] for token in self.tokens}
        hazardous = [token for token in self.tokens if self.kinds[token] == "hazardous"]
        missing = [token for token in hazardous if token not in targets]
        if missing:
            raise ValueError(f"Hazardous training samples without SFT targets: {missing[:5]}")
        self.targets = {token: np.array(targets[token], dtype=np.float64) for token in hazardous}

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        token = self.tokens[index]
        features, targets, loaded = self.base[self._base_index[token]]
        if loaded != token:
            raise RuntimeError(f"Cached sample identity mismatch: {loaded} != {token}")
        if token in self.targets:
            gt = targets["trajectory"]
            if tuple(gt.shape) != (8, 3):
                raise ValueError(f"Unexpected cached GT shape for {token}: {tuple(gt.shape)}")
            targets = {**targets, "trajectory": torch.tensor(self.targets[token], dtype=gt.dtype)}
        return features, targets, token


class TokenDirectoryDataset(Dataset):
    """Cached samples from explicit token directories (the held-out triplet originals)."""

    def __init__(self, directories, feature_builders, target_builders):
        self._builders = (list(feature_builders), list(target_builders))
        self._valid_cache_paths = {token: Path(path) for token, path in sorted(directories.items())}
        for path in self._valid_cache_paths.values():
            for builder in (*self._builders[0], *self._builders[1]):
                if not (path / f"{builder.get_unique_name()}.gz").is_file():
                    raise FileNotFoundError(f"Missing cached {builder.get_unique_name()} under {path}")
        self.tokens = list(self._valid_cache_paths)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        token = self.tokens[index]
        loaded = []
        for builders in self._builders:
            values = {}
            for builder in builders:
                with gzip.open(self._valid_cache_paths[token] / f"{builder.get_unique_name()}.gz", "rb") as handle:
                    values.update(pickle.load(handle))
            loaded.append(values)
        return loaded[0], loaded[1], token


class EpochOriginalSampler(DistributedSampler):
    """Every edited sample plus ``originals_per_epoch`` originals, disjoint across epochs.

    The view's originals (the pool) are permuted once per pass, seeded by (seed, pass), and each
    pass is cut into ``len(pool) // originals_per_epoch`` disjoint chunks, one per epoch; later
    epochs start a new pass. A pool of exactly k x originals_per_epoch therefore uses every
    original once over k epochs. The epoch's index list depends only on (seed, epoch), is
    identical on every rank, and is shuffled and then sharded like ``DistributedSampler`` (tail
    padded by repetition). Being a DistributedSampler, Lightning keeps it and calls
    ``set_epoch`` before each epoch.
    """

    def __init__(self, dataset, originals_per_epoch, *, seed, num_replicas, rank):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False, seed=seed, drop_last=False)
        kinds = [dataset.kinds[token] for token in dataset.tokens]
        self.edited = np.flatnonzero([kind != "original" for kind in kinds])
        self.originals = np.flatnonzero([kind == "original" for kind in kinds])
        if (type(originals_per_epoch) is not int or not 0 < originals_per_epoch <= len(self.originals)
                or not len(self.edited)):
            raise ValueError("originals_per_epoch must be a positive integer up to the number of originals, "
                             "with edited samples present")
        self.originals_per_epoch = originals_per_epoch
        self.disjoint_epochs = len(self.originals) // originals_per_epoch  # Epochs per pass over the pool.
        self.epoch_size = len(self.edited) + originals_per_epoch
        self.num_samples = math.ceil(self.epoch_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def epoch_originals(self, epoch):
        """The dataset indices of one epoch's originals: one chunk of its pass's pool permutation."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        epoch_pass, chunk = divmod(epoch, self.disjoint_epochs)
        order = np.random.default_rng([int(self.seed), POOL_ORDER_KEY, epoch_pass]).permutation(self.originals)
        return order[chunk * self.originals_per_epoch:(chunk + 1) * self.originals_per_epoch]

    def epoch_indices(self, epoch=None):
        """The full (unsharded, padded) index list of one epoch."""
        epoch = self.epoch if epoch is None else epoch
        chosen = self.epoch_originals(int(epoch))
        rng = np.random.default_rng([int(self.seed), int(epoch)])
        indices = np.concatenate((self.edited, chosen))
        rng.shuffle(indices)
        indices = indices.tolist()
        padding = self.total_size - len(indices)
        while padding > 0:
            indices += indices[:padding]
            padding = self.total_size - len(indices)
        return indices

    def __iter__(self):
        return iter(self.epoch_indices()[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples


def sampler_protocol(manifest, originals_per_epoch, seed):
    """Recorded in the mixed-training protocol, so a resume cannot change the population or draw."""
    edits = manifest["counts"]["hazardous"] + manifest["counts"]["benign"]
    originals = manifest["counts"]["original"]
    return {"version": 2, "sampler": SAMPLER, "seed": seed, "originals_per_epoch": originals_per_epoch,
            "epoch_size": edits + originals_per_epoch, "originals": originals,
            "disjoint_epochs": originals // originals_per_epoch, "original_pool": manifest.get("original_pool"),
            "targets": manifest["hazard_targets"]["artifact_sha256"],
            "ranking": manifest["hazard_targets"]["ranking"], "loss": "stage2-epsilon-mse-per-sample-mean-v1"}


# --- Held-out triplet monitoring (L1 to the training target; V0 and stock PDMS in each input's own world) ---

def monitoring_inputs(manifest, targets):
    """One spec per held-out triplet input; each source's original appears once."""
    specs = {}
    for triplet in manifest["triplets"]:
        token = triplet["source_token"]
        source = manifest["original_sources"][token]
        gt = sample_file(source, "trajectory_target.gz")
        specs.setdefault(token, {"sample_id": token, "token": token, "kind": "original", "variant": None,
                                 "gt": gt, "target": gt, "hazard_target": None,
                                 "metric": sample_file(source, "metric")})
        for kind in ("benign", "hazardous"):
            sample_id = triplet[kind]
            sample = manifest["heldout_samples"][sample_id]
            specs[sample_id] = {
                "sample_id": sample_id, "token": token, "kind": kind, "variant": triplet["variant"], "gt": gt,
                "target": sample_file(sample, "trajectory_target.gz") if kind == "benign" else None,
                "hazard_target": targets[sample_id].tolist() if kind == "hazardous" else None,
                "metric": sample_file(sample, "metric")}
    return specs


def monitoring_contract(manifest, specs):
    scorer, reward, _ = reward_settings()
    _, _, stock = stock_pdms()
    return {"protocol": MONITORING_PROTOCOL, "targets": manifest["hazard_targets"]["artifact_sha256"],
            "inputs": digest(specs), "scorer_weights": scorer_weights(scorer), "minimum_jerk": reward.as_dict(),
            "pdms": {"scorer_weights": scorer_weights(stock._config), "sampling": [40, 0.1],
                     "execution": "pdm-closed-reference-lqr-bicycle-v1"},
            "l1": "mean-abs-error-over-8x3-native-poses-v1"}


def score_monitoring_input(spec, poses):
    """L1 to the input's training target, V0 and stock executed PDMS in its own world (clean GT for EP)."""
    record = {key: spec[key] for key in ("sample_id", "token", "kind", "variant")}
    record.update(l1=None, reward=None, pdms=None, error=None)
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (8, 3) or not np.isfinite(poses).all():
        record["error"] = "invalid native predicted trajectory"
        return record
    errors = []
    try:
        target = (np.asarray(spec["hazard_target"], dtype=np.float64) if spec["kind"] == "hazardous"
                  else load_gt(spec["target"]))
        record["l1"] = float(np.abs(poses - target).mean())
    except Exception as exc:
        errors.append(f"l1: {type(exc).__name__}: {exc}")
    try:
        cache = load_metric_cache(spec["metric"])
    except Exception as exc:  # Neither world score is available.
        errors.extend(f"{name}: {type(exc).__name__}: {exc}" for name in ("reward", "pdms"))
        record["error"] = "; ".join(errors)
        return record
    try:
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.minimum_jerk_pdm_score import minimum_jerk_scores
        scorer, reward, _ = reward_settings()
        [result] = minimum_jerk_scores(cache, [Trajectory(poses)], load_gt(spec["gt"]), scorer, reward)
        record["reward"] = {
            "score": float(result.score), "nc": float(result.no_at_fault_collisions),
            "dac": float(result.drivable_area_compliance), "ttc": float(result.time_to_collision_within_bound),
            "ep": float(result.ego_progress), "comfort": float(result.comfort), "beta": float(result.beta),
            "beta_backward": float(result.beta_backward), "contact": result.contact_speed is not None,
            "contact_speed_mps": None if result.contact_speed is None else float(result.contact_speed),
            "infeasible_fit": bool(result.infeasible_fit)}
    except Exception as exc:
        errors.append(f"reward: {type(exc).__name__}: {exc}")
    try:
        record["pdms"] = pdms_result(cache, poses)
    except Exception as exc:
        errors.append(f"pdms: {type(exc).__name__}: {exc}")
    record["error"] = "; ".join(errors) or None
    return record


def failed_monitoring_record(spec, error):
    return {**{key: spec[key] for key in ("sample_id", "token", "kind", "variant")},
            "l1": None, "reward": None, "pdms": None, "error": error}


def _cohort_summary(rows):
    def mean(values):
        return float(np.mean(values)) if values else None
    l1 = [r["l1"] for r in rows if r.get("l1") is not None]
    rewards = [r["reward"] for r in rows if r.get("reward") is not None]
    fitted = [x for x in rewards if not x["infeasible_fit"]]
    speeds = [x["contact_speed_mps"] for x in fitted if x["contact"] and x["contact_speed_mps"] is not None]
    pdms = [r["pdms"] for r in rows if r.get("pdms") is not None]
    return {"expected": len(rows), "failed": sum(r.get("error") is not None for r in rows),
            "l1": mean(l1), "l1_complete": len(l1),
            "reward": mean([x["score"] for x in rewards]), "reward_complete": len(rewards),
            "infeasible_fit_fraction": mean([float(x["infeasible_fit"]) for x in rewards]), "fitted": len(fitted),
            **{name: mean([x[name] for x in fitted]) for name in REWARD_COMPONENTS},
            "contact_fraction": mean([float(x["contact"]) for x in fitted]), "contact_speed_mps": mean(speeds),
            "pdms": mean([x["score"] for x in pdms]), "pdms_complete": len(pdms),
            **{f"pdms_{name}": mean([x[name] for x in pdms]) for name, _ in PDMS_FIELDS}}


def aggregate_monitoring(records, specs):
    """Per-cohort (and per edited variant) means over unique held-out inputs."""
    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(specs):
        raise ValueError("Hazard SFT monitoring coverage contains missing, duplicate or unexpected inputs")
    for record in records:
        if any(record.get(key) != specs[record["sample_id"]][key] for key in ("token", "kind", "variant")):
            raise ValueError("Hazard SFT monitoring membership mismatch")
    cohorts = {kind: _cohort_summary([r for r in records if r["kind"] == kind]) for kind in COHORTS}
    variants = {kind: {variant: _cohort_summary([r for r in records if r["kind"] == kind and r["variant"] == variant])
                       for variant in sorted({r["variant"] for r in records if r["kind"] == kind})}
                for kind in ("benign", "hazardous")}
    return {"protocol": MONITORING_PROTOCOL, "cohorts": cohorts, "by_variant": variants}
