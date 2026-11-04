"""Markdown renderings of the summary tables (values as in the paper; '*' = 95% CI includes zero)."""
from __future__ import annotations

from ..runs import OBJECTS, OBJECT_LABELS
from .runs import DATASETS, LABELS
from .stats import excludes_zero

TIR_EPSILONS = (0.05, 0.10, 0.20, 0.50)


def _f(value, digits=2, signed=True):
    if value is None:
        return "–"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _ci(interval, digits=2):
    if interval is None or None in interval:
        return ""
    return f" [{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def _gray(interval):
    return "" if interval is not None and excludes_zero(interval) else "*"


def _object(obj):
    return OBJECT_LABELS.get(obj, obj)


def main_table(records, dataset):
    """Main table: dP per model, object and tau."""
    if not records:
        return ""
    objects = [o for o in OBJECTS if any(r["object"] == o for r in records)]
    taus = sorted({r["tau_s"] for r in records})
    lines = [f"### {DATASETS[dataset]['label']}: ΔP (pp)", "",
             "| Model | " + " | ".join(f"{_object(o)} τ={t:g}" for o in objects for t in taus) + " |",
             "|---|" + "---:|" * (len(objects) * len(taus))]
    for model in dict.fromkeys(r["model"] for r in records):
        cells = []
        for obj in objects:
            for tau in taus:
                match = [r for r in records if (r["model"], r["object"], r["tau_s"]) == (model, obj, tau)]
                cells.append(_f(match[0]["delta_p_pp"]) + _gray(match[0]["delta_p_ci95_pp"]) if match else "–")
        lines.append(f"| {LABELS.get(model, model)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def detailed_table(records, dataset):
    lines = [f"### {DATASETS[dataset]['label']}: detailed results", "",
             "| Model | Object | τ (s) | P O → E (Δ) | NCins O → E (Δ) | CS O → E (Δ km/h) | n |",
             "|---|---|---:|---|---|---|---:|"]
    for r in records:
        lines.append(
            f"| {LABELS.get(r['model'], r['model'])} | {_object(r['object'])} | {r['tau_s']:g} | "
            f"{_f(r['p_original_pct'], signed=False)} → {_f(r['p_edited_pct'], signed=False)} "
            f"({_f(r['delta_p_pp'])}{_gray(r['delta_p_ci95_pp'])}) | "
            f"{_f(r['nc_ins_original_pct'], signed=False)} → {_f(r['nc_ins_edited_pct'], signed=False)} "
            f"({_f(r['delta_nc_ins_pp'])}) | "
            f"{_f(r['cs_original_kmh'], signed=False)} → {_f(r['cs_edited_kmh'], signed=False)} "
            f"({_f(r['delta_cs_kmh'])}{_gray(r['delta_cs_ci95_kmh'])}) | {r['paired_n']} |")
    return "\n".join(lines)


def tir_table(records):
    if not records:
        return ""
    lines = ["### Trajectory invariance rate (% of pairs with ADE < ε, pooled over τ)", "",
             "| Dataset | Model | Object | " + " | ".join(f"ε={e:.2f} m" for e in TIR_EPSILONS) + " | n |",
             "|---|---|---|" + "---:|" * (len(TIR_EPSILONS) + 1)]
    for r in records:
        values = " | ".join(f"{r[f'tir_{e:.2f}_pct']:.1f}" for e in TIR_EPSILONS)
        lines.append(f"| {DATASETS[r['dataset']]['label']} | {LABELS.get(r['model'], r['model'])} | "
                     f"{_object(r['object'])} | {values} | {r['n']} |")
    return "\n".join(lines)


def training_stages_table(records):
    """dP below 20 km/h at tau = 2.0 s; '*' = CI of the change from the previous stage includes zero."""
    if not records:
        return ""
    n = sorted({r["n_low"] for r in records})
    lines = [f"### Training stages: ΔP (pp) on ReactNAV at τ = 2.0 s, v < 20 km/h (n = {', '.join(map(str, n))})", "",
             "| Model | Stage | PDMS | " + " | ".join(_object(o) for o in OBJECTS) + " |",
             "|---|---|---:|" + "---:|" * len(OBJECTS)]
    for key in dict.fromkeys((r["model"], r["stage"]) for r in records):
        cells = {r["object"]: r for r in records if (r["model"], r["stage"]) == key}
        pdms = next(iter(cells.values()))["pdms"]
        values = []
        for obj in OBJECTS:
            r = cells.get(obj)
            if r is None:
                values.append("–")
                continue
            gray = "*" if r["change_ci95_pp"] is not None and not excludes_zero(r["change_ci95_pp"]) else ""
            values.append(_f(r["delta_p_low_pp"]) + gray)
        lines.append(f"| {key[0]} | {key[1]} | {pdms:.1f} | " + " | ".join(values) + " |")
    return "\n".join(lines)


NAVTEST_COLUMNS = (("no_at_fault_collisions", "NC"), ("drivable_area_compliance", "DAC"),
                   ("time_to_collision_within_bound", "TTC"), ("ego_progress", "EP"), ("comfort", "C"),
                   ("score", "PDMS"))


def fine_tuning_table(records):
    if not records:
        return ""
    lines = ["### Fine-tuning on ReactTrain: ReactNAV (mean over the 12 runs) and navtest", "",
             "| Checkpoint | ΔP (pp) | ΔCS (km/h) | " + " | ".join(label for _, label in NAVTEST_COLUMNS) + " |",
             "|---|---:|---:|" + "---:|" * len(NAVTEST_COLUMNS)]
    for r in records:
        navtest = r.get("navtest") or {}
        lines.append(f"| {LABELS.get(r['checkpoint'], r['checkpoint'])} | {_f(r['delta_p_pp'], 1)} | "
                     f"{_f(r['delta_cs_kmh'], 1)} | "
                     + " | ".join(_f(navtest.get(key), 1, signed=False) for key, _ in NAVTEST_COLUMNS) + " |")
    return "\n".join(lines)


def generalization_table(unseen, real):
    if not unseen and not real:
        return ""
    real = {r["checkpoint"]: r for r in real}
    lines = ["### Generalization (mean over the four τ)", "",
             "| Checkpoint | Unseen objects ΔP (pp) | ΔCS (km/h) | Real scenes P (%) | CS (km/h) |",
             "|---|---:|---:|---:|---:|"]
    names = list(dict.fromkeys([r["checkpoint"] for r in unseen] + list(real)))
    unseen = {r["checkpoint"]: r for r in unseen}
    for name in names:
        u, s = unseen.get(name, {}), real.get(name, {})
        lines.append(f"| {LABELS.get(name, name)} | {_f(u.get('delta_p_pp'), 1)} | {_f(u.get('delta_cs_kmh'), 1)} | "
                     f"{_f(s.get('p_pct'), 1, signed=False)} | {_f(s.get('cs_kmh'), 1, signed=False)} |")
    if unseen:
        lines.append("")
        lines.append(f"Unseen objects: {next(iter(unseen.values()))['n_tokens']} scenes common to every animal run.")
    return "\n".join(lines)


def speed_distribution_table(counts):
    """counts: {dataset: {speed bin: samples}}."""
    counts = {k: v for k, v in counts.items() if v}
    if not counts:
        return ""
    bins = list(dict.fromkeys(b for v in counts.values() for b in v))
    lines = ["### Samples per ego-speed bin (km/h) at the current frame", "",
             "| Dataset | " + " | ".join(bins) + " |", "|---|" + "---:|" * len(bins)]
    for dataset, value in counts.items():
        lines.append(f"| {DATASETS[dataset]['label']} | " + " | ".join(str(value.get(b, 0)) for b in bins) + " |")
    return "\n".join(lines)


def feasibility_table(records):
    if not records:
        return ""
    lines = ["### τ = 0.5 s: ΔP on all samples and on the avoidable ones (braking ≤ 8 m/s²)", "",
             "| Dataset | Model | Object | Unavoidable (%) | ΔP all (pp) | ΔP avoidable (pp) |",
             "|---|---|---|---:|---|---|"]
    for r in records:
        lines.append(f"| {DATASETS[r['dataset']]['label']} | {LABELS.get(r['model'], r['model'])} | "
                     f"{_object(r['object'])} | {r['unavoidable_pct']:.1f} | "
                     f"{_f(r['all_delta_p_pp'])}{_gray(r['all_ci95_pp'])} (n = {r['all_n']}) | "
                     f"{_f(r['avoidable_delta_p_pp'])}{_gray(r['avoidable_ci95_pp'])} (n = {r['avoidable_n']}) |")
    return "\n".join(lines)


def cot_swap_table(records):
    if not records:
        return ""
    codes = list(dict.fromkeys(r["condition"] for r in records))
    lines = ["### CoT swap on ReactPAI at τ = 2.0 s: NC (%) per (image, CoT); O = original, E = edited", "",
             "| Model | Object | " + " | ".join(f"I{c[0]} C{c[1]}" for c in codes) + " | n |",
             "|---|---|" + "---|" * len(codes) + "---:|"]
    for key in dict.fromkeys((r["model"], r["object"]) for r in records):
        cells = {r["condition"]: r for r in records if (r["model"], r["object"]) == key}
        lines.append(f"| {LABELS.get(key[0], key[0])} | {_object(key[1])} | "
                     + " | ".join(f"{_f(cells[c]['nc_pct'], 1, signed=False)}{_ci(cells[c]['ci95_pct'], 1)}"
                                  for c in codes) + f" | {max(r['n'] for r in cells.values())} |")
    return "\n".join(lines)


def fixed_cot_table(records):
    if not records:
        return ""
    conditions = list(dict.fromkeys(r["condition"] for r in records))
    lines = ["### Fixed CoTs on ReactPAI at τ = 2.0 s: ΔNC (pp) vs. the generated CoT", "",
             "| Model | Object | " + " | ".join(c.replace("_", " ") for c in conditions) + " |",
             "|---|---|" + "---:|" * len(conditions)]
    for key in dict.fromkeys((r["model"], r["object"]) for r in records):
        cells = {r["condition"]: r for r in records if (r["model"], r["object"]) == key}
        lines.append(f"| {LABELS.get(key[0], key[0])} | {_object(key[1])} | "
                     + " | ".join(_f(cells[c]["delta_nc_pp"]) + _gray(cells[c]["ci95_pp"]) for c in conditions) + " |")
    return "\n".join(lines)


def exploration_probe_table(records):
    if not records:
        return ""
    lines = ["### GRPO exploration probe (ReCogDrive IL): % of groups", "",
             "| Object | Signal G=8 | 16 | 32 | Safe signal G=8 | 16 | 32 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        values = [r[f"{kind}_g{k}_pct"] for kind in ("signal", "safe_signal") for k in (8, 16, 32)]
        label = "All" if r["object"] == "all" else _object(r["object"])
        lines.append(f"| {label} | " + " | ".join(_f(v, 1, signed=False) for v in values) + " |")
    return "\n".join(lines)


KNOCKOUT_LABELS = {"ade_knockout": "ADE KO", "ade_reseed": "ADE RS", "fde_knockout": "FDE KO", "fde_reseed": "FDE RS"}


def knockout_table(summary):
    """Attention knockout (meters; means with 95% CIs) from ``analysis.knockout.analyze``'s summary."""
    if not summary:
        return ""
    models = list(summary["models"])

    def cell(value, signed=False):
        return f"{_f(value['mean'], signed=signed)}{_ci(value['ci'])}"

    lines = [f"### Attention knockout on {summary['samples']} original ReactPAI samples (first 4 s, m)", "",
             "| | " + " | ".join(summary["models"][m]["label"] for m in models) + " | Difference |",
             "|---|---|---|---|"]
    for metric, label in KNOCKOUT_LABELS.items():
        lines.append(f"| {label} | " + " | ".join(cell(summary["models"][m]["metrics"][metric]) for m in models)
                     + f" | {cell(summary['difference'][metric], signed=True)} |")
    return "\n".join(lines)
