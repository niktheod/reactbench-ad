"""Alpamayo 1.5's CoT prompt format, used to force a given CoT (CoT swap and fixed CoTs).

Alpamayo 1.5 wraps its reasoning in ``<|cot_start|>``/``<|cot_end|>`` and starts the trajectory with
``<|traj_future_start|>``. A forced rollout puts ``<|cot_start|>{cot}<|cot_end|><|traj_future_start|>`` into
the assistant turn (``agent.py``, ``forced_cot=``); Alpamayo 2's format is in
``models/alpamayo2/forced_cot.py``.
"""
from __future__ import annotations

RESERVED_COT_TOKENS = (
    "<|cot_start|>",
    "<|cot_end|>",
    "<|traj_future_start|>",
    "<|traj_future_end|>",
)


def forced_cot_prefix(cot: str) -> str:
    """Return the complete assistant prefix consumed by trajectory generation."""

    if not isinstance(cot, str):
        raise TypeError(f"forced_cot must be a string, got {type(cot).__name__}")
    contained = [token for token in RESERVED_COT_TOKENS if token in cot]
    if contained:
        raise ValueError(
            "forced_cot contains reserved Alpamayo token(s): "
            f"{contained}"
        )
    return f"<|cot_start|>{cot}<|cot_end|><|traj_future_start|>"
