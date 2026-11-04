"""Forced-reasoning Alpamayo 2 rollouts for CoT-swap and fixed-CoT experiments.

Alpamayo 2 writes its reasoning as plain assistant text closed by
``<|traj_future_start|>``; unlike Alpamayo 1.5 it emits no ``<|cot_start|>``/
``<|cot_end|>`` wrappers. A forced rollout prefills ``{cot}<|traj_future_start|>``
as the assistant turn and runs one throwaway generation step, which returns the
prompt KV cache read by the unchanged diffusion expert. Upstream ``StopAfterEOS``
also feeds ``<|traj_future_start|>`` through the VLM before stopping, so the
cache length and expert position offset match a natural rollout.

The natural prompt construction, paired seeding and rollout are reused by
temporarily overriding three instance attributes, so the natural runner
(``inference.py``/``runner.py``) needs no forced-CoT code; Alpamayo 1.5's format
is in ``models/alpamayo1_5/forced_cot.py``. Note that the original-cache signature
(``runner.signature_for``) hashes these runner sources and Alpamayo 1.5's
``agent_paiav.py``/``agent_navsim.py``, so editing any of them changes it.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from ..original_cache import file_sha256


TRAJ_FUTURE_START = "<|traj_future_start|>"
ASSISTANT_MARKER = "<|im_start|>assistant\n"
PREFIX_FORMAT = "{cot}" + TRAJ_FUTURE_START
ADAPTER_VERSION = 1
# Discrete trajectory tokens are spelled <i0>, <i1>, ... in the vocabulary.
_TRAJECTORY_TOKEN = re.compile(r"<i\d+>")


def forced_cot_prefix(cot: str) -> str:
    """Assistant prefix replaying ``cot`` as Alpamayo 2's closed reasoning."""
    if not isinstance(cot, str):
        raise TypeError(f"Forced CoT must be a string, got {type(cot).__name__}")
    if cot != cot.strip():
        raise ValueError("Forced CoT must not start or end with whitespace; upstream strips decoded reasoning")
    # Chat-control and trajectory tokens would change the prompt structure, not the reasoning.
    if "<|" in cot or "|>" in cot or _TRAJECTORY_TOKEN.search(cot):
        raise ValueError(f"Forced CoT contains chat-control or trajectory token text: {cot[:200]!r}")
    return f"{cot}{TRAJ_FUTURE_START}"


def unwrap_singleton(value: Any, field: str) -> str:
    """Upstream text outputs are nested (batch, set, sample) singletons."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one {field} output, got {len(value)}")
        value = value[0]
    if not isinstance(value, str):
        raise ValueError(f"Expected a string {field} output, got {type(value).__name__}")
    return value


def assistant_generation(raw_output: str) -> tuple[str, bool]:
    """Assistant text before ``<|traj_future_start|>`` and whether that token occurs."""
    index = raw_output.rfind(ASSISTANT_MARKER)
    if index < 0:
        raise ValueError("Decoded output has no assistant turn")
    text, separator, _ = raw_output[index + len(ASSISTANT_MARKER):].partition(TRAJ_FUTURE_START)
    return text, bool(separator)


def natural_cot(document: Mapping[str, Any]) -> str:
    """Donor reasoning saved by a natural rollout.

    The decoded assistant text must be exactly the saved CoT followed by
    ``<|traj_future_start|>``. Generations that never closed their reasoning, or
    that carry a separate meta-action, cannot be replayed faithfully.
    """
    cot = unwrap_singleton(document.get("cot"), "cot")
    if "meta_action" in document and unwrap_singleton(document["meta_action"], "meta_action"):
        raise ValueError("Natural output carries a separate meta-action; the forced prefix would omit it")
    text, closed = assistant_generation(unwrap_singleton(document.get("raw_outputs"), "raw_outputs"))
    if not closed:
        raise ValueError("Natural generation never emitted <|traj_future_start|>; its text is not a closed CoT")
    if text != cot:
        raise ValueError("Saved CoT differs from the decoded assistant text")
    forced_cot_prefix(cot)
    return cot


def check_forced_output(extra: Mapping[str, Any], cot: str) -> None:
    """The decoded prompt must end in the forced prefix, and the CoT must round-trip."""
    text, closed = assistant_generation(unwrap_singleton(extra.get("raw_outputs"), "raw_outputs"))
    if not closed or text != cot:
        raise ValueError("Forced rollout prompt did not end with the CoT and <|traj_future_start|>")
    returned = unwrap_singleton(extra.get("cot"), "cot")
    if returned != cot:
        raise ValueError(f"Forced rollout did not preserve the CoT: expected {cot!r}, got {returned!r}")


def _append_prefix(messages, prefix):
    messages = list(messages)
    if messages and messages[-1].get("role") == "assistant":
        if messages[-1].get("content"):
            raise ValueError("The natural prompt already contains assistant text")
        messages.pop()
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("The natural prompt must end with the user turn")
    return [*messages, {"role": "assistant", "content": [{"type": "text", "text": prefix}]}]


class _PrefixedHelper:
    """Upstream helper module whose generation messages end in a forced prefix."""

    def __init__(self, helper, prefix):
        self._helper, self._prefix = helper, prefix

    def create_messages(self, *args, **kwargs):
        return _append_prefix(self._helper.create_messages(*args, **kwargs), self._prefix)

    def __getattr__(self, name):
        return getattr(self._helper, name)


@contextmanager
def forced_prompt(inference, prefix: str):
    """Run the natural input and rollout path with an assistant prefix.

    ``_make_model_input`` then renders with ``continue_final_message=True``,
    with or without navigation text. Every attribute is restored on exit.
    """
    if inference.model is None or inference.helper is None or inference.conversation_builder is None:
        raise RuntimeError("initialize() must be called before inference")
    saved = {name: getattr(inference, name)
             for name in ("helper", "conversation_builder", "max_generation_length")}
    builder = saved["conversation_builder"]
    inference.helper = _PrefixedHelper(saved["helper"], prefix)
    inference.conversation_builder = lambda *args, **kwargs: _append_prefix(builder(*args, **kwargs), prefix)
    # The prompt already ends in <|traj_future_start|>. One throwaway step makes
    # generate() return the complete prompt KV cache for the diffusion expert;
    # the sampled token follows the boundary and upstream replaces it by padding.
    inference.max_generation_length = 1
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(inference, name, value)


def compute_forced_trajectory(inference, agent_input, cot: str, *, text_seed: int, diffusion_seed: int):
    """Replay ``cot`` on these inputs; the diffusion seed matches the natural rollout's."""
    prefix = forced_cot_prefix(cot)
    with forced_prompt(inference, prefix):
        prediction = inference.compute_trajectory(
            agent_input, text_seed=text_seed, diffusion_seed=diffusion_seed)
    check_forced_output(prediction.extra, cot)
    return prediction


def adapter_identity() -> dict[str, Any]:
    return {"version": ADAPTER_VERSION, "prefix_format": PREFIX_FORMAT,
            "generation_steps": 1, "source_sha256": file_sha256(Path(__file__))}
