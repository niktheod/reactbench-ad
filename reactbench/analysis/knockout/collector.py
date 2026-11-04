"""Expert attention statistics collected beside the unchanged SDPA calls.

In Alpamayo 1.5 and Alpamayo 2 the trajectory expert is a Qwen3-VL text model:
expert layer l attends non-causally over its action tokens and the VLM layer-l
KV cache. The collector wraps the registered ``sdpa`` attention function. Every
call returns the original function's output unchanged. Only for the expert's
attention modules, the attention weights are recomputed in float64 and reduced
to per-span sums. Nothing is written into the model, its inputs or its cache.

With ``hidden_spans`` (a knockout), expert attention instead receives a copy of
its mask that also hides those spans' keys; all other calls are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


SPANS = ("sink", "image", "ego_history", "prompt_text", "cot", "action")
SINK, IMAGE, EGO_HISTORY, PROMPT_TEXT, COT, ACTION = range(len(SPANS))

IMAGE_TOKENS = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
HISTORY_START_TOKEN = "<|traj_history_start|>"
HISTORY_END_TOKEN = "<|traj_history_end|>"
BOUNDARY_TOKEN = "<|traj_future_start|>"

# Tolerance for span distributions that must sum to one (stored as float32).
SUM_TOLERANCE = 1e-4


@dataclass(frozen=True)
class SpanTokenIds:
    image: tuple[int, ...]
    history_start: int
    history_end: int
    boundary: int


def resolve_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"The tokenizer has no id for {token!r}")
    if tokenizer.convert_ids_to_tokens(token_id) != token:
        raise ValueError(f"Token id {token_id} does not map back to {token!r}")
    return token_id


def resolve_span_token_ids(tokenizer: Any) -> SpanTokenIds:
    ids = SpanTokenIds(
        image=tuple(resolve_token_id(tokenizer, token) for token in IMAGE_TOKENS),
        history_start=resolve_token_id(tokenizer, HISTORY_START_TOKEN),
        history_end=resolve_token_id(tokenizer, HISTORY_END_TOKEN),
        boundary=resolve_token_id(tokenizer, BOUNDARY_TOKEN),
    )
    values = [*ids.image, ids.history_start, ids.history_end, ids.boundary]
    if len(set(values)) != len(values):
        raise ValueError("Span marker tokens must have distinct ids")
    return ids


def assign_spans(
    sequence: Any,
    *,
    prompt_len: int,
    cache_len: int,
    n_action: int,
    ids: SpanTokenIds,
) -> np.ndarray:
    """Span id of every key the expert sees: the cached sequence, then its actions.

    ``sequence`` is the generated batch-one sequence: prompt, generated tokens
    ending in the boundary token, then one sampled token that never entered
    the cache (the stopping criterion stops one step after the boundary).
    """
    tokens = np.asarray(sequence)
    if tokens.ndim != 1 or tokens.dtype.kind not in "iu":
        raise ValueError("The generated sequence must be a 1D integer array")
    tokens = tokens.astype(np.int64)
    prompt_len, cache_len, n_action = int(prompt_len), int(cache_len), int(n_action)
    if n_action <= 0:
        raise ValueError("The expert needs at least one action token")
    if not 0 < prompt_len < cache_len:
        raise ValueError(
            f"Expected 0 < prompt length ({prompt_len}) < cache length ({cache_len})"
        )
    if tokens.shape[0] != cache_len + 1:
        raise ValueError(
            f"The sequence has {tokens.shape[0]} tokens; expected the cache length "
            f"{cache_len} plus one sampled token after the boundary"
        )
    boundaries = np.flatnonzero(tokens[:cache_len] == ids.boundary)
    if boundaries.tolist() != [cache_len - 1]:
        raise ValueError(
            "The boundary token must occur exactly once in the cache, as its last "
            f"token; found it at positions {boundaries.tolist()}"
        )

    prompt = tokens[:prompt_len]
    spans = np.full(cache_len + n_action, PROMPT_TEXT, dtype=np.int64)
    image = np.isin(prompt, np.asarray(ids.image, dtype=np.int64))
    if not image.any():
        raise ValueError("The prompt contains no image tokens")
    if image[0]:
        raise ValueError("The first prompt token is an image token")
    spans[:prompt_len][image] = IMAGE

    starts = np.flatnonzero(prompt == ids.history_start)
    ends = np.flatnonzero(prompt == ids.history_end)
    if len(starts) != 1 or len(ends) != 1 or not 0 < starts[0] < ends[0]:
        raise ValueError(
            "The prompt needs one ego-history start token followed by one end "
            f"token; found starts {starts.tolist()} and ends {ends.tolist()}"
        )
    if image[starts[0] : ends[0] + 1].any():
        raise ValueError("Image tokens occur inside the ego-history span")
    spans[starts[0] : ends[0] + 1] = EGO_HISTORY
    spans[0] = SINK
    spans[prompt_len:cache_len] = COT
    spans[cache_len:] = ACTION
    return spans


def head_value_gram(
    o_proj_weight: torch.Tensor, *, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Per-head W_O^T W_O in float64, so that ||W_O^h v||^2 = v^T G_h v."""
    weight = o_proj_weight.detach()
    if weight.device.type == "meta":
        raise ValueError("The expert o_proj weight is not materialized")
    hidden, width = weight.shape
    if width != num_heads * head_dim:
        raise ValueError(
            f"o_proj takes {width} inputs; expected {num_heads} heads x {head_dim}"
        )
    # The attention output is reshaped from [heads, head_dim], so input column
    # h * head_dim + i of o_proj belongs to head h.
    per_head = weight.to(torch.float64).view(hidden, num_heads, head_dim).permute(1, 0, 2)
    return torch.matmul(per_head.transpose(1, 2), per_head)


def visible_keys(
    attention_mask: torch.Tensor, *, query_len: int, key_len: int
) -> torch.Tensor:
    """Keys every expert query may attend to, from a 0/min (or boolean) 4D mask."""
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 4:
        raise ValueError("Expert attention requires a 4D attention mask")
    mask = attention_mask[:, :, :, :key_len]  # SDPA slices the mask the same way.
    if mask.shape[0] != 1 or mask.shape[-2] != query_len or mask.shape[-1] != key_len:
        raise ValueError(
            f"Mask shape {tuple(attention_mask.shape)} does not match batch one, "
            f"{query_len} queries and {key_len} keys"
        )
    if mask.dtype == torch.bool:
        visible = mask
    elif mask.is_floating_point():
        if bool(torch.isnan(mask).any()):
            raise ValueError("The expert attention mask contains NaN")
        visible = mask > torch.finfo(mask.dtype).min / 2
        if bool((mask.masked_select(visible) != 0).any()):
            raise ValueError(
                "The expert attention mask adds biases; only 0/min visibility "
                "masks are supported"
            )
    else:
        raise TypeError(f"Unsupported attention mask dtype {mask.dtype}")
    row = visible[0, 0, 0]
    if not bool((visible == row).all()):
        raise ValueError("Key visibility differs across expert queries or heads")
    if not bool(row.any()):
        raise ValueError("The expert attention mask hides every key")
    return row


def expert_attention_statistics(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    gram: torch.Tensor,
    num_key_value_groups: int,
    scaling: float,
    span_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean over queries of span mass and span contribution share, per head.

    Returns ``raw`` and ``share`` with shape [heads, spans] (float64) and the
    [keys] visibility vector. For query q and key j of head h, the weight is
    alpha = softmax(q k^T * scaling) over visible keys and the contribution is
    alpha * ||W_O^h v_j||. Head h reads KV head h // num_key_value_groups, as in
    ``repeat_kv``.
    """
    # Autocast would run these matmuls in reduced precision; inputs are read only.
    with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
        if query.ndim != 4 or key.ndim != 4 or tuple(value.shape) != tuple(key.shape):
            raise ValueError(
                "Expected query [1, heads, queries, dim] and key/value "
                "[1, kv_heads, keys, dim]"
            )
        batch, heads, query_len, head_dim = query.shape
        if batch != 1 or key.shape[0] != 1:
            raise ValueError("Expert attention statistics require batch size one")
        if key.shape[1] * int(num_key_value_groups) != heads or key.shape[-1] != head_dim:
            raise ValueError(
                f"Key shape {tuple(key.shape)} does not match {heads} query heads "
                f"with {num_key_value_groups} heads per KV head"
            )
        key_len = key.shape[-2]
        if tuple(span_ids.shape) != (key_len,):
            raise ValueError(f"Expected {key_len} span ids, got {tuple(span_ids.shape)}")
        if tuple(gram.shape) != (heads, head_dim, head_dim):
            raise ValueError(f"Gram shape {tuple(gram.shape)} does not match the heads")

        device = query.device
        visible = visible_keys(
            attention_mask.to(device), query_len=query_len, key_len=key_len
        )
        head_to_kv = torch.arange(heads, device=device) // int(num_key_value_groups)
        q = query[0].to(torch.float64)
        k = key[0].to(device=device, dtype=torch.float64).index_select(0, head_to_kv)
        v = value[0].to(device=device, dtype=torch.float64).index_select(0, head_to_kv)

        scores = torch.matmul(q, k.transpose(-1, -2)) * float(scaling)
        weights = torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)
        onehot = torch.nn.functional.one_hot(
            span_ids.to(device=device, dtype=torch.int64), num_classes=len(SPANS)
        ).to(torch.float64)
        raw = torch.matmul(weights, onehot).mean(dim=1)

        squared = (torch.matmul(v, gram.to(device)) * v).sum(dim=-1)
        norms = squared.clamp_min(0.0).sqrt()
        span_contributions = torch.matmul(weights * norms[:, None, :], onehot)
        totals = span_contributions.sum(dim=-1, keepdim=True)
        if not bool((totals > 0).all()):
            raise ValueError("Every attention contribution of a query vanished")
        share = (span_contributions / totals).mean(dim=1)
    return raw, share, visible


def check_span_distributions(name: str, values: Any) -> None:
    """Values [..., spans] must be finite, non-negative and sum to one."""
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != len(SPANS):
        raise ValueError(f"{name} must end in {len(SPANS)} spans, got {array.shape}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} is empty or contains non-finite values")
    if float(array.min()) < -SUM_TOLERANCE:
        raise ValueError(f"{name} contains negative values")
    deviation = float(np.abs(array.astype(np.float64).sum(axis=-1) - 1.0).max())
    if deviation > SUM_TOLERANCE:
        raise ValueError(f"{name} does not sum to one over spans (off by {deviation:.2e})")


def _normalized_text(text: str) -> str:
    return "".join(text.split())


class ExpertAttentionCollector:
    """Collect per-span expert attention during one natural rollout at a time.

    Call ``reset()`` before each sample and ``finalize()`` after it. The
    collector is installed on construction; ``close()`` restores the SDPA
    registry entry, ``vlm.generate`` and the expert's hooks, and ``install()``
    puts them back.

    ``hidden_spans`` (span ids, default none) knocks those cached spans out of
    every expert attention call; their keys get zero attention.
    """

    def __init__(
        self,
        *,
        vlm: Any,
        expert: torch.nn.Module,
        tokenizer: Any,
        n_action_tokens: int,
        expected_steps: int,
        hidden_spans: Sequence[int] = (),
        attention_functions: Any = None,
    ) -> None:
        if attention_functions is None:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            attention_functions = ALL_ATTENTION_FUNCTIONS
        if not isinstance(getattr(attention_functions, "_local_mapping", None), dict):
            raise TypeError("Unsupported transformers attention registry")
        self.vlm = vlm
        self.expert = expert
        self.tokenizer = tokenizer
        self.ids = resolve_span_token_ids(tokenizer)
        self.n_action_tokens = int(n_action_tokens)
        self.expected_steps = int(expected_steps)
        if self.n_action_tokens <= 0 or self.expected_steps <= 0:
            raise ValueError("Action tokens and denoising steps must be positive")
        hidden = list(hidden_spans)
        if any(
            isinstance(span, bool) or not isinstance(span, (int, np.integer))
            or not 0 <= span < ACTION
            for span in hidden
        ) or len(set(hidden)) != len(hidden):
            raise ValueError(f"Hidden spans must be distinct cached span ids, got {hidden}")
        self.hidden_spans = tuple(sorted(int(span) for span in hidden))

        config = expert.config
        implementation = getattr(config, "_attn_implementation", None)
        if implementation != "sdpa":
            raise ValueError(
                f"The expert attention runs {implementation!r}; the collector "
                "requires SDPA"
            )
        layers = list(expert.layers)
        if not layers or len(layers) != int(config.num_hidden_layers):
            raise ValueError("The expert layer count does not match its config")
        self.num_layers = len(layers)
        self.num_heads = int(config.num_attention_heads)
        self._attention_modules: list[torch.nn.Module] = []
        self._layer_by_module: dict[int, int] = {}
        for index, layer in enumerate(layers):
            attention = layer.self_attn
            if int(attention.layer_idx) != index:
                raise ValueError(f"Expert layer {index} reports layer_idx {attention.layer_idx}")
            if getattr(attention.config, "_attn_implementation", None) != "sdpa":
                raise ValueError(f"Expert layer {index} does not run SDPA")
            groups = int(attention.num_key_value_groups)
            if groups * int(config.num_key_value_heads) != self.num_heads:
                raise ValueError("Expert query heads do not match its KV heads")
            self._attention_modules.append(attention)
            self._layer_by_module[id(attention)] = index

        self._functions = attention_functions
        self._gram: dict[int, torch.Tensor] = {}
        self._installed = False
        self._dispatcher: Callable[..., Any] | None = None
        self._had_local_sdpa = False
        self._previous_local_sdpa: Any = None
        self._generate: Callable[..., Any] | None = None
        self._had_generate = False
        self._previous_generate: Any = None
        self._hook_handle: Any = None
        self.reset()
        self.install()

    # Installation -----------------------------------------------------------

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("The attention collector is already installed")
        original_sdpa = self._functions["sdpa"]
        original_generate = self.vlm.generate
        collector = self

        def sdpa_with_statistics(module, query, key, value, attention_mask, *args, **kwargs):
            layer = collector._layer_by_module.get(id(module))
            is_expert = layer is not None and collector._attention_modules[layer] is module
            if is_expert and collector.hidden_spans:
                attention_mask = collector._hide_keys(attention_mask, key_len=key.shape[-2])
            output = original_sdpa(module, query, key, value, attention_mask, *args, **kwargs)
            if is_expert:
                collector._record(layer, module, query, key, value, attention_mask, args, kwargs)
            return output

        def generate(*args, **kwargs):
            return collector._capture_generate(original_generate, args, kwargs)

        local = self._functions._local_mapping
        self._had_local_sdpa = "sdpa" in local
        self._previous_local_sdpa = local.get("sdpa")
        attributes = vars(self.vlm)
        self._had_generate = "generate" in attributes
        self._previous_generate = attributes.get("generate")
        self._installed = True
        try:
            # Record each replacement only after it succeeded, so close() undoes
            # exactly what was applied.
            self._hook_handle = self.expert.register_forward_pre_hook(
                self._before_expert, with_kwargs=True
            )
            self.vlm.generate = generate
            self._generate = generate
            self._functions["sdpa"] = sdpa_with_statistics
            self._dispatcher = sdpa_with_statistics
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Restore everything ``install()`` replaced; safe to call repeatedly."""
        foreign: list[str] = []
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        if self._generate is not None:
            if vars(self.vlm).get("generate") is self._generate:
                if self._had_generate:
                    self.vlm.generate = self._previous_generate
                else:
                    del self.vlm.generate
            else:
                foreign.append("vlm.generate")
            self._generate = None
        if self._dispatcher is not None:
            if self._functions._local_mapping.get("sdpa") is self._dispatcher:
                if self._had_local_sdpa:
                    self._functions["sdpa"] = self._previous_local_sdpa
                else:
                    del self._functions["sdpa"]
            else:
                foreign.append("the sdpa attention function")
            self._dispatcher = None
        self._installed = False
        if foreign:
            raise RuntimeError(
                f"Something else replaced {', '.join(foreign)} while the collector "
                "was installed; it was left in place"
            )

    def __enter__(self) -> "ExpertAttentionCollector":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # Per-sample state -------------------------------------------------------

    def reset(self) -> None:
        self._generate_calls = 0
        self._sequences: np.ndarray | None = None
        self._prompt_len: int | None = None
        self._cache_len: int | None = None
        self._spans: np.ndarray | None = None
        self._span_ids_by_device: dict[torch.device, torch.Tensor] = {}
        self._hidden_by_device: dict[torch.device, torch.Tensor] = {}
        self._step = -1
        self._records: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _capture_generate(self, original: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
        if self._generate_calls:
            raise RuntimeError("VLM generation ran twice for one sample; call reset() per sample")
        self._generate_calls += 1
        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else (args[0] if args else None)
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("The collector requires batch-one input_ids for VLM generation")
        prompt = input_ids.detach().to(device="cpu", dtype=torch.int64).clone()
        outputs = original(*args, **kwargs)
        sequences = getattr(outputs, "sequences", None)
        if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2 or sequences.shape[0] != 1:
            raise ValueError("VLM generation did not return batch-one sequences")
        # Upstream later overwrites tokens after the boundary in place; keep a copy.
        sequences = sequences.detach().to(device="cpu", dtype=torch.int64).clone()
        prompt_len = prompt.shape[1]
        if sequences.shape[1] <= prompt_len or not torch.equal(sequences[0, :prompt_len], prompt[0]):
            raise ValueError("The generated sequence does not extend the prompt")
        self._prompt_len = int(prompt_len)
        self._sequences = sequences[0].numpy()
        return outputs

    def _before_expert(self, _module: torch.nn.Module, _args: tuple, kwargs: dict) -> None:
        if self._sequences is None or self._prompt_len is None:
            raise RuntimeError("The expert ran before VLM generation was captured")
        cache = kwargs.get("past_key_values")
        embeds = kwargs.get("inputs_embeds")
        if cache is None or not isinstance(embeds, torch.Tensor):
            raise ValueError("Expert calls must pass inputs_embeds and past_key_values by keyword")
        if embeds.ndim != 3 or tuple(embeds.shape[:2]) != (1, self.n_action_tokens):
            raise ValueError(
                f"Expected expert inputs [1, {self.n_action_tokens}, hidden], "
                f"got {tuple(embeds.shape)}"
            )
        cache_len = int(cache.get_seq_length())
        if self._cache_len is None:
            self._spans = assign_spans(
                self._sequences,
                prompt_len=self._prompt_len,
                cache_len=cache_len,
                n_action=self.n_action_tokens,
                ids=self.ids,
            )
            self._cache_len = cache_len
        elif cache_len != self._cache_len:
            raise ValueError(
                f"The prompt cache changed from {self._cache_len} to {cache_len} "
                "tokens between denoising steps"
            )
        if self._step >= 0:
            self._require_complete_step(self._step)
        if self._step + 1 >= self.expected_steps:
            raise RuntimeError(
                f"The expert ran more than {self.expected_steps} times for one sample"
            )
        self._step += 1

    def _require_complete_step(self, step: int) -> None:
        missing = [layer for layer in range(self.num_layers) if (step, layer) not in self._records]
        if missing:
            raise RuntimeError(f"Denoising step {step} has no attention for expert layers {missing}")

    def _span_ids_on(self, device: torch.device) -> torch.Tensor:
        if self._spans is None:
            raise RuntimeError("Spans are assigned at the first expert call")
        if device not in self._span_ids_by_device:
            self._span_ids_by_device[device] = torch.as_tensor(
                self._spans, dtype=torch.int64, device=device
            )
        return self._span_ids_by_device[device]

    def _hidden_keys_on(self, device: torch.device) -> torch.Tensor:
        """Boolean [keys]: True for keys in a hidden span."""
        if device not in self._hidden_by_device:
            self._hidden_by_device[device] = torch.isin(
                self._span_ids_on(device),
                torch.as_tensor(self.hidden_spans, dtype=torch.int64, device=device),
            )
        return self._hidden_by_device[device]

    def _hide_keys(self, attention_mask: torch.Tensor | None, *, key_len: int) -> torch.Tensor:
        """A copy of the expert mask that also hides every key in the hidden spans."""
        if self._step < 0 or self._spans is None:
            raise RuntimeError("Expert attention ran outside a captured expert call")
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 4:
            raise ValueError("Hiding keys requires the expert's 4D attention mask")
        if key_len != len(self._spans) or attention_mask.shape[-1] < key_len:
            raise ValueError(
                f"Expected {len(self._spans)} expert keys and a mask covering them, got "
                f"{key_len} keys and mask {tuple(attention_mask.shape)}"
            )
        mask = attention_mask[:, :, :, :key_len]  # SDPA slices the mask the same way.
        hidden = self._hidden_keys_on(mask.device)
        if mask.dtype == torch.bool:
            return mask & ~hidden
        if mask.is_floating_point():
            # Upstream hides its own masked keys with the dtype minimum, too.
            return mask.masked_fill(hidden, torch.finfo(mask.dtype).min)
        raise TypeError(f"Unsupported attention mask dtype {mask.dtype}")

    def _gram_for(self, layer: int, module: torch.nn.Module) -> torch.Tensor:
        # The expert is frozen, so the per-head Gram matrices are computed once.
        if layer not in self._gram:
            self._gram[layer] = head_value_gram(
                module.o_proj.weight, num_heads=self.num_heads, head_dim=int(module.head_dim)
            )
        return self._gram[layer]

    def _record(
        self,
        layer: int,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        args: tuple,
        kwargs: Mapping[str, Any],
    ) -> None:
        if self._step < 0 or self._cache_len is None:
            raise RuntimeError("Expert attention ran outside a captured expert call")
        if args:
            raise ValueError("Unexpected positional arguments in an expert attention call")
        if float(kwargs.get("dropout", 0.0) or 0.0) != 0.0:
            raise ValueError("Expert attention dropout must be zero at inference")
        if kwargs.get("output_attentions") or kwargs.get("head_mask") is not None:
            raise ValueError("Expert attention must not request attention outputs or head masks")
        if attention_mask is None:
            raise ValueError("Expert attention ran without its attention mask")
        if kwargs.get("is_causal"):
            raise ValueError("Expert attention must be non-causal")
        record_key = (self._step, layer)
        if record_key in self._records:
            raise RuntimeError(f"Expert layer {layer} ran twice in denoising step {self._step}")
        expected_keys = self._cache_len + self.n_action_tokens
        if query.shape[-2] != self.n_action_tokens or key.shape[-2] != expected_keys:
            raise ValueError(
                f"Expert layer {layer} saw {query.shape[-2]} queries and {key.shape[-2]} "
                f"keys; expected {self.n_action_tokens} and {expected_keys}"
            )
        scaling = kwargs.get("scaling")
        if scaling is None:
            scaling = query.shape[-1] ** -0.5
        raw, share, visible = expert_attention_statistics(
            query,
            key,
            value,
            attention_mask,
            gram=self._gram_for(layer, module),
            num_key_value_groups=int(module.num_key_value_groups),
            scaling=float(scaling),
            span_ids=self._span_ids_on(query.device),
        )
        if self.hidden_spans:
            if not torch.equal(visible, ~self._hidden_keys_on(visible.device)):
                raise ValueError(
                    "The expert attention mask must hide exactly the hidden spans' keys"
                )
        elif not bool(visible.all()):
            raise ValueError(
                "The expert attention mask hides cached keys; batch-one rollouts "
                "should see the whole cache"
            )
        self._records[record_key] = (raw, share)

    def finalize(self, *, returned_cot: str) -> dict[str, Any]:
        """Arrays [layers, heads, steps, spans] and metadata for the sample."""
        if self._sequences is None or self._prompt_len is None:
            raise RuntimeError("No VLM generation was captured for this sample")
        if self._step < 0 or self._cache_len is None or self._spans is None:
            raise RuntimeError("The expert never ran for this sample")
        steps = self._step + 1
        if steps != self.expected_steps:
            raise RuntimeError(f"Expected {self.expected_steps} denoising steps, observed {steps}")
        for step in range(steps):
            self._require_complete_step(step)

        shape = (self.num_layers, self.num_heads, steps, len(SPANS))
        raw_mass = np.empty(shape, dtype=np.float64)
        contrib_share = np.empty(shape, dtype=np.float64)
        for (step, layer), (raw, share) in self._records.items():
            raw_mass[layer, :, step, :] = raw.detach().cpu().numpy()
            contrib_share[layer, :, step, :] = share.detach().cpu().numpy()
        check_span_distributions("raw_mass", raw_mass)
        check_span_distributions("contrib_share", contrib_share)

        generated = self._sequences[self._prompt_len : self._cache_len]
        generated_text = self.tokenizer.decode(generated.tolist(), skip_special_tokens=False)
        if not isinstance(returned_cot, str) or not returned_cot.strip():
            raise ValueError("The model returned an empty CoT")
        if _normalized_text(returned_cot) not in _normalized_text(generated_text):
            raise ValueError(
                f"The returned CoT {returned_cot!r} is not part of the generated "
                f"tokens {generated_text!r}"
            )
        span_counts = np.bincount(self._spans, minlength=len(SPANS)).astype(np.int64)
        hidden_keys = int(span_counts[list(self.hidden_spans)].sum())
        meta = {
            "spans": list(SPANS),
            "prompt_len": int(self._prompt_len),
            "cache_len": int(self._cache_len),
            "generated_tokens": int(self._cache_len - self._prompt_len),
            "action_tokens": int(self.n_action_tokens),
            "visible_keys": int(span_counts.sum()) - hidden_keys,
            "num_layers": int(self.num_layers),
            "num_heads": int(self.num_heads),
            "num_steps": int(steps),
            "generated_text": generated_text,
            "cot": returned_cot,
        }
        if self.hidden_spans:
            meta["hidden_spans"] = [SPANS[span] for span in self.hidden_spans]
            meta["hidden_keys"] = hidden_keys
        return {
            "raw_mass": raw_mass.astype(np.float32),
            "contrib_share": contrib_share.astype(np.float32),
            "span_counts": span_counts,
            "meta": meta,
        }
