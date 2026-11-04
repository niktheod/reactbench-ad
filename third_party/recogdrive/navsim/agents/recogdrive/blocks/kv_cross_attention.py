"""Cross-attention to a frozen decoder prefix; cached keys already contain RoPE."""
import torch
from torch import nn
from torch.nn import functional as F

from .rmsnorm import RMSNorm


def query_rotary_factors(position_ids, head_dim, theta=1_000_000.0, dtype=torch.float32):
    """Qwen2's unscaled, split-half RoPE, with independent positions per sample."""
    if head_dim % 2 or position_ids.ndim != 2:
        raise ValueError("RoPE requires an even head dimension and [B, T] positions")
    with torch.autocast(device_type=position_ids.device.type, enabled=False):
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=position_ids.device).float() / head_dim))
        freqs = position_ids.float()[..., None] * inv_freq
        angles = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
        return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_queries(q, factors):
    cos, sin = (factor.to(device=q.device, dtype=q.dtype) for factor in factors)
    first, second = q.chunk(2, dim=-1)
    return q * cos + torch.cat((-second, first), dim=-1) * sin


class KVCacheCrossAttention(nn.Module):
    def __init__(self, query_dim, heads=8, kv_heads=2, dim_head=128,
                 dropout=0.0, bias=True, q_norm=False, rope_theta=1_000_000.0):
        super().__init__()
        if heads <= 0 or kv_heads <= 0 or heads % kv_heads or dim_head <= 0 or dim_head % 2:
            raise ValueError("GQA needs query heads divisible by KV heads and a positive even head dimension")
        self.num_heads, self.kv_heads, self.head_dim = heads, kv_heads, dim_head
        self.rope_theta = rope_theta
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=bias)
        self.q_norm = RMSNorm(dim_head) if q_norm else nn.Identity()
        self.to_out = nn.Sequential(nn.Linear(heads * dim_head, query_dim), nn.Dropout(dropout))

    def forward(self, hidden_states, kv_cache, context_valid_mask,
                query_position_ids=None, query_rope=None):
        batch, tokens, _ = hidden_states.shape
        k, v = kv_cache
        if (k.ndim != 4 or k.shape != v.shape or k.shape[:2] != (batch, self.kv_heads)
                or k.shape[-1] != self.head_dim):
            raise ValueError("Cache must contain K/V shaped [B, KV heads, S, head_dim]")
        if context_valid_mask.dtype != torch.bool or context_valid_mask.shape != (batch, k.shape[2]):
            raise ValueError("context_valid_mask must be boolean [B, S], True for valid tokens")
        q = self.to_q(hidden_states).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        if query_rope is None:
            if query_position_ids is None:
                query_position_ids = context_valid_mask.sum(-1)[:, None] + torch.arange(tokens, device=q.device)
            query_rope = query_rotary_factors(query_position_ids, self.head_dim, self.rope_theta, q.dtype)
        q = rotate_queries(q, query_rope)
        # The native cache may be BF16 while the planner runs FP16 autocast.
        k, v = k.to(device=q.device, dtype=q.dtype), v.to(device=q.device, dtype=q.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=context_valid_mask[:, None, None, :].to(q.device),
            dropout_p=self.to_out[-1].p if self.training else 0.0,
            is_causal=False, enable_gqa=True,
        )
        return self.to_out(out.transpose(1, 2).reshape(batch, tokens, -1))
