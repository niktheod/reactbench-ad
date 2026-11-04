# ------------------------------------------------------------------------
# Modified from LightningDiT(https://github.com/hustvl/LightningDiT)
# Copyright (c) Xiaomi Corporation. All rights reserved.
# ------------------------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np

from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)

from .blocks.rmsnorm import RMSNorm
from .blocks.attention import Attention
from .blocks.rope import RotaryEmbedding
from .blocks.encoder import SwiGLUFFN
from .blocks.kv_cross_attention import KVCacheCrossAttention

class TimestepEncoder(nn.Module):
    """Encodes scalar timesteps into a high-dimensional vector."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = self.timestep_embedder.linear_1.weight.dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb



class FinalLayer(nn.Module):
    """
    The final output layer of the DiT model.

    Encapsulates the final adaptive layer normalization and the projection to
    the output dimension.
    """
    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_dim)
        self.modulation_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size )
        )

    @torch.compile
    def modulate(self, x, shift, scale):
        if shift is None:
            return x * (1 + scale.unsqueeze(1))
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation_proj(conditioning).chunk(2, dim=1)
        x = self.modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        norm_type: str = "layer_norm", 
        norm_eps: float = 1e-5,
        kv_attention_config: Optional[dict] = None,
    ):
        super().__init__()

        if norm_type == "layer_norm":
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(dim)

        self.attn = KVCacheCrossAttention(query_dim=dim, **kv_attention_config) if kv_attention_config else Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
        )

        if norm_type == "layer_norm":
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm2 = RMSNorm(dim)

        self.ffn = SwiGLUFFN(
            dim,
            bias=True,
        )

        self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim, bias=True)
            )

    @torch.compile
    def modulate(self, x, shift, scale):
        if shift is None:
            return x * (1 + scale.unsqueeze(1))
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        rotary_embedder: Optional[nn.Module] = None,
        context_valid_mask=None,
        kv_cache=None,
        query_position_ids=None,
        query_rope=None,
    ) -> torch.Tensor:
        mod_params = self.adaLN_modulation(conditioning)
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = \
            mod_params.chunk(6, dim=1)

        normed_states = self.norm1(hidden_states)
        modulated_states = self.modulate(normed_states, shift_attn, scale_attn)

        if isinstance(self.attn, KVCacheCrossAttention):
            attn_output = self.attn(modulated_states, kv_cache, context_valid_mask,
                                    query_position_ids=query_position_ids, query_rope=query_rope)
        else:
            mask = context_valid_mask[:, None, None, :] if context_valid_mask is not None else None
            attn_output = self.attn(modulated_states, encoder_hidden_states=encoder_hidden_states,
                                    attention_mask=mask, rotary_embedder=rotary_embedder)
        hidden_states = hidden_states + gate_attn.unsqueeze(1) * attn_output

        normed_states = self.norm2(hidden_states)
        modulated_states = self.modulate(normed_states, shift_ffn, scale_ffn)
        ffn_output = self.ffn(modulated_states)

        hidden_states = hidden_states + gate_ffn.unsqueeze(1) * ffn_output
        
        return hidden_states

class LightningDiT(nn.Module):

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 48,
        output_dim: int = 512,
        num_layers: int = 16,
        dropout: float = 0.0,
        attention_bias: bool = True,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        interleave_attention: bool = True,
        conditioning_mode: str = "hidden_states",
        cross_attention_heads: int = 8,
        cross_attention_kv_heads: int = 2,
        cross_attention_head_dim: int = 128,
        cross_attention_q_norm: bool = False,
        vlm_rope_theta: float = 1_000_000.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.output_dim = output_dim
        self.interleave_attention = interleave_attention
        if conditioning_mode not in ("hidden_states", "vlm_kv"):
            raise ValueError("Unknown conditioning_mode")
        self.conditioning_mode = conditioning_mode
        self.num_cross_blocks = sum(i % 2 != 0 or not interleave_attention for i in range(num_layers))
        kv_config = dict(heads=cross_attention_heads, kv_heads=cross_attention_kv_heads,
                         dim_head=cross_attention_head_dim, q_norm=cross_attention_q_norm,
                         rope_theta=vlm_rope_theta, dropout=dropout, bias=attention_bias)

        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim
        )

        self.rotary_embedder = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=8, 
        )

        self.transformer_blocks = nn.ModuleList([
            LightningDiTBlock(
                dim=self.inner_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                attention_bias=attention_bias,
                norm_type=norm_type,
                norm_eps=norm_eps,
                cross_attention_dim=self.inner_dim if (idx % 2 != 0 or not interleave_attention) else None,
                kv_attention_config=kv_config if conditioning_mode == "vlm_kv" and
                    (idx % 2 != 0 or not interleave_attention) else None,
            ) for idx in range(num_layers)
        ])
                                           
        self.final_layer = FinalLayer(self.inner_dim, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initializes weights for stable training.
        - Initializes positional embeddings with sine/cosine values.
        - Zeroes out the weights of modulation and final output layers.
        """
        def zero_out_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        for block in self.transformer_blocks:
            block.adaLN_modulation.apply(zero_out_init)
        self.final_layer.modulation_proj.apply(zero_out_init)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        conditioning_features: torch.Tensor,
        timesteps: torch.LongTensor,
        return_hidden_states: bool = False,
        vlm_conditioning=None,
    ) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the UnifiedDiT model.

        Args:
            hidden_states (torch.Tensor): Input sequence. Shape: (B, N, D).
            encoder_hidden_states (torch.Tensor): Context sequence for cross-attention.
            conditioning_features (torch.Tensor): Additional conditioning features.
            timesteps (torch.LongTensor): Diffusion timesteps.
            return_hidden_states (bool): If True, returns the output and a list of
                all intermediate hidden states.

        Returns:
            The final output tensor, or a tuple of the output and all hidden states.
        """
        seq_len = hidden_states.shape[1]
        #hidden_states = hidden_states + self.pos_embed[:, :seq_len, :]

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        conditioning_features = conditioning_features.contiguous()
        if self.conditioning_mode == "vlm_kv":
            if vlm_conditioning is None or len(vlm_conditioning.kv_cache) != self.num_cross_blocks:
                raise ValueError("Supply one VLM cache pair for each cross-attention block")
        if vlm_conditioning is not None:
            mask = vlm_conditioning.valid_mask
            if mask.dtype != torch.bool or mask.shape != encoder_hidden_states.shape[:2]:
                raise ValueError("Conditioning mask must match the final hidden-state sequence")

        time_embedding = self.timestep_encoder(timesteps)
        conditioning = time_embedding + conditioning_features

        all_hidden_states = [hidden_states]
        cross_index = 0
        for idx, block in enumerate(self.transformer_blocks):
            use_cross_attention = not (idx % 2 == 0 and self.interleave_attention)
            current_encoder_states = encoder_hidden_states if use_cross_attention else None
            
            extra = {}
            if use_cross_attention and vlm_conditioning is not None:
                extra["context_valid_mask"] = vlm_conditioning.valid_mask
                if self.conditioning_mode == "vlm_kv":
                    extra.update(kv_cache=vlm_conditioning.kv_cache[cross_index],
                                 query_position_ids=vlm_conditioning.query_position_ids,
                                 query_rope=vlm_conditioning.query_rope)
            hidden_states = block(
                hidden_states,
                conditioning=conditioning,
                encoder_hidden_states=current_encoder_states,
                rotary_embedder=self.rotary_embedder,
                **extra,
            )
            cross_index += int(use_cross_attention)
            all_hidden_states.append(hidden_states)

        output = self.final_layer(hidden_states, conditioning)
        
        return (output, all_hidden_states) if return_hidden_states else output


