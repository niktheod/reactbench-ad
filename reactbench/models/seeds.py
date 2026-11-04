"""Per-token seeds shared by the original and the edited scene of a ReactBench pair.

Every model draws its random numbers (diffusion noise, text sampling) from a seed derived from the scene token,
so the two scenes of a pair share them. Each model has its own namespace string, e.g. ``recogdrive-diffusion``.
"""
from __future__ import annotations

import hashlib

MAX_SEED = (1 << 63) - 1


def paired_seed(namespace: str, base_seed: int, token: str) -> int:
    """The first 63 bits of SHA-256(``"{namespace}:{base_seed}:{token}"``)."""
    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")
    base_seed = int(base_seed)
    if not 0 <= base_seed <= MAX_SEED:
        raise ValueError(f"base_seed must be between 0 and 2**63 - 1, got {base_seed}")
    material = f"{namespace}:{base_seed}:{token}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") >> 1
