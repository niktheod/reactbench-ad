"""Backend identity and paired seeds of the Alpamayo 1.5 ReactPAI runner."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from ..original_cache import file_sha256


VLM_AGENT_TARGET = "alpamayo.evaluation.vlm_agent.AlpamayoVLMAgent"  # text-only variant, not part of the release


def paired_seed(token: str, base_seed: int, namespace: str) -> int:
    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")
    if not 0 <= int(base_seed) <= (1 << 63) - 1:
        raise ValueError("base_seed must be between 0 and 2**63 - 1")
    material = f"{namespace}:{int(base_seed)}:{token}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") >> 1


@dataclass(frozen=True)
class AlpamayoInference:
    vlm_only: bool
    seed_base: int
    reasoning_seed: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], *, require_vlm: bool = False):
        vlm_only = cfg.get("_target_") == VLM_AGENT_TARGET
        if require_vlm and not vlm_only:
            raise ValueError(f"VLM-only evaluation requires _target_: {VLM_AGENT_TARGET}")
        seed_key = "generation_seed_base" if vlm_only else "diffusion_seed_base"
        result = cls(vlm_only, int(cfg.get(seed_key, 42)), int(cfg.get("seed", 42)))
        result.seed_for_token("seed-validation")
        return result

    @property
    def seed_key(self) -> str:
        return "generation_seed" if self.vlm_only else "diffusion_seed"

    @property
    def namespace(self) -> str:
        return "alpamayo-vlm-actions" if self.vlm_only else "alpamayo-diffusion"

    @property
    def evaluator(self) -> str:
        return "alpamayo-paiav-vlm" if self.vlm_only else "alpamayo-paiav"

    def seed_for_token(self, token: str) -> int:
        return paired_seed(token, self.seed_base, self.namespace)

    def seed_kwargs(self, seed: int) -> dict[str, int]:
        return {self.seed_key: seed}

    def options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        return {**options, **({"inference_backend": "vlm", "decoding_version": 1}
                              if self.vlm_only else {})}

    def metadata(self, config_path: Path) -> dict[str, Any]:
        result = {
            "vlm_seed": self.reasoning_seed,
            f"{self.seed_key}_base": self.seed_base,
            f"{self.seed_key}_strategy": (
                f"SHA-256 of {self.namespace}:<base>:<token>; first 63 bits "
                "shared by all natural, crossed, and replay conditions"
            ),
        }
        result.update(self.identity_metadata(config_path))
        return result

    def identity_metadata(self, config_path: Path) -> dict[str, Any]:
        return ({"inference_backend": "vlm", "decoding_version": 1,
                 "evaluation_config_sha256": file_sha256(config_path)}
                if self.vlm_only else {})
