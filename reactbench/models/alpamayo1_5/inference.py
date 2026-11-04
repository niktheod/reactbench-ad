"""Backend identity and paired seeds of the Alpamayo 1.5 ReactPAI runner."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..seeds import paired_seed


@dataclass(frozen=True)
class AlpamayoInference:
    seed_base: int
    reasoning_seed: int

    seed_key = "diffusion_seed"
    namespace = "alpamayo-diffusion"
    evaluator = "alpamayo-paiav"

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]):
        result = cls(int(cfg.get("diffusion_seed_base", 42)), int(cfg.get("seed", 42)))
        result.seed_for_token("seed-validation")
        return result

    def seed_for_token(self, token: str) -> int:
        return paired_seed(self.namespace, self.seed_base, token)

    def seed_kwargs(self, seed: int) -> dict[str, int]:
        return {self.seed_key: seed}

    def options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        return dict(options)

    def metadata(self, config_path: Path) -> dict[str, Any]:
        return {
            "vlm_seed": self.reasoning_seed,
            f"{self.seed_key}_base": self.seed_base,
            f"{self.seed_key}_strategy": (
                f"SHA-256 of {self.namespace}:<base>:<token>; first 63 bits "
                "shared by all natural and crossed conditions"
            ),
        }
