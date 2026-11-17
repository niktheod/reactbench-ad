"""Alpamayo 1.5 attention knockout on original ReactPAI scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .common import (
    SampleContext,
    gpu_info,
    load_yaml_mapping,
    resolve_directory,
    resolve_file,
    run_loop,
    single_text,
)
from .knockout import (
    RESEED_BASE_OFFSET,
    KnockoutOutputManager,
    build_collector,
    build_parser,
    run_conditions,
    saved_reasoning,
    select_variant_tokens,
    validate_args,
)


MODEL = "alpamayo1_5"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "models" / "alpamayo1_5" / "config.yaml"


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser(
        description="Alpamayo 1.5 attention knockout on original ReactPAI scenes.",
        default_config=DEFAULT_CONFIG,
    )
    args = parser.parse_args(argv)
    validate_args(parser, args)
    from hydra.utils import instantiate

    from ...datasets.paiav import PAIAVCompactVariant
    from ...models.alpamayo1_5.eval import SEED_NAMESPACE
    from ...models.alpamayo1_5.inputs import model_input
    from ...models.seeds import paired_seed

    variant_dir = resolve_directory(args.variant_dir, "variant directory")
    config_path = resolve_file(args.config, "Alpamayo 1.5 configuration")
    cfg = load_yaml_mapping(config_path)
    seed_base = int(cfg.get("diffusion_seed_base", 42))

    with PAIAVCompactVariant(variant_dir) as loader:
        tokens = select_variant_tokens(list(loader.tokens), args)
        agent = instantiate(cfg)
        agent.initialize()
        model = agent.model
        expected_steps = agent.diffusion_kwargs.get("inference_step") or model.diffusion.num_inference_steps
        knockout = build_collector(
            vlm=model.vlm,
            expert=model.expert,
            tokenizer=model.tokenizer,
            n_action_tokens=model.action_space.get_action_space_dims()[0],
            expected_steps=expected_steps,
        )
        manager = KnockoutOutputManager(model=MODEL, root=args.output_root)
        gpus = gpu_info()

        def run_one(token: str, context: SampleContext) -> dict[str, Any]:
            context.stage = "load_input"
            natural_seed = paired_seed(SEED_NAMESPACE, seed_base, token)
            reseed_seed = paired_seed(SEED_NAMESPACE, seed_base + RESEED_BASE_OFFSET, token)
            context.details.update(diffusion_seed=natural_seed, reseed_diffusion_seed=reseed_seed)
            with model_input(loader, token, original=True) as agent_input:
                result = run_conditions(
                    lambda seed: agent.compute_trajectory(agent_input, diffusion_seed=seed),
                    natural_seed=natural_seed,
                    reseed_seed=reseed_seed,
                    knockout=knockout,
                    context=context,
                )
            saved = saved_reasoning(MODEL, token)
            result["meta"].update(
                token=token,
                model=MODEL,
                dataset="pai-av",
                inputs="original",
                run_dir=variant_dir.name,  # key kept for existing results
                config=config_path.name,
                vlm_seed=int(agent.seed),
                diffusion_seed=int(natural_seed),
                reseed_diffusion_seed=int(reseed_seed),
                reseed_base_offset=RESEED_BASE_OFFSET,
                navigation_command=None,
                saved_cot_match=(
                    None if saved is None else single_text(saved, "cot") == result["meta"]["cot"]
                ),
                gpus=gpus,
            )
            return result

        try:
            run_loop(
                tokens=tokens,
                manager=manager,
                overwrite=args.overwrite,
                run_one=run_one,
            )
        finally:
            knockout.close()


if __name__ == "__main__":
    main()
