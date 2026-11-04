"""Alpamayo 1.5 attention knockout on original ReactPAI scenes."""

from __future__ import annotations

import dataclasses
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
    build_collectors,
    build_parser,
    run_conditions,
    saved_reasoning,
    select_run_tokens,
    validate_args,
)


MODEL = "alpamayo1_5"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "models" / "alpamayo1_5" / "config_paiav.yaml"


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser(
        description="Alpamayo 1.5 attention knockout on original ReactPAI scenes.",
        default_config=DEFAULT_CONFIG,
    )
    args = parser.parse_args(argv)
    validate_args(parser, args)
    from hydra.utils import instantiate

    from ...datasets.paiav import PAIAVCompactRun
    from ...models.alpamayo1_5 import eval_paiav as evaluation
    from ...models.alpamayo1_5.inference import AlpamayoInference

    run_dir = resolve_directory(args.run_dir, "run directory")
    config_path = resolve_file(args.config, "Alpamayo PAI-AV configuration")
    cfg = load_yaml_mapping(config_path)
    inference = AlpamayoInference.from_config(cfg)
    reseed = dataclasses.replace(inference, seed_base=inference.seed_base + RESEED_BASE_OFFSET)

    with PAIAVCompactRun(run_dir) as loader:
        tokens = select_run_tokens(list(loader.tokens), args)
        agent = instantiate(cfg)
        agent.initialize()
        model = agent.model
        expected_steps = agent.diffusion_kwargs.get("inference_step") or model.diffusion.num_inference_steps
        knockout, passthrough = build_collectors(
            check_passthrough=args.check_passthrough,
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
            natural_seed = inference.seed_for_token(token)
            reseed_seed = reseed.seed_for_token(token)
            context.details.update(diffusion_seed=natural_seed, reseed_diffusion_seed=reseed_seed)
            ego_history, _ = loader.ego_history(token)
            with loader.materialize_original_images(token) as original_paths:
                model_input = evaluation.make_input(
                    ego_history,
                    original_paths,
                    zero_ego_state=False,
                    navigation_command=None,
                )
                result = run_conditions(
                    lambda seed: agent.compute_trajectory(
                        model_input, **inference.seed_kwargs(seed)
                    ),
                    natural_seed=natural_seed,
                    reseed_seed=reseed_seed,
                    knockout=knockout,
                    passthrough=passthrough,
                    context=context,
                )
            saved = saved_reasoning(MODEL, token)
            result["meta"].update(
                token=token,
                model=MODEL,
                dataset="pai-av",
                inputs="original",
                run_dir=run_dir.name,
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
            if passthrough is not None:
                passthrough.close()


if __name__ == "__main__":
    main()
