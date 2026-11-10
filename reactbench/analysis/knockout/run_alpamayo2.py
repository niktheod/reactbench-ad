"""Alpamayo 2 attention knockout on original ReactPAI scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .common import (
    SampleContext,
    gpu_info,
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
    select_run_tokens,
    validate_args,
)


MODEL = "alpamayo2"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "models" / "alpamayo2" / "config.yaml"


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser(
        description="Alpamayo 2 attention knockout on original ReactPAI scenes.",
        default_config=DEFAULT_CONFIG,
    )
    args = parser.parse_args(argv)
    validate_args(parser, args)
    from ...models.alpamayo2.eval import TARGET
    from ...models.alpamayo2.inference import Alpamayo2Inference, paired_seed
    from ...models.alpamayo2.paiav_loader import PAIAVRun
    from ...models.runner import read_config

    run_dir = resolve_directory(args.run_dir, "run directory")
    config_path = resolve_file(args.config, "Alpamayo 2 configuration")
    cfg = read_config(config_path)
    if cfg.get("_target_") != TARGET:
        raise ValueError(f"Alpamayo 2 configuration must have _target_: {TARGET}")
    agent = Alpamayo2Inference(**{key: value for key, value in cfg.items() if not key.startswith("_")})
    with PAIAVRun(run_dir) as loader:
        tokens = select_run_tokens(list(loader.tokens), args)
        agent.initialize()
        model = agent.model
        if getattr(model, "_diffusion_expert_cuda_graph", None) is not None:
            raise RuntimeError("Diffusion-expert CUDA graphs would bypass the knockout")
        expected_steps = (
            agent.diffusion_kwargs.get("inference_step") or model.expert.diffusion.num_inference_steps
        )
        knockout = build_collector(
            vlm=model.vlm,
            expert=model.expert.expert,
            tokenizer=model.tokenizer,
            n_action_tokens=model.expert.action_space.get_action_space_dims()[0],
            expected_steps=expected_steps,
        )
        manager = KnockoutOutputManager(model=MODEL, root=args.output_root)
        gpus = gpu_info()
        hardware_class = agent.hardware()["class"]

        def run_one(token: str, context: SampleContext) -> dict[str, Any]:
            context.stage = "load_input"
            text_seed = paired_seed(token, agent.seed, "text")
            natural_seed = paired_seed(token, agent.diffusion_seed_base, "diffusion")
            reseed_seed = paired_seed(
                token, agent.diffusion_seed_base + RESEED_BASE_OFFSET, "diffusion"
            )
            context.details.update(
                text_seed=text_seed, diffusion_seed=natural_seed, reseed_diffusion_seed=reseed_seed
            )
            # The saved original run records its seeds; ours must be the same.
            saved = saved_reasoning(MODEL, token)
            if saved is not None and (
                int(saved["text_seed"]) != text_seed or int(saved["diffusion_seed"]) != natural_seed
            ):
                raise ValueError("The seeds differ from the saved original run's")
            with loader.materialize_inputs(token, original=True) as paths:
                model_input = loader.model_input(token, paths)
                result = run_conditions(
                    lambda seed: agent.compute_trajectory(
                        model_input, text_seed=text_seed, diffusion_seed=seed
                    ),
                    natural_seed=natural_seed,
                    reseed_seed=reseed_seed,
                    knockout=knockout,
                    context=context,
                )
            result["meta"].update(
                token=token,
                model=MODEL,
                dataset="pai-av",
                inputs="original",
                run_dir=run_dir.name,
                config=config_path.name,
                text_seed=int(text_seed),
                diffusion_seed=int(natural_seed),
                reseed_diffusion_seed=int(reseed_seed),
                reseed_base_offset=RESEED_BASE_OFFSET,
                navigation_command=model_input.get("navigation_command"),
                saved_cot_match=(
                    None if saved is None else single_text(saved, "cot") == result["meta"]["cot"]
                ),
                hardware_class=hardware_class,
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
