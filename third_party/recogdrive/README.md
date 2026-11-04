# ReCogDrive (modified copy)

A pruned and modified copy of [ReCogDrive](https://github.com/xiaomi-research/recogdrive)
(Apache-2.0, see `LICENSE`), including its NAVSIM devkit. ReactBench uses it for

- ReCogDrive inference on ReactNAV and the generalization sets (`reactbench/models/recogdrive`),
- the NAVSIM devkit behind ReactNAV scoring, metric caching and navtest PDMS,
- the ReactTrain fine-tuning of the IL checkpoint (paper section 5) and the GRPO exploration probe.

It is not installed; the release scripts put `third_party/recogdrive` and the repository root on
`PYTHONPATH`. See `CHANGES.md` for the differences from upstream and the repository README for the
commands.
