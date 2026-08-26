# Package layout

The package follows the repository pipeline:

```text
environment/       connect to ShopSimulator and enforce the action contract
training/sft/      build assistant-only SFT examples
training/grpo/     veRL AgentLoop, Reward v4 adapter and dynamic sampling
evaluation/        hard checks, Rubric curation, trajectory Judge and aggregation
cli.py             small installed-package commands
smoke.py           CPU-only public smoke path
```

User-facing commands remain in the repository-level `scripts/` directory.
Those launchers call these modules; they are not a second implementation.

The GRPO runtime code and base configuration are present. Fresh GRPO data,
checkpoints and results are intentionally absent until a new isolated split is built.
