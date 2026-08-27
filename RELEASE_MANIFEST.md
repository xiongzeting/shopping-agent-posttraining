# GitHub Release Manifest

This directory is a curated copy of the working project rather than a full workspace mirror.

## Included

- Core Python package, training/evaluation scripts, configs, tests and documentation.
- ShopSimulator integration runtime without local virtual environments, indexes or logs.
- Final-240 task contract and small GRPO development task definitions.
- Aggregate Final-240 dashboard and per-task derived comparison files.
- Environment-variable example file and restrictive `.gitignore`.

## Excluded

- `.git` history from the private working directory.
- `.env`, API keys, credentials and provider call caches.
- Model weights, checkpoints, LoRA adapters and tokenizer caches.
- `outputs/`, `artifacts/`, temporary Codex audit directories and runtime logs.
- Full Teacher/SFT trajectory data, GRPO rollout candidates and training Parquet files.
- Raw Final-240 trajectories, raw Judge JSONL, rubric call records and historical report copies.
- Virtual environments, generated SQLite indexes, decompressed product JSON and Python caches.
- Resume documents, slide decks, interview notes and other personal materials.

The copy should remain small enough for ordinary Git hosting without Git LFS. Run a secret and large-file scan again immediately before publishing.
