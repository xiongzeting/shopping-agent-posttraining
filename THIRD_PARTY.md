# Third-Party Components

## ShopSimulator

- Upstream: <https://github.com/ShopAgent-Team/ShopSimulator>
- Recorded base commit: `51bb26012cee31aea7ac26177c5ffe807026ac07`
- Recorded integration source commit: `9ecba272963960ab4a10e1a781bd05cd7634ce20`
- Local provenance record: `environments/ShopSimulator/EMBEDDED_SOURCE.json`

The included integration snapshot contains the runtime subset required by this project, project-specific environment changes, tests, templates, and a compressed product archive. This repository does not grant additional rights to upstream ShopSimulator content. Before public redistribution, review the upstream repository's current license and data-use terms.

## veRL

- Upstream: <https://github.com/volcengine/verl>
- Training version used by this project: `0.8.0`

The repository does not vendor veRL itself. Project-specific compatibility and rollout patches are stored in `patches/` and applied by the setup scripts.

## Qwen and other model services

Model weights are not included. Users must obtain Qwen checkpoints and any external teacher/judge model access independently under the corresponding providers' licenses and terms.

