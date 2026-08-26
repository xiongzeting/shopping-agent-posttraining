# GRPO step-30 probe snapshot

This snapshot is intended for a short 30-optimizer-step GRPO run.

- Training probe rows: 60 (30 steps × train batch size 2).
- Historical online admission pool: 251 tasks.
- The recovered admission files contain 70 tasks with 0/4 Gold-or-Valid
  trajectories; these are excluded. The earlier 56 figure was an incomplete
  summary and is not used for filtering.
- Candidate pool after that exclusion: 181 tasks.
- The current frozen selection is 60 frontier tasks, ranked by Probe
  Gold/Valid success count. All 60 selected frontier tasks had 3/4 Probe
  successes; no 0/4 task is included.
- Validation frequency: every 10 optimizer steps.

The 60-row snapshot must be built from the post-rollout admission files. Offline
candidate rows are not silently promoted to training data.
