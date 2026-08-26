# Canonical SFT runbook

The versioned run plan is `configs/sft_canonical.json`. It freezes the source
difficulty quotas, natural Teacher policy, post-hoc trajectory length policy,
runtime contract, Qwen3.5-2B fingerprint, LoRA recipe, resource gates, and the
records required for every attempt.

## Gates before training

Training may start only after all of these are true:

1. The canonical corpus contains exactly 1,000 strict accepted trajectories.
2. Every accepted trajectory ends in `gold_purchase` with `reward_valid=true`.
   This result gate is unchanged and remains non-negotiable.
3. The recoverable process gate rejects malformed or truncated calls, executed
   invalid steps, missing `search -> open_product -> buy_now` paths, and
   consecutive duplicate actions. Runtime Guard rejections are allowed only
   when the trajectory later reaches strict success; rejected assistant/tool
   messages are removed before SFT serialization.
4. The final 1,000-row raw Teacher corpus passes
   `shopping-teacher-data-gate-v1`. Its immutable report must satisfy the
   retrieval-difficulty quotas, minimum recovery/comparison/option/long-horizon
   coverage, and action-sequence/8-step caps in `configs/sft_canonical.json`.
   `data/sft/metadata.json` records the report path and SHA-256.
5. Integer step boundaries are frozen after collection so Short/Medium/Long
   approximate 40%/40%/20%; trajectories are never padded to meet a bucket.
6. No exact action sequence exceeds 12% of the final corpus and 8-step
   trajectories do not exceed 30%. Selection prioritizes search recovery,
   candidate comparison, evidence-page diversity, and multi-option coverage.
7. The 900/100 split has zero task overlap and zero Final-240 task, ASIN, family,
   or semantic overlap.
8. The Qwen3.5-2B tokenizer audit keeps all canonical rows at max length 30000.
9. The canonical data, model, package, disk, CUDA, BF16, Flash Attention 2,
   Liger, and single-GPU
   preflight passes on the training host.

The checked-in Teacher-1000 is the promoted `shopping-sft-dataset-v3` corpus
built under `shopping-teacher-recoverable-process-v4`. Its immutable data-gate
report is `data/sft/data_gate.json`; its tokenizer audit is
`data/sft/token_audit.json`. Do not edit metadata merely to make runtime
preflight green: every replacement still requires a newly generated report.

Audit the final raw corpus before promotion:

```text
python scripts/audit_teacher_data_gate.py \
  --input outputs/<teacher-run>/final_raw.jsonl \
  --report outputs/<teacher-run>/data_gate.json
```

The command exits with status 1 while any quota is missing. After a real pass,
copy the report beside the promoted metadata and record this immutable contract:

```json
"data_gate": {
  "schema_version": "shopping-teacher-data-gate-v1",
  "status": "passed",
  "path": "data_gate.json",
  "sha256": "<report sha256>"
}
```

## Preparation and execution

The existing launcher separates auditable preparation from training:

```text
bash scripts/sft.sh --preflight-only
bash scripts/sft.sh --preflight-only --skip-gpu-check
bash scripts/sft.sh
```

`--preflight-only` creates a complete attempt directory but never loads model
weights for training. The second command is allowed only after the user
explicitly requests SFT execution.

`--skip-gpu-check` is restricted to `--preflight-only`. It records all data,
model, tokenizer, dependency and storage checks available on a host without a
visible GPU. The actual training attempt must rerun preflight without this flag
and pass the Linux, CUDA 13, BF16, Flash Attention 2, Liger and GPU-memory gates.

Each attempt records the planned recipe and SHA-256, launcher command, safe
launcher environment, resolved paths, preflight contract, exact training
command, resolved and effective training parameters, tokenization statistics,
metrics, GPU samples, checkpoints, status transitions, console output, summary,
and any failure. Secrets are not written to these records.

Use one stable `SFT_RUN_ID` for all attempts that belong to the same run. Use a
new `SFT_ATTEMPT_ID` for every preflight, start, or resume. A resumed attempt
must match the first attempt's `run_contract.json` and use the highest complete
checkpoint selected by the launcher.
