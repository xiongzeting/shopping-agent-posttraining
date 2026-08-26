#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---preflight-only}"

BASE_MODEL="${BASE_MODEL:-$ROOT/Qwen3.5-2B}"
SFT_ADAPTER="${SFT_ADAPTER:-$ROOT/outputs/models/fresh-sft-convergence-repair-v5-30k-lora}"
GRPO_MODEL="${GRPO_MODEL:-$ROOT/outputs/models/grpo-30k-8way-step100-merged}"
GRPO_ADAPTER="${GRPO_ADAPTER:-$GRPO_MODEL/lora_adapter}"
BENCHMARK="$ROOT/data/evaluation/tasks.jsonl"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/evaluation/final240-v22-sft-v5-vs-grpo-step100-rewardfix-maxtokens768-20260812}"

SHOPSIM_PORT="${SHOPSIM_PORT:-5700}"
VLLM_PORT="${VLLM_PORT:-8000}"
WORKERS="${WORKERS:-20}"
MAX_STEPS="${MAX_STEPS:-45}"
MAX_TOKENS="${MAX_TOKENS:-768}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-30000}"
SEED="${SEED:-20260806}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -s "$1" ]] || fail "missing required file: $1"
}

require_dir() {
  [[ -d "$1" ]] || fail "missing required directory: $1"
}

check_adapter() {
  local adapter="$1"
  require_file "$adapter/adapter_config.json"
  if [[ ! -s "$adapter/adapter_model.safetensors" && ! -s "$adapter/adapter_model.bin" ]]; then
    fail "adapter weights missing: $adapter"
  fi
}

check_full_model() {
  local model="$1"
  require_file "$model/config.json"
  if [[ ! -s "$model/model.safetensors" && ! -s "$model/model.safetensors.index.json" ]]; then
    fail "full model weights missing: $model"
  fi
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

write_identity_manifest() {
  mkdir -p "$RUN_ROOT"
  "$ROOT/.venv/bin/python" - "$RUN_ROOT/model-identity.json" <<PY
import hashlib
import json
from pathlib import Path
import sys

def digest(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

payload = {
    "schema": "shopping-final240-two-model-rerun-v1",
    "benchmark": {
        "path": str(Path("$BENCHMARK").resolve()),
        "sha256": digest("$BENCHMARK"),
        "expected_tasks": 240,
    },
    "models": {
        "sft_v5_lora": {
            "served_model": "qwen35-sft-v5",
            "base_model": str(Path("$BASE_MODEL").resolve()),
            "adapter": str(Path("$SFT_ADAPTER").resolve()),
            "adapter_config_sha256": digest("$SFT_ADAPTER/adapter_config.json"),
            "lineage": "Qwen3.5-2B + fresh-sft-convergence-repair-v5-30k-lora",
        },
        "sft_v5_plus_grpo_step100": {
            "served_model": "qwen35-sft-v5-grpo-step100",
            "exported_model": str(Path("$GRPO_MODEL").resolve()),
            "adapter": str(Path("$GRPO_ADAPTER").resolve()),
            "adapter_config_sha256": digest("$GRPO_ADAPTER/adapter_config.json"),
            "lineage": "fresh-sft-convergence-repair-v5-30k-merged -> GRPO global_step_100 export + GRPO lora_adapter",
        },
    },
    "evaluation": {
        "reward": "shopsimulator-reward-v4-current-working-tree",
        "environment": "shopsimulator-environment-v2.4",
        "termination": "shopping-termination-v3.1",
        "observation": "shopping-observation-v2",
        "tool_schema": "shopping-tools-v2",
        "workers": int("$WORKERS"),
        "max_steps": int("$MAX_STEPS"),
        "max_tokens": int("$MAX_TOKENS"),
        "context_window": int("$CONTEXT_WINDOW"),
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": int("$SEED"),
    },
}
Path(sys.argv[1]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

preflight() {
  require_file "$BENCHMARK"
  require_file "$ROOT/data/evaluation/metadata.json"
  require_file "$ROOT/data/evaluation/slices.jsonl"
  require_file "$ROOT/data/environment.json"
  require_file "$ROOT/.venv/bin/python"
  require_file "$ROOT/.venv/bin/vllm"
  require_dir "$BASE_MODEL"
  check_full_model "$BASE_MODEL"
  check_adapter "$SFT_ADAPTER"
  check_full_model "$GRPO_MODEL"
  check_adapter "$GRPO_ADAPTER"

  local task_count
  task_count="$(wc -l < "$BENCHMARK" | tr -d ' ')"
  [[ "$task_count" == "240" ]] || fail "Final-240 task count is $task_count, expected 240"

  "$ROOT/.venv/bin/python" - "$SFT_ADAPTER/adapter_config.json" "$GRPO_ADAPTER/adapter_config.json" <<'PY'
import json
import sys
from pathlib import Path

for value in sys.argv[1:]:
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    rank = payload.get("r")
    if rank != 16:
        raise SystemExit(f"unexpected LoRA rank in {value}: {rank!r}, expected 16")
print("adapter_rank_check=passed")
PY

  write_identity_manifest
  echo "preflight=passed"
  echo "benchmark_sha256=$(sha256_file "$BENCHMARK")"
  echo "sft_model=Qwen3.5-2B + $SFT_ADAPTER"
  echo "grpo_model=$GRPO_MODEL + $GRPO_ADAPTER"
  echo "identity_manifest=$RUN_ROOT/model-identity.json"
}

env_pid=""
vllm_pid=""
cleanup_model() {
  if [[ -n "$vllm_pid" ]]; then
    kill "$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
    vllm_pid=""
  fi
}
cleanup_all() {
  cleanup_model
  if [[ -n "$env_pid" ]]; then
    kill "$env_pid" 2>/dev/null || true
    wait "$env_pid" 2>/dev/null || true
  fi
}
trap cleanup_all EXIT

wait_for_model() {
  local required="$1"
  local models_json="$2"
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$VLLM_PORT/v1/models" >"$models_json" 2>/dev/null && grep -Fq "\"id\":\"$required\"" "$models_json"; then
      return 0
    fi
    kill -0 "$vllm_pid" 2>/dev/null || fail "vLLM exited before serving $required"
    sleep 2
  done
  fail "timed out waiting for served model: $required"
}

model_identity_probe() {
  local model="$1"
  "$ROOT/.venv/bin/python" - "$model" "$VLLM_PORT" <<'PY'
import json
import sys
import urllib.request

model, port = sys.argv[1], sys.argv[2]
body = {
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "temperature": 0,
    "max_tokens": 8,
}
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    payload = json.load(response)
if payload.get("model") != model:
    raise SystemExit(f"wrong model returned: expected={model} actual={payload.get('model')}")
print("model_identity_probe=passed model=" + model)
PY
}

start_environment() {
  mkdir -p "$RUN_ROOT"
  env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
      SHOP_SEARCH_INDEX="$ROOT/environments/ShopSimulator/shop_env/search_engine/products.sqlite3" \
      SHOPSIM_ENV_SLOTS="$WORKERS" SHOP_MAX_STEPS="$MAX_STEPS" SHOPSIM_PORT="$SHOPSIM_PORT" \
      bash "$ROOT/scripts/start_environment.sh" >"$RUN_ROOT/shopsimulator.log" 2>&1 &
  env_pid=$!
  for _ in $(seq 1 120); do
    "$ROOT/.venv/bin/python" -c "import socket; s=socket.create_connection(('127.0.0.1',$SHOPSIM_PORT),2); s.close()" >/dev/null 2>&1 && return 0
    kill -0 "$env_pid" 2>/dev/null || fail "ShopSimulator exited before readiness"
    sleep 2
  done
  fail "ShopSimulator readiness timeout"
}

evaluate_model() {
  local model="$1"
  local label="$2"
  local tokenizer="$3"
  local run_dir="$RUN_ROOT/$label"
  mkdir -p "$run_dir"
  "$ROOT/.venv/bin/python" "$ROOT/scripts/evaluate_shop_benchmark.py" \
    --benchmark "$BENCHMARK" \
    --benchmark-metadata "$ROOT/data/evaluation/metadata.json" \
    --benchmark-slices "$ROOT/data/evaluation/slices.jsonl" \
    --environment-manifest "$ROOT/data/environment.json" \
    --run-dir "$run_dir" \
    --base-url "http://127.0.0.1:$SHOPSIM_PORT" \
    --llm-base-url "http://127.0.0.1:$VLLM_PORT/v1" \
    --api-key EMPTY \
    --model "$model" \
    --actor-label "$label" \
    --tokenizer-path "$tokenizer" \
    --max-steps "$MAX_STEPS" \
    --workers "$WORKERS" \
    --temperature 0 \
    --top-p 1 \
    --timeout 180 \
    --max-tokens "$MAX_TOKENS" \
    --context-window "$CONTEXT_WINDOW" \
    --context-safety-margin 512 \
    --context-compaction \
    --observation-token-budget 1536 \
    --observation-detail-token-budget 4096 \
    --observation-generic-token-budget 768 \
    --observation-search-top-k 20 \
    --seed "$SEED" \
    >"$run_dir/evaluation.log" 2>&1
}

run_sft() {
  local log="$RUN_ROOT/sft-v5-lora"
  mkdir -p "$log"
  env VLLM_USE_FLASHINFER_SAMPLER=0 "$ROOT/.venv/bin/vllm" serve "$BASE_MODEL" \
    --served-model-name qwen35-base \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --max-model-len "$CONTEXT_WINDOW" \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs "$WORKERS" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules "qwen35-sft-v5=$SFT_ADAPTER" \
    >"$log/vllm.log" 2>&1 &
  vllm_pid=$!
  wait_for_model qwen35-sft-v5 "$log/models.json"
  model_identity_probe qwen35-sft-v5
  evaluate_model qwen35-sft-v5 sft-v5-lora "$BASE_MODEL"
  cleanup_model
}

run_grpo() {
  local log="$RUN_ROOT/sft-v5-grpo-step100"
  mkdir -p "$log"
  env VLLM_USE_FLASHINFER_SAMPLER=0 "$ROOT/.venv/bin/vllm" serve "$GRPO_MODEL" \
    --served-model-name grpo-step100-export-base \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --max-model-len "$CONTEXT_WINDOW" \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs "$WORKERS" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules "qwen35-sft-v5-grpo-step100=$GRPO_ADAPTER" \
    >"$log/vllm.log" 2>&1 &
  vllm_pid=$!
  wait_for_model qwen35-sft-v5-grpo-step100 "$log/models.json"
  model_identity_probe qwen35-sft-v5-grpo-step100
  evaluate_model qwen35-sft-v5-grpo-step100 sft-v5-grpo-step100 "$GRPO_MODEL"
  cleanup_model
}

case "$MODE" in
  --preflight-only)
    preflight
    ;;
  --execute)
    preflight
    start_environment
    run_sft
    run_grpo
    ;;
  *)
    fail "usage: bash scripts/run_final240_sft_grpo_rerun.sh [--preflight-only|--execute]"
    ;;
esac
