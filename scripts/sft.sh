#!/usr/bin/env bash
set -Eeuo pipefail

ORIGINAL_ARGV=("$@")

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/.env" && "${SFT_SKIP_PROJECT_ENV:-0}" != "1" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
cd "${ROOT}"

resolve_path() {
  local value="$1"
  if [[ "${value}" == "~" ]]; then
    value="${HOME}"
  elif [[ "${value}" == "~/"* ]]; then
    value="${HOME}/${value:2}"
  fi
  realpath -m -- "${value}"
}

resolve_output_path() {
  local value="$1"
  if [[ "${value}" == "~" ]]; then
    value="${HOME}"
  elif [[ "${value}" == "~/"* ]]; then
    value="${HOME}/${value:2}"
  fi
  realpath -m -s -- "${value}"
}

BASE_MODEL="${BASE_MODEL:-${ROOT}/Qwen3.5-2B}"
TRAIN_DATA="${SFT_TRAIN_DATA:-${ROOT}/data/sft/train.jsonl}"
VALIDATION_DATA="${SFT_VALIDATION_DATA:-${ROOT}/data/sft/validation.jsonl}"
ALL_DATA="${SFT_ALL_DATA:-${ROOT}/data/sft/all.jsonl}"
METADATA="${SFT_METADATA:-${ROOT}/data/sft/metadata.json}"
SFT_PLAN="${SFT_PLAN:-${ROOT}/configs/sft_canonical.json}"
DEFAULT_ADAPTER_DIR="${ROOT}/outputs/models/sft-lora"
DEFAULT_MERGED_DIR="${ROOT}/outputs/models/sft-merged"
ADAPTER_DIR="${SFT_ADAPTER_DIR:-${DEFAULT_ADAPTER_DIR}}"
MERGED_DIR="${SFT_MERGED_DIR:-${DEFAULT_MERGED_DIR}}"
RUN_ID="${SFT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_DIR="${SFT_RUN_DIR:-${ROOT}/outputs/runs/sft/${RUN_ID}}"
ATTEMPT_ID="${SFT_ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
ATTEMPT_DIR="${RUN_DIR}/attempts/${ATTEMPT_ID}"
MIN_GPU_MEMORY_GIB="${SFT_MIN_GPU_MEMORY_GIB:-94}"
MIN_FREE_GPU_MEMORY_GIB="${SFT_MIN_FREE_GPU_MEMORY_GIB:-92}"
MIN_FREE_DISK_GIB="${SFT_MIN_FREE_DISK_GIB:-50}"
MAX_LENGTH="${SFT_MAX_LENGTH:-30000}"
EPOCHS="${SFT_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${SFT_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${SFT_LEARNING_RATE:-1e-4}"
WARMUP_RATIO="${SFT_WARMUP_RATIO:-0.03}"
GRADIENT_CHECKPOINTING="${SFT_GRADIENT_CHECKPOINTING:-0}"
RESUME_FROM_CHECKPOINT="${SFT_RESUME_FROM_CHECKPOINT:-}"
MERGE_AFTER_TRAINING="${SFT_MERGE:-0}"
PREFLIGHT_ONLY=0
SKIP_GPU_PREFLIGHT=0
MERGE_ONLY=0
EXTRA_TRAIN_ARGS=(--liger-kernel --flash-attention-2)

BASE_MODEL="$(resolve_path "${BASE_MODEL}")"
TRAIN_DATA="$(resolve_path "${TRAIN_DATA}")"
VALIDATION_DATA="$(resolve_path "${VALIDATION_DATA}")"
ALL_DATA="$(resolve_path "${ALL_DATA}")"
METADATA="$(resolve_path "${METADATA}")"
SFT_PLAN="$(resolve_path "${SFT_PLAN}")"
ADAPTER_DIR="$(resolve_path "${ADAPTER_DIR}")"
MERGED_DIR="$(resolve_output_path "${MERGED_DIR}")"
RUN_DIR="$(resolve_path "${RUN_DIR}")"
DEFAULT_ADAPTER_DIR="$(resolve_path "${DEFAULT_ADAPTER_DIR}")"
DEFAULT_MERGED_DIR="$(resolve_output_path "${DEFAULT_MERGED_DIR}")"
ATTEMPT_DIR="${RUN_DIR}/attempts/${ATTEMPT_ID}"

if [[ ! "${MAX_LENGTH}" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "${TRAIN_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "${EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SFT length/batch/accumulation parameters must be positive integers" >&2
  exit 2
fi
if [[ "${GRADIENT_CHECKPOINTING}" != "0" && "${GRADIENT_CHECKPOINTING}" != "1" ]]; then
  echo "SFT_GRADIENT_CHECKPOINTING must be 0 or 1" >&2
  exit 2
fi

while (($#)); do
  case "$1" in
    --preflight-only)
      PREFLIGHT_ONLY=1
      shift
      ;;
    --skip-gpu-check)
      SKIP_GPU_PREFLIGHT=1
      shift
      ;;
    --merge)
      MERGE_AFTER_TRAINING=1
      shift
      ;;
    --merge-only)
      MERGE_ONLY=1
      shift
      ;;
    --no-merge)
      MERGE_AFTER_TRAINING=0
      shift
      ;;
    --resume-from-checkpoint)
      if (($# < 2)); then
        echo "--resume-from-checkpoint 缺少路径" >&2
        exit 2
      fi
      RESUME_FROM_CHECKPOINT="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_TRAIN_ARGS=("$@")
      break
      ;;
    *)
      echo "未知参数：$1；训练器高级参数请放在 -- 后" >&2
      exit 2
      ;;
  esac
done

RAW_EXTRA_TRAIN_ARGS=("${EXTRA_TRAIN_ARGS[@]}")
EXTRA_TRAIN_ARGS=()
USE_LIGER=0
USE_QLORA=0
USE_FLASH_ATTN_2=0
ATTENTION_IMPLEMENTATION="sdpa"
DEVICE_MAP=""
SMOKE_MAX_STEPS=""
extra_index=0
while ((extra_index < ${#RAW_EXTRA_TRAIN_ARGS[@]})); do
  extra_arg="${RAW_EXTRA_TRAIN_ARGS[extra_index]}"
  case "${extra_arg}" in
    --liger-kernel)
      USE_LIGER=1
      EXTRA_TRAIN_ARGS+=("${extra_arg}")
      extra_index=$((extra_index + 1))
      ;;
    --qlora)
      USE_QLORA=1
      EXTRA_TRAIN_ARGS+=("${extra_arg}")
      extra_index=$((extra_index + 1))
      ;;
    --flash-attention-2)
      USE_FLASH_ATTN_2=1
      ATTENTION_IMPLEMENTATION="flash_attention_2"
      extra_index=$((extra_index + 1))
      ;;
    --device-map)
      if ((extra_index + 1 >= ${#RAW_EXTRA_TRAIN_ARGS[@]})); then
        echo "--device-map 缺少映射策略" >&2
        exit 2
      fi
      DEVICE_MAP="${RAW_EXTRA_TRAIN_ARGS[extra_index + 1]}"
      case "${DEVICE_MAP}" in
        auto|balanced|balanced_low_0|sequential) ;;
        *)
          echo "--device-map 只允许 auto、balanced、balanced_low_0 或 sequential" >&2
          exit 2
          ;;
      esac
      EXTRA_TRAIN_ARGS+=("--device-map" "${DEVICE_MAP}")
      extra_index=$((extra_index + 2))
      ;;
    --max-steps)
      if ((extra_index + 1 >= ${#RAW_EXTRA_TRAIN_ARGS[@]})); then
        echo "--max-steps 缺少正整数" >&2
        exit 2
      fi
      SMOKE_MAX_STEPS="${RAW_EXTRA_TRAIN_ARGS[extra_index + 1]}"
      if [[ ! "${SMOKE_MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "--max-steps 必须为正整数" >&2
        exit 2
      fi
      EXTRA_TRAIN_ARGS+=("--max-steps" "${SMOKE_MAX_STEPS}")
      extra_index=$((extra_index + 2))
      ;;
    *)
      echo "禁止通过 -- 覆盖未被预检的训练参数：${extra_arg}" >&2
      echo "允许项仅为 --liger-kernel、--qlora、--flash-attention-2、--device-map <策略> 和 --max-steps <正整数>。" >&2
      exit 2
      ;;
  esac
done

RECIPE_VARIANT="canonical"
if ((USE_LIGER)) && ((USE_QLORA)); then
  RECIPE_VARIANT="liger+qlora"
elif ((USE_LIGER)); then
  RECIPE_VARIANT="liger"
elif ((USE_QLORA)); then
  RECIPE_VARIANT="qlora"
fi
if ((USE_FLASH_ATTN_2)); then
  RECIPE_VARIANT="flash-attn2+${RECIPE_VARIANT}"
fi
if [[ "${RECIPE_VARIANT}" == "flash-attn2+liger" ]]; then
  RECIPE_VARIANT="canonical"
fi
if [[ -n "${DEVICE_MAP}" ]]; then
  RECIPE_VARIANT="device-map-${DEVICE_MAP}+${RECIPE_VARIANT}"
fi
if [[ -n "${SMOKE_MAX_STEPS}" ]]; then
  RECIPE_VARIANT="${RECIPE_VARIANT}+max-steps-${SMOKE_MAX_STEPS}"
fi
if [[ -n "${SFT_RECIPE_VARIANT:-}" ]]; then
  if [[ "${RECIPE_VARIANT}" == "canonical" ]]; then
    RECIPE_VARIANT="${SFT_RECIPE_VARIANT}"
  else
    RECIPE_VARIANT="${SFT_RECIPE_VARIANT}+${RECIPE_VARIANT}"
  fi
fi
if ((PREFLIGHT_ONLY)) && ((MERGE_ONLY)); then
  echo "--preflight-only 与 --merge-only 不能同时使用" >&2
  exit 2
fi
if ((SKIP_GPU_PREFLIGHT)) && ((PREFLIGHT_ONLY == 0)); then
  echo "--skip-gpu-check 仅允许与 --preflight-only 一起使用" >&2
  exit 2
fi
if ((PREFLIGHT_ONLY)) && [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "--preflight-only 不能与 --resume-from-checkpoint 同时使用；实际恢复会自动先预检。" >&2
  exit 2
fi
if ((PREFLIGHT_ONLY)) && [[ "${MERGE_AFTER_TRAINING}" == "1" ]]; then
  echo "--preflight-only 不能与 --merge/SFT_MERGE=1 同时使用" >&2
  exit 2
fi
if ((MERGE_ONLY)) && [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "--merge-only 不能与 --resume-from-checkpoint 同时使用" >&2
  exit 2
fi
if ((MERGE_ONLY)) && [[ "${MERGE_AFTER_TRAINING}" == "1" ]]; then
  echo "--merge-only 不能与 --merge/SFT_MERGE=1 同时使用" >&2
  exit 2
fi
if ((MERGE_ONLY)) && ((${#EXTRA_TRAIN_ARGS[@]})); then
  echo "--merge-only 不能接收训练器高级参数" >&2
  exit 2
fi
if [[ "${RECIPE_VARIANT}" != "canonical" ]] && [[ "${ADAPTER_DIR}" == "${DEFAULT_ADAPTER_DIR}" ]]; then
  echo "非 canonical 运行必须设置独立的 SFT_ADAPTER_DIR" >&2
  exit 2
fi
if [[ "${RECIPE_VARIANT}" != "canonical" ]] && [[ "${MERGE_AFTER_TRAINING}" == "1" ]] \
  && [[ "${MERGED_DIR}" == "${DEFAULT_MERGED_DIR}" ]]; then
  echo "非 canonical 运行若要合并，必须设置独立的 SFT_MERGED_DIR" >&2
  exit 2
fi
if [[ "${MERGE_AFTER_TRAINING}" == "1" ]] && ((MERGE_ONLY == 0)); then
  if [[ -e "${MERGED_DIR}" ]] || [[ -L "${MERGED_DIR}" ]]; then
    echo "自动合并输出必须在训练前不存在：${MERGED_DIR}" >&2
    exit 1
  fi
  case "${MERGED_DIR}/" in
    "${BASE_MODEL}/"*|"${ADAPTER_DIR}/"*)
      echo "SFT_MERGED_DIR 不能位于 base model 或 adapter 内部" >&2
      exit 1
      ;;
  esac
  case "${BASE_MODEL}/" in
    "${MERGED_DIR}/"*)
      echo "SFT_MERGED_DIR 不能是 base model 的父目录" >&2
      exit 1
      ;;
  esac
  case "${ADAPTER_DIR}/" in
    "${MERGED_DIR}/"*)
      echo "SFT_MERGED_DIR 不能是 adapter 的父目录" >&2
      exit 1
      ;;
  esac
fi

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "SFT_RUN_ID 只能包含字母、数字、点、下划线和短横线" >&2
  exit 2
fi
if [[ ! "${ATTEMPT_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "SFT_ATTEMPT_ID 只能包含字母、数字、点、下划线和短横线" >&2
  exit 2
fi
if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "缺少 SFT Python 环境；请先运行 bash scripts/setup_sft.sh" >&2
  exit 1
fi
if [[ ! -f "${SFT_PLAN}" ]]; then
  echo "missing SFT run plan: ${SFT_PLAN}" >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1 || ! command -v sha256sum >/dev/null 2>&1; then
  echo "缺少 flock 或 sha256sum；无法保证同一 run/adapter 不被并发写入。" >&2
  exit 1
fi
LOCK_ROOT="${ROOT}/outputs/locks"
mkdir -p "${LOCK_ROOT}"
ADAPTER_LOCK_KEY="$(printf '%s' "${ADAPTER_DIR}" | sha256sum | awk '{print $1}')"
RUN_LOCK_KEY="$(printf '%s' "${RUN_DIR}" | sha256sum | awk '{print $1}')"
exec 9>"${LOCK_ROOT}/sft-adapter-${ADAPTER_LOCK_KEY}.lock"
if ! flock -n 9; then
  echo "另一个 SFT 进程正在使用该 adapter：${ADAPTER_DIR}" >&2
  exit 1
fi
exec 8>"${LOCK_ROOT}/sft-run-${RUN_LOCK_KEY}.lock"
if ! flock -n 8; then
  echo "另一个 SFT 进程正在使用该 run：${RUN_DIR}" >&2
  exit 1
fi
if ((MERGE_ONLY)) || [[ "${MERGE_AFTER_TRAINING}" == "1" ]]; then
  MERGED_LOCK_KEY="$(printf '%s' "${MERGED_DIR}" | sha256sum | awk '{print $1}')"
  exec 7>"${LOCK_ROOT}/sft-merged-${MERGED_LOCK_KEY}.lock"
  if ! flock -n 7; then
    echo "另一个 SFT 进程正在使用该 merged 输出：${MERGED_DIR}" >&2
    exit 1
  fi
fi

checkpoint_is_complete() {
  local checkpoint_dir="$1"
  local required_file
  local required_checkpoint_files=(
    adapter_config.json
    trainer_state.json
    optimizer.pt
    scheduler.pt
    rng_state.pth
    training_args.bin
  )
  for required_file in "${required_checkpoint_files[@]}"; do
    [[ -s "${checkpoint_dir}/${required_file}" ]] || return 1
  done
  if [[ ! -s "${checkpoint_dir}/adapter_model.safetensors" ]] \
    && [[ ! -s "${checkpoint_dir}/adapter_model.bin" ]]; then
    return 1
  fi
  "${ROOT}/.venv/bin/python" -c \
    'import json, re, struct, sys, zipfile; from pathlib import Path
path = Path(sys.argv[1])
for name in ("adapter_config.json", "trainer_state.json"):
    with (path / name).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise SystemExit(1)
state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
match = re.fullmatch(r"checkpoint-(\d+)", path.name)
if not match or int(state.get("global_step", -1)) != int(match.group(1)):
    raise SystemExit(1)
for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"):
    if not zipfile.is_zipfile(path / name):
        raise SystemExit(1)
weights = path / "adapter_model.safetensors"
if weights.is_file():
    with weights.open("rb") as stream:
        size = weights.stat().st_size
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if header_size < 2 or header_size > size - 8:
            raise SystemExit(1)
        header = json.loads(stream.read(header_size).decode("utf-8"))
        data_bytes = size - 8 - header_size
        tensors = [value for name, value in header.items() if name != "__metadata__"]
        if not tensors:
            raise SystemExit(1)
        for tensor in tensors:
            offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise SystemExit(1)
            if not all(isinstance(value, int) for value in offsets):
                raise SystemExit(1)
            if offsets[0] < 0 or offsets[0] > offsets[1] or offsets[1] > data_bytes:
                raise SystemExit(1)
elif not zipfile.is_zipfile(path / "adapter_model.bin"):
    raise SystemExit(1)' \
    "${checkpoint_dir}" >/dev/null 2>&1
}

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_FROM_CHECKPOINT="$(resolve_path "${RESUME_FROM_CHECKPOINT}")"
  case "${RESUME_FROM_CHECKPOINT}/" in
    "${ADAPTER_DIR}/"*) ;;
    *)
      echo "checkpoint 必须位于当前 SFT_ADAPTER_DIR 内：${ADAPTER_DIR}" >&2
      exit 1
      ;;
  esac
  if [[ ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
    echo "恢复 checkpoint 不存在：${RESUME_FROM_CHECKPOINT}" >&2
    exit 1
  fi
  if ! checkpoint_is_complete "${RESUME_FROM_CHECKPOINT}"; then
    echo "checkpoint 不完整或状态 JSON 损坏：${RESUME_FROM_CHECKPOINT}" >&2
    exit 1
  fi
fi

if [[ -d "${ATTEMPT_DIR}" ]] && [[ -n "$(find "${ATTEMPT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "attempt 记录目录非空：${ATTEMPT_DIR}" >&2
  echo "请让 SFT_ATTEMPT_ID 自动生成，或显式设置一个新的值。" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/attempts"
if ! mkdir "${ATTEMPT_DIR}" 2>/dev/null; then
  echo "无法创建唯一 attempt 目录：${ATTEMPT_DIR}" >&2
  exit 1
fi
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'state=running\nstarted_at=%s\nrun_id=%s\n' \
  "${STARTED_AT}" "${RUN_ID}" > "${ATTEMPT_DIR}/status.txt"

export PYTHONUNBUFFERED=1
exec > >(tee -a "${ATTEMPT_DIR}/console.log" "${RUN_DIR}/console.log") 2>&1

GPU_MONITOR_PID=""
finish_run() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
  local state="completed"
  if ((exit_code == 130 || exit_code == 143)); then
    state="interrupted"
  elif ((exit_code != 0)); then
    state="failed"
  fi
  printf 'state=%s\nfinished_at=%s\nexit_code=%s\n' \
    "${state}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${exit_code}" >> "${ATTEMPT_DIR}/status.txt"
  exit "${exit_code}"
}
trap finish_run EXIT

if ((MERGE_ONLY == 0)) && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 30 > "${ATTEMPT_DIR}/gpu_metrics.csv" 2> "${ATTEMPT_DIR}/gpu_monitor.stderr.log" &
  GPU_MONITOR_PID=$!
fi

printf 'BASE_MODEL=%q\nTRAIN_DATA=%q\nVALIDATION_DATA=%q\nALL_DATA=%q\nMETADATA=%q\nSFT_PLAN=%q\nADAPTER_DIR=%q\nMERGED_DIR=%q\nRUN_DIR=%q\nATTEMPT_DIR=%q\n' \
  "${BASE_MODEL}" "${TRAIN_DATA}" "${VALIDATION_DATA}" "${ALL_DATA}" "${METADATA}" "${SFT_PLAN}" "${ADAPTER_DIR}" \
  "${MERGED_DIR}" "${RUN_DIR}" "${ATTEMPT_DIR}" > "${ATTEMPT_DIR}/resolved_paths.env"

cp "${SFT_PLAN}" "${ATTEMPT_DIR}/planned_recipe.json"
sha256sum "${ATTEMPT_DIR}/planned_recipe.json" | awk '{print $1}' \
  > "${ATTEMPT_DIR}/planned_recipe.sha256"
{
  printf '%q ' bash scripts/sft.sh "${ORIGINAL_ARGV[@]}"
  printf '\n'
} > "${ATTEMPT_DIR}/launcher_command.sh"
printf 'SFT_RUN_ID=%q\nSFT_ATTEMPT_ID=%q\nSFT_RECIPE_VARIANT=%q\nSFT_ATTENTION_IMPLEMENTATION=%q\nSFT_MAX_LENGTH=%q\nSFT_EPOCHS=%q\nSFT_TRAIN_BATCH_SIZE=%q\nSFT_EVAL_BATCH_SIZE=%q\nSFT_GRADIENT_ACCUMULATION_STEPS=%q\nSFT_LEARNING_RATE=%q\nSFT_WARMUP_RATIO=%q\nSFT_GRADIENT_CHECKPOINTING=%q\nSFT_MIN_GPU_MEMORY_GIB=%q\nSFT_MIN_FREE_GPU_MEMORY_GIB=%q\nSFT_MIN_FREE_DISK_GIB=%q\nSFT_MERGE=%q\nSFT_SWANLAB=%q\nSFT_SWANLAB_PROJECT=%q\nSFT_SWANLAB_RUN_NAME=%q\nSFT_SWANLAB_MODE=%q\nCUDA_VISIBLE_DEVICES=%q\n' \
  "${RUN_ID}" "${ATTEMPT_ID}" "${RECIPE_VARIANT}" "${ATTENTION_IMPLEMENTATION}" \
  "${MAX_LENGTH}" "${EPOCHS}" "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" \
  "${GRADIENT_ACCUMULATION_STEPS}" "${LEARNING_RATE}" "${WARMUP_RATIO}" "${GRADIENT_CHECKPOINTING}" \
  "${MIN_GPU_MEMORY_GIB}" "${MIN_FREE_GPU_MEMORY_GIB}" "${MIN_FREE_DISK_GIB}" "${MERGE_AFTER_TRAINING}" \
  "${SFT_SWANLAB:-0}" "${SFT_SWANLAB_PROJECT:-shopping-grpo}" \
  "${SFT_SWANLAB_RUN_NAME:-${RUN_ID}}" "${SFT_SWANLAB_MODE:-local}" \
  "${CUDA_VISIBLE_DEVICES:-}" > "${ATTEMPT_DIR}/launcher_environment.env"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PREFLIGHT_RECIPE_VARIANT="${RECIPE_VARIANT}"
PREFLIGHT_OUTPUT_ROOT="${ADAPTER_DIR}"
if ((MERGE_ONLY)); then
  PREFLIGHT_RECIPE_VARIANT="merge-only"
  PREFLIGHT_OUTPUT_ROOT="${MERGED_DIR}"
fi
preflight_args=(
  --model "${BASE_MODEL}"
  --train "${TRAIN_DATA}"
  --validation "${VALIDATION_DATA}"
  --all-data "${ALL_DATA}"
  --metadata "${METADATA}"
  --output-root "${PREFLIGHT_OUTPUT_ROOT}"
  --storage-path "${RUN_DIR}"
  --storage-path "${MERGED_DIR}"
  --min-gpu-memory-gib "${MIN_GPU_MEMORY_GIB}"
  --min-free-gpu-memory-gib "${MIN_FREE_GPU_MEMORY_GIB}"
  --min-free-disk-gib "${MIN_FREE_DISK_GIB}"
  --recipe-variant "${PREFLIGHT_RECIPE_VARIANT}"
  --attention-implementation "${ATTENTION_IMPLEMENTATION}"
  --report "${ATTEMPT_DIR}/preflight.json"
)
if ((MERGE_ONLY == 0)) && [[ -f "${RUN_DIR}/run_contract.json" ]]; then
  preflight_args+=(--compare-report "${RUN_DIR}/run_contract.json")
elif ((MERGE_ONLY == 0)) \
  && { [[ -n "${RESUME_FROM_CHECKPOINT}" ]] || [[ -f "${RUN_DIR}/training_attempts.txt" ]]; }; then
  if [[ ! -f "${RUN_DIR}/run_contract.json" ]]; then
    echo "恢复 run 缺少首次训练契约：${RUN_DIR}/run_contract.json" >&2
    exit 1
  fi
fi
if ((MERGE_ONLY)) || ((SKIP_GPU_PREFLIGHT)); then
  preflight_args+=(--skip-gpu-check)
fi
if [[ -n "${DEVICE_MAP}" ]]; then
  preflight_args+=(--allow-model-variant --allow-data-gate-policy-variant)
  if ((MERGE_ONLY == 0)) && ((SKIP_GPU_PREFLIGHT == 0)); then
    preflight_args+=(--allow-multiple-gpus)
  fi
fi
if ((MERGE_ONLY == 0)); then
  preflight_args+=(
    --tokenize-data
    --max-length "${MAX_LENGTH}"
    --epochs "${EPOCHS}"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  )
fi

cd "${ROOT}"
echo "SFT run_id=${RUN_ID}"
echo "attempt_id=${ATTEMPT_ID}"
echo "run records=${ATTEMPT_DIR}"
printf 'phase=preflight\n' >> "${ATTEMPT_DIR}/status.txt"
"${ROOT}/.venv/bin/python" scripts/check_sft_runtime.py "${preflight_args[@]}"
printf 'preflight_state=completed\n' >> "${ATTEMPT_DIR}/status.txt"

if ((PREFLIGHT_ONLY)); then
  echo "SFT 预检通过；未启动训练。报告：${ATTEMPT_DIR}/preflight.json"
  exit 0
fi

if ((MERGE_ONLY)); then
  printf 'phase=merge\nmerge_state=running\n' >> "${ATTEMPT_DIR}/status.txt"
  if "${ROOT}/.venv/bin/python" scripts/merge_lora_adapter.py \
    --base-model "${BASE_MODEL}" \
    --adapter "${ADAPTER_DIR}" \
    --output "${MERGED_DIR}" \
    --preflight-report "${ATTEMPT_DIR}/preflight.json" \
    --record-dir "${ATTEMPT_DIR}" \
    --bf16;
  then
    printf 'merge_state=completed\n' >> "${ATTEMPT_DIR}/status.txt"
  else
    merge_exit_code=$?
    merge_state="failed"
    if ((merge_exit_code == 130 || merge_exit_code == 143)); then
      merge_state="interrupted"
    fi
    printf 'merge_state=%s\n' "${merge_state}" >> "${ATTEMPT_DIR}/status.txt"
    exit "${merge_exit_code}"
  fi
  echo "merged model=${MERGED_DIR}"
  exit 0
fi

if [[ -f "${ADAPTER_DIR}/train_summary.json" ]] \
  && "${ROOT}/.venv/bin/python" -c \
    'import json, sys; raise SystemExit(json.load(open(sys.argv[1], encoding="utf-8")).get("status") != "completed")' \
    "${ADAPTER_DIR}/train_summary.json";
then
  echo "adapter 已有 completed 训练摘要，拒绝继续训练：${ADAPTER_DIR}" >&2
  echo "请执行 --merge-only，或为新实验设置新的 SFT_RUN_ID 与 SFT_ADAPTER_DIR。" >&2
  exit 1
fi

recoverable_checkpoint=""
recoverable_checkpoint_step=-1
partial_checkpoints=()
if [[ -d "${ADAPTER_DIR}" ]]; then
  while IFS= read -r -d '' checkpoint_dir; do
    if checkpoint_is_complete "${checkpoint_dir}"; then
      checkpoint_step="${checkpoint_dir##*/checkpoint-}"
      if ((10#${checkpoint_step} > recoverable_checkpoint_step)); then
        recoverable_checkpoint="${checkpoint_dir}"
        recoverable_checkpoint_step=$((10#${checkpoint_step}))
      fi
    else
      partial_checkpoints+=("${checkpoint_dir}")
    fi
  done < <(find "${ADAPTER_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print0)
fi
if ((${#partial_checkpoints[@]})) \
  && { [[ -n "${RESUME_FROM_CHECKPOINT}" ]] || [[ -f "${RUN_DIR}/training_attempts.txt" ]]; }; then
  quarantine_dir="${RUN_DIR}/partial_checkpoints/${ATTEMPT_ID}"
  mkdir -p "${quarantine_dir}"
  for checkpoint_dir in "${partial_checkpoints[@]}"; do
    checkpoint_name="$(basename "${checkpoint_dir}")"
    quarantine_target="${quarantine_dir}/${checkpoint_name}"
    mv -- "${checkpoint_dir}" "${quarantine_target}"
    "${ROOT}/.venv/bin/python" -c \
      'import json, sys; from datetime import datetime, timezone; from pathlib import Path
record = {"timestamp": datetime.now(timezone.utc).isoformat(), "reason": "incomplete_checkpoint", "source": sys.argv[2], "destination": sys.argv[3]}
with Path(sys.argv[1]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")' \
      "${ATTEMPT_DIR}/partial_checkpoint_quarantine.jsonl" \
      "${checkpoint_dir}" "${quarantine_target}"
  done
  echo "已隔离未写完整的 checkpoint：${quarantine_dir}"
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]] \
  && [[ -n "${recoverable_checkpoint}" ]] \
  && [[ "${RESUME_FROM_CHECKPOINT}" != "${recoverable_checkpoint}" ]]; then
  echo "只能从最高步的完整 checkpoint 恢复：${recoverable_checkpoint}" >&2
  echo "拒绝回退到：${RESUME_FROM_CHECKPOINT}" >&2
  exit 1
fi
if [[ -f "${RUN_DIR}/training_attempts.txt" ]] && [[ -z "${RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ -n "${recoverable_checkpoint}" ]] \
    || [[ -s "${ADAPTER_DIR}/adapter_model.safetensors" ]] \
    || [[ -s "${ADAPTER_DIR}/adapter_model.bin" ]]; then
    echo "该 run 已经产生可恢复训练状态：${RUN_DIR}" >&2
    if [[ -n "${recoverable_checkpoint}" ]]; then
      echo "恢复时请使用 --resume-from-checkpoint ${recoverable_checkpoint}" >&2
    else
      echo "adapter 根目录已有权重，但没有可验证的 checkpoint；请勿在原目录重试。" >&2
    fi
    echo "独立重跑请设置新的 SFT_RUN_ID。" >&2
    exit 1
  fi
  echo "上一次 attempt 未产生完整 checkpoint；允许在同一 run 中从头重试。"
fi

if [[ -d "${ADAPTER_DIR}" ]] && [[ -n "$(find "${ADAPTER_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
  && [[ -z "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "拒绝把新训练写入非空 adapter 目录：${ADAPTER_DIR}" >&2
  echo "请设置新的 SFT_ADAPTER_DIR，或使用 --resume-from-checkpoint。" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/run_contract.json" ]]; then
  cp "${ATTEMPT_DIR}/preflight.json" "${RUN_DIR}/run_contract.json"
fi

train_args=(
  --model "${BASE_MODEL}"
  --train "${TRAIN_DATA}"
  --validation "${VALIDATION_DATA}"
  --output "${ADAPTER_DIR}"
  --record-dir "${ATTEMPT_DIR}"
  --training-contract "${RUN_DIR}/run_contract.json"
  --max-length "${MAX_LENGTH}"
  --epochs "${EPOCHS}"
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --warmup-ratio "${WARMUP_RATIO}"
  --lora-r 16
  --lora-alpha 32
  --lora-dropout 0.05
  --dtype bf16
  --attention-implementation "${ATTENTION_IMPLEMENTATION}"
  --logging-steps 5
  --save-total-limit 3
  --seed 42
  --recipe-variant "${RECIPE_VARIANT}"
)
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  train_args+=(--gradient-checkpointing)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  train_args+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ "${SFT_SWANLAB:-0}" == "1" ]]; then
  train_args+=(
    --swanlab
    --swanlab-project "${SFT_SWANLAB_PROJECT:-shopping-grpo}"
    --swanlab-run-name "${SFT_SWANLAB_RUN_NAME:-${RUN_ID}}"
    --swanlab-mode "${SFT_SWANLAB_MODE:-local}"
  )
fi
train_args+=("${EXTRA_TRAIN_ARGS[@]}")

{
  printf '%q ' "${ROOT}/.venv/bin/python" scripts/train_lora_sft.py "${train_args[@]}"
  printf '\n'
} > "${ATTEMPT_DIR}/resolved_command.sh"

printf '%s\n' "${ATTEMPT_ID}" >> "${RUN_DIR}/training_attempts.txt"

printf 'phase=training\ntraining_state=running\n' >> "${ATTEMPT_DIR}/status.txt"
if "${ROOT}/.venv/bin/python" scripts/train_lora_sft.py "${train_args[@]}"; then
  printf 'training_state=completed\n' >> "${ATTEMPT_DIR}/status.txt"
else
  training_exit_code=$?
  training_state="failed"
  if ((training_exit_code == 130 || training_exit_code == 143)); then
    training_state="interrupted"
  fi
  printf 'training_state=%s\n' "${training_state}" >> "${ATTEMPT_DIR}/status.txt"
  exit "${training_exit_code}"
fi

if [[ "${MERGE_AFTER_TRAINING}" == "1" ]]; then
  printf 'phase=merge\nmerge_state=running\n' >> "${ATTEMPT_DIR}/status.txt"
  if "${ROOT}/.venv/bin/python" scripts/merge_lora_adapter.py \
    --base-model "${BASE_MODEL}" \
    --adapter "${ADAPTER_DIR}" \
    --output "${MERGED_DIR}" \
    --preflight-report "${ATTEMPT_DIR}/preflight.json" \
    --record-dir "${ATTEMPT_DIR}" \
    --bf16;
  then
    printf 'merge_state=completed\n' >> "${ATTEMPT_DIR}/status.txt"
  else
    merge_exit_code=$?
    merge_state="failed"
    if ((merge_exit_code == 130 || merge_exit_code == 143)); then
      merge_state="interrupted"
    fi
    printf 'merge_state=%s\n' "${merge_state}" >> "${ATTEMPT_DIR}/status.txt"
    echo "adapter 已训练完成，但自动合并失败；可修复后单独运行 --merge-only。" >&2
    exit "${merge_exit_code}"
  fi
  echo "merged model=${MERGED_DIR}"
else
  echo "训练已完成，未自动合并。核验记录后可执行："
  echo "bash scripts/sft.sh --merge-only"
fi

echo "adapter=${ADAPTER_DIR}"
echo "run records=${ATTEMPT_DIR}"
