#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
MAIN_PYTHON="${MAIN_PYTHON:-3.12}"
MIN_GPU_MEMORY_GIB="${SFT_MIN_GPU_MEMORY_GIB:-94}"
MIN_FREE_GPU_MEMORY_GIB="${SFT_MIN_FREE_GPU_MEMORY_GIB:-92}"
MIN_FREE_DISK_GIB="${SFT_MIN_FREE_DISK_GIB:-50}"
SETUP_MIN_FREE_DISK_GIB="${SFT_SETUP_MIN_FREE_DISK_GIB:-20}"
ADAPTER_DIR="${SFT_ADAPTER_DIR:-${ROOT}/outputs/models/sft-lora}"
MERGED_DIR="${SFT_MERGED_DIR:-${ROOT}/outputs/models/sft-merged}"
RUN_DIR="${SFT_RUN_DIR:-${ROOT}/outputs/runs/sft}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "SFT 训练环境仅支持 Linux。" >&2
  exit 1
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "canonical SFT 锁文件只支持 Linux x86_64，当前架构：$(uname -m)" >&2
  exit 1
fi
glibc_description="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
glibc_version="$(awk '{print $2}' <<<"${glibc_description}")"
if [[ "${glibc_description}" != glibc\ * ]] || [[ -z "${glibc_version}" ]]; then
  echo "无法确认 glibc 版本；canonical SFT 要求 glibc >= 2.28。" >&2
  exit 1
fi
if [[ "$(printf '%s\n' "2.28" "${glibc_version}" | sort -V | head -n 1)" != "2.28" ]]; then
  echo "glibc 版本过低：${glibc_version}；至少需要 2.28。" >&2
  exit 1
fi
if [[ "${MAIN_PYTHON}" != "3.12" ]]; then
  echo "canonical SFT 固定使用 Python 3.12；不要覆盖 MAIN_PYTHON=${MAIN_PYTHON}。" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "缺少 uv，请先安装：https://docs.astral.sh/uv/" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "缺少 git；锁定的 Transformers 依赖来自固定 Git revision。" >&2
  exit 1
fi
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]] \
  && [[ "${UV_PROJECT_ENVIRONMENT}" != ".venv" ]] \
  && [[ "${UV_PROJECT_ENVIRONMENT}" != "${ROOT}/.venv" ]]; then
  echo "UV_PROJECT_ENVIRONMENT 指向仓库外：${UV_PROJECT_ENVIRONMENT}" >&2
  echo "请取消该变量；本项目固定使用 ${ROOT}/.venv。" >&2
  exit 1
fi

available_kib="$(df -Pk "${ROOT}" | awk 'END {print $4}')"
required_kib="$(awk -v gib="${SETUP_MIN_FREE_DISK_GIB}" 'BEGIN {printf "%.0f", gib * 1024 * 1024}')"
if [[ ! "${available_kib}" =~ ^[0-9]+$ ]] || ((available_kib < required_kib)); then
  echo "仓库/.venv 文件系统可用空间不足；安装前至少需要 ${SETUP_MIN_FREE_DISK_GIB} GiB。" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "未找到 nvidia-smi；请先确认租用实例已挂载 NVIDIA GPU 与驱动。" >&2
  exit 1
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "nvidia-smi 无法访问 GPU；请检查驱动、容器 GPU 挂载与实例状态。" >&2
  exit 1
fi
uv_cache_dir="$(uv cache dir)"
uv_cache_parent="${uv_cache_dir}"
while [[ ! -e "${uv_cache_parent}" ]] && [[ "${uv_cache_parent}" != "/" ]]; do
  uv_cache_parent="$(dirname "${uv_cache_parent}")"
done
if [[ ! -d "${uv_cache_parent}" ]] || [[ ! -w "${uv_cache_parent}" ]]; then
  echo "uv 缓存路径的现有父目录不可写：${uv_cache_parent}" >&2
  exit 1
fi
cache_available_kib="$(df -Pk "${uv_cache_parent}" | awk 'END {print $4}')"
cache_required_kib=$((10 * 1024 * 1024))
if [[ ! "${cache_available_kib}" =~ ^[0-9]+$ ]] \
  || ((cache_available_kib < cache_required_kib)); then
  echo "uv 缓存文件系统可用空间不足；安装前至少需要 10 GiB：${uv_cache_dir}" >&2
  exit 1
fi

extras=(--extra sft)
runtime_recipe_variant="canonical"
if [[ "${SFT_INSTALL_ACCELERATED:-0}" == "1" ]]; then
  extras+=(--extra sft-accelerated)
  runtime_recipe_variant="liger+qlora"
fi
if [[ "${SFT_INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  extras+=(--extra sft-flash-attn)
  runtime_recipe_variant="flash-attn2+${runtime_recipe_variant}"
fi

cd "${ROOT}"
UV_PROJECT_ENVIRONMENT="${ROOT}/.venv" \
  MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-4}" \
  uv sync --project "${ROOT}" --locked --python "${MAIN_PYTHON}" \
    --no-build-isolation-package flash-attn "${extras[@]}"
UV_PROJECT_ENVIRONMENT="${ROOT}/.venv" \
  uv sync --project "${ROOT}" --locked --check "${extras[@]}"

mkdir -p "${ROOT}/outputs/setup"
"${ROOT}/.venv/bin/python" scripts/check_sft_runtime.py \
  --data-only \
  --report "${ROOT}/outputs/setup/sft-data-preflight.json"
"${ROOT}/.venv/bin/python" scripts/check_sft_runtime.py \
  --runtime-only \
  --output-root "${ROOT}/outputs" \
  --storage-path "${RUN_DIR}" \
  --storage-path "${ADAPTER_DIR}" \
  --storage-path "${MERGED_DIR}" \
  --min-gpu-memory-gib "${MIN_GPU_MEMORY_GIB}" \
  --min-free-gpu-memory-gib "${MIN_FREE_GPU_MEMORY_GIB}" \
  --min-free-disk-gib "${MIN_FREE_DISK_GIB}" \
  --recipe-variant "${runtime_recipe_variant}" \
  --attention-implementation "$([[ "${SFT_INSTALL_FLASH_ATTN:-0}" == "1" ]] && printf flash_attention_2 || printf sdpa)" \
  --report "${ROOT}/outputs/setup/sft-runtime-preflight.json"

"${ROOT}/.venv/bin/python" - "${ROOT}/.venv" <<'PY'
import sys
from importlib.metadata import version

expected_prefix = sys.argv[1]
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected Python 3.12, got {sys.version}")
if sys.prefix != expected_prefix:
    raise SystemExit(f"unexpected environment: {sys.prefix} != {expected_prefix}")
for package in ("torch", "torchvision", "transformers", "peft", "accelerate", "swanlab"):
    print(f"{package}={version(package)}")
PY

echo
echo "SFT 依赖、数据与 CUDA/BF16/SDPA 运行时已准备完成。"
echo "安装报告：outputs/setup/sft-data-preflight.json 与 sft-runtime-preflight.json"
echo "把 Qwen3.5-2B 放到 models/Qwen3.5-2B，或设置 BASE_MODEL=/你的/模型目录。"
echo "模型就位后先运行：bash scripts/sft.sh --preflight-only"
