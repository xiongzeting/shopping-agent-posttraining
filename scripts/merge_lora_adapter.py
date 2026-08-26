#!/usr/bin/env python3
"""把已完成且可审计的 SFT LoRA adapter 原子合并为 GRPO 独立起点。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import signal
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OPTIMIZER_STEPS = 177
TRAIN_SUMMARY_SCHEMA = "shopping-sft-train-summary-v1"
MERGE_MANIFEST_SCHEMA = "shopping-sft-merge-manifest-v1"
STAGING_MARKER = ".shopping-sft-merge-origin.json"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _best_effort_print(message, *, error=False):
    try:
        print(message, file=sys.stderr if error else sys.stdout)
    except (OSError, UnicodeError):
        pass


def _raise_termination(signum, frame):
    del frame
    raise InterruptedError(f"received signal {signum}")


def _load_json(path, label):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"缺少{label}：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}无法解析：{path}：{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是 JSON object：{path}")
    return value


def _artifact(path):
    path = Path(path)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _tree_artifacts(root, exclude=()):
    root = Path(root)
    excluded = set(exclude)
    return [
        {
            "name": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]


def _paths_overlap(first, second):
    first = Path(first).resolve()
    second = Path(second).resolve()
    return first == second or first in second.parents or second in first.parents


def _absolute_without_resolving_leaf(path):
    return Path(os.path.abspath(Path(path).expanduser()))


def _package_versions():
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _read_cgroup_number(path):
    path = Path(path)
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _memory_snapshot():
    meminfo = Path("/proc/meminfo")
    host_available_bytes = None
    if meminfo.is_file():
        values = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(":")
            if raw:
                values[key] = int(raw.strip().split()[0])
        available_kib = values.get("MemAvailable")
        if available_kib:
            host_available_bytes = available_kib * 1024

    cgroup_available_bytes = None
    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ):
        limit = _read_cgroup_number(limit_path)
        usage = _read_cgroup_number(usage_path)
        if limit is not None and usage is not None:
            cgroup_available_bytes = max(0, limit - usage)
            break

    candidates = [
        value
        for value in (host_available_bytes, cgroup_available_bytes)
        if value is not None
    ]
    effective_bytes = min(candidates) if candidates else None

    def as_gib(value):
        return round(value / 1024**3, 2) if value is not None else None

    return {
        "host_available_gib": as_gib(host_available_bytes),
        "cgroup_available_gib": as_gib(cgroup_available_bytes),
        "effective_available_gib": as_gib(effective_bytes),
    }


def _cleanup_stale_staging(output):
    cleaned = []
    prefix = f".{output.name}.tmp-"
    for candidate in output.parent.glob(f"{prefix}*"):
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        marker_path = candidate / STAGING_MARKER
        try:
            marker = _load_json(marker_path, "merge staging marker")
        except ValueError:
            continue
        if (
            marker.get("schema_version") != "shopping-sft-merge-staging-v1"
            or Path(str(marker.get("output", ""))).resolve() != output
            or candidate.parent.resolve() != output.parent.resolve()
        ):
            continue
        shutil.rmtree(candidate)
        cleaned.append(str(candidate))
    return cleaned


def choose_model_class(config, causal_model_class, multimodal_model_class):
    """Qwen3.5 走官方多模态类；其他 CausalLM 保持普通路径。"""

    if str(getattr(config, "model_type", "")).startswith("qwen3_5"):
        return multimodal_model_class
    return causal_model_class


def build_merge_manifest(base_model, adapter_path, output_path, model_type, **details):
    """构造向后兼容且可扩展的合并清单。"""

    manifest = {
        "schema_version": MERGE_MANIFEST_SCHEMA,
        "operation": "peft_merge_and_unload",
        "source": {
            "base_model": str(base_model),
            "adapter": str(adapter_path),
            "model_type": str(model_type),
        },
        "output": str(output_path),
        "next_step": (
            "load this standalone checkpoint as GRPO base and attach a new LoRA adapter"
        ),
    }
    manifest.update(details)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="原子合并 LoRA SFT adapter，为 GRPO 创建 BF16 起点")
    parser.add_argument("--base-model", required=True, help="与 SFT 完全一致的本地基座模型目录")
    parser.add_argument("--adapter", type=Path, required=True, help="已完成的 SFT LoRA adapter")
    parser.add_argument("--output", type=Path, required=True, help="新的 merged checkpoint，必须不存在")
    parser.add_argument("--bf16", action="store_true", help="以 BF16 在 CPU 上合并并保存")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--preflight-report",
        type=Path,
        required=True,
        help="本次模型、数据与运行时预检报告；sft.sh 会自动传入",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="合并记录目录；sft.sh 使用当前 attempt 目录",
    )
    return parser.parse_args()


def _validate_adapter(adapter, base_model, output, preflight_report):
    errors = []
    if not adapter.is_dir():
        raise ValueError(f"adapter 目录不存在：{adapter}")

    adapter_config = _load_json(adapter / "adapter_config.json", "adapter_config.json")
    summary = _load_json(adapter / "train_summary.json", "train_summary.json")
    contract = _load_json(adapter / "training_contract.json", "training_contract.json")

    if summary.get("schema_version") != TRAIN_SUMMARY_SCHEMA:
        errors.append(
            f"train_summary schema 不匹配：expected={TRAIN_SUMMARY_SCHEMA} "
            f"actual={summary.get('schema_version')}"
        )
    if summary.get("status") != "completed":
        errors.append("adapter 的 train_summary.status 必须为 completed")
    if adapter_config.get("peft_type") != "LORA":
        errors.append(f"adapter peft_type 必须为 LORA：{adapter_config.get('peft_type')}")
    if adapter_config.get("task_type") != "CAUSAL_LM":
        errors.append(f"adapter task_type 必须为 CAUSAL_LM：{adapter_config.get('task_type')}")
    declared_base = adapter_config.get("base_model_name_or_path")
    if declared_base:
        declared_path = Path(str(declared_base)).expanduser()
        if declared_path.exists() and declared_path.resolve() != base_model:
            errors.append(
                "adapter_config.base_model_name_or_path 与本次 --base-model 不一致"
            )

    weight_candidates = [
        path
        for path in (adapter / "adapter_model.safetensors", adapter / "adapter_model.bin")
        if path.is_file()
    ]
    if len(weight_candidates) != 1 or weight_candidates[0].stat().st_size == 0:
        errors.append("adapter 必须且只能包含一个非空 adapter_model.safetensors/bin")

    expected_artifacts = {
        item.get("name"): item
        for item in summary.get("adapter_artifacts", [])
        if isinstance(item, dict) and item.get("name")
    }
    actual_root_files = {
        path.name: path
        for path in adapter.iterdir()
        if path.is_file() and path.name != "train_summary.json"
    }
    if set(actual_root_files) != set(expected_artifacts):
        errors.append(
            "adapter 根目录文件集合与训练摘要不一致："
            f"expected={sorted(expected_artifacts)} actual={sorted(actual_root_files)}"
        )
    for path in actual_root_files.values():
        actual = _artifact(path)
        expected = expected_artifacts.get(path.name)
        if not expected:
            errors.append(f"train_summary 未记录 adapter artifact：{path.name}")
        elif actual.get("bytes") != expected.get("bytes") or actual.get("sha256") != expected.get(
            "sha256"
        ):
            errors.append(f"adapter artifact 已变化：{path.name}")

    arguments = summary.get("arguments") or {}
    expected_config = {
        "r": arguments.get("lora_r"),
        "lora_alpha": arguments.get("lora_alpha"),
        "lora_dropout": arguments.get("lora_dropout"),
    }
    for key, expected in expected_config.items():
        if expected is not None and adapter_config.get(key) != expected:
            errors.append(
                f"adapter_config.{key} 与训练摘要不一致："
                f"expected={expected} actual={adapter_config.get(key)}"
            )
    expected_targets = set(arguments.get("target_modules") or [])
    actual_targets = set(adapter_config.get("target_modules") or [])
    if expected_targets and actual_targets != expected_targets:
        errors.append("adapter_config.target_modules 与训练摘要不一致")

    recipe = summary.get("recipe") or {}
    canonical = bool(recipe.get("canonical")) and recipe.get("name") == "canonical"
    canonical_output = (ROOT / "outputs/models/sft-merged").resolve()
    if not canonical and output == canonical_output:
        errors.append("非 canonical adapter 必须使用独立的 SFT_MERGED_DIR")
    if canonical:
        if summary.get("optimizer_steps") != EXPECTED_OPTIMIZER_STEPS:
            errors.append(
                "canonical adapter optimizer_steps 不完整："
                f"expected={EXPECTED_OPTIMIZER_STEPS} actual={summary.get('optimizer_steps')}"
            )
        if int(arguments.get("max_steps", -1)) > 0:
            errors.append("canonical adapter 不能来自 --max-steps 截断训练")
        if not summary.get("last_eval") or int(summary.get("validation_examples", 0)) <= 0:
            errors.append("canonical adapter 缺少完整验证记录")

    summary_contract = summary.get("training_contract") or {}
    contract_hash = _sha256(adapter / "training_contract.json")
    if summary_contract.get("sha256") != contract_hash:
        errors.append("adapter 内 training_contract.json 与 train_summary 记录不一致")
    if contract.get("status") != "passed":
        errors.append("首次训练 training_contract.status 必须为 passed")

    current_preflight = None
    if preflight_report:
        current_preflight = _load_json(preflight_report, "merge preflight report")
        if current_preflight.get("status") != "passed":
            errors.append("本次 merge preflight 未通过")
        current_model = ((current_preflight.get("checks") or {}).get("model") or {})
        if Path(current_model.get("path", ".")).resolve() != base_model:
            errors.append("merge preflight 校验的 base model 与本次 --base-model 不一致")
        contract_checks = contract.get("checks") or {}
        current_checks = current_preflight.get("checks") or {}
        for split in ("train", "validation", "evaluation"):
            expected_sha = ((contract_checks.get("datasets") or {}).get(split) or {}).get(
                "sha256"
            )
            actual_sha = ((current_checks.get("datasets") or {}).get(split) or {}).get(
                "sha256"
            )
            if expected_sha != actual_sha:
                errors.append(f"当前 {split} 数据与首次训练契约不一致")
        contract_model = contract_checks.get("model") or {}
        for key in ("metadata_sha256", "weight_files", "lora_target_parameter_counts"):
            if contract_model.get(key) != current_model.get(key):
                errors.append(f"当前 base model 的 {key} 与首次训练契约不一致")

    summary_data = summary.get("data") or {}
    contract_data = ((contract.get("checks") or {}).get("datasets") or {})
    for split in ("train", "validation"):
        if (summary_data.get(split) or {}).get("sha256") != (
            contract_data.get(split) or {}
        ).get("sha256"):
            errors.append(f"train_summary 的 {split} 数据哈希与首次训练契约不一致")

    if errors:
        raise ValueError("\n".join(f"- {error}" for error in errors))
    return {
        "adapter_config": adapter_config,
        "summary": summary,
        "contract": contract,
        "current_preflight": current_preflight,
        "canonical": canonical,
        "weight": _artifact(weight_candidates[0]),
        "config": _artifact(adapter / "adapter_config.json"),
        "summary_artifact": _artifact(adapter / "train_summary.json"),
        "contract_artifact": _artifact(adapter / "training_contract.json"),
    }


def _verify_saved_model(output, model_class, auto_config, auto_processor, dtype):
    from safetensors import safe_open

    if (output / "adapter_config.json").exists() or list(output.glob("adapter_model*")):
        raise ValueError("merged 输出仍残留 PEFT adapter 文件")
    config = auto_config.from_pretrained(
        str(output), trust_remote_code=True, local_files_only=True
    )
    auto_processor.from_pretrained(str(output), trust_remote_code=True, local_files_only=True)

    index_candidates = list(output.glob("*.safetensors.index.json"))
    if index_candidates:
        index = _load_json(index_candidates[0], "merged safetensors index")
        names = sorted({str(value) for value in (index.get("weight_map") or {}).values()})
        weight_files = [output / name for name in names]
    else:
        weight_files = sorted(output.glob("*.safetensors"))
    if not weight_files or any(not path.is_file() for path in weight_files):
        raise ValueError("merged 输出权重分片缺失")

    dtype_counts = {}
    tensor_count = 0
    for path in weight_files:
        with safe_open(path, framework="pt", device="cpu") as stream:
            for key in stream.keys():
                tensor_count += 1
                tensor_dtype = str(stream.get_slice(key).get_dtype())
                dtype_counts[tensor_dtype] = dtype_counts.get(tensor_dtype, 0) + 1
    if tensor_count == 0:
        raise ValueError("merged safetensors 不含 tensor")
    floating_dtypes = {
        name: count
        for name, count in dtype_counts.items()
        if name in {"BF16", "F16", "F32", "F64"}
    }
    if not floating_dtypes or set(floating_dtypes) != {"BF16"}:
        raise ValueError(f"merged 浮点权重必须全部为 BF16：{dtype_counts}")

    verification_model = model_class.from_pretrained(
        str(output),
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )
    if hasattr(verification_model, "peft_config"):
        raise ValueError("离线重载后模型仍是 PEFT wrapper")
    del verification_model
    gc.collect()
    return {
        "model_type": str(getattr(config, "model_type", "")),
        "weight_files": [path.name for path in weight_files],
        "tensor_count": tensor_count,
        "dtype_counts": dtype_counts,
        "offline_reload": "passed",
    }


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, _raise_termination)
    started_at = _utc_now()
    base_model = Path(args.base_model).expanduser().resolve()
    adapter = args.adapter.expanduser().resolve()
    output = _absolute_without_resolving_leaf(args.output)
    record_dir = args.record_dir.expanduser().resolve() if args.record_dir else None
    temp_output = None
    publish_attempted = False

    try:
        if not base_model.is_dir():
            raise ValueError(f"base model 必须是本地目录：{base_model}")
        if not args.bf16:
            raise ValueError("本项目的 SFT→GRPO 合并固定使用 BF16；必须传入 --bf16")
        if os.path.lexists(output):
            raise ValueError(f"merged 输出必须不存在，拒绝覆盖：{output}")
        if _paths_overlap(output, base_model) or _paths_overlap(output, adapter):
            raise ValueError("merged 输出不能与 base/adapter 相同或互为父子目录")
        if record_dir and any(
            _paths_overlap(record_dir, path) for path in (base_model, adapter, output)
        ):
            raise ValueError("--record-dir 不能位于 base、adapter 或 merged 输出内部")
        memory = _memory_snapshot()
        available_memory_gib = memory["effective_available_gib"]
        if available_memory_gib is not None and available_memory_gib < 16:
            raise ValueError(
                f"CPU merge 可用内存仅 {available_memory_gib} GiB；至少需要 16 GiB"
            )

        preflight = (
            args.preflight_report.expanduser().resolve() if args.preflight_report else None
        )
        validation = _validate_adapter(adapter, base_model, output, preflight)

        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForMultimodalLM,
                AutoProcessor,
            )
        except ImportError as exc:
            raise ValueError("缺少合并依赖；请执行：bash scripts/setup_sft.sh") from exc

        config = AutoConfig.from_pretrained(
            str(base_model), trust_remote_code=True, local_files_only=True
        )
        model_class = choose_model_class(
            config, AutoModelForCausalLM, AutoModelForMultimodalLM
        )
        dtype = torch.bfloat16
        output.parent.mkdir(parents=True, exist_ok=True)
        stale_staging_cleaned = _cleanup_stale_staging(output)
        temp_output = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
        )
        _write_json(
            temp_output / STAGING_MARKER,
            {
                "schema_version": "shopping-sft-merge-staging-v1",
                "created_at": _utc_now(),
                "output": str(output),
            },
        )

        print(f"CPU 加载 base={base_model} model_type={config.model_type} dtype={dtype}")
        base = model_class.from_pretrained(
            str(base_model),
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
        )
        merged = PeftModel.from_pretrained(base, str(adapter), is_trainable=False).merge_and_unload()
        merged = merged.to(dtype=dtype)
        merged.save_pretrained(
            str(temp_output),
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )

        processor_source = adapter
        processor = AutoProcessor.from_pretrained(
            str(adapter), trust_remote_code=True, local_files_only=True
        )
        processor.save_pretrained(str(temp_output))
        del processor, merged, base
        gc.collect()

        verification = _verify_saved_model(
            temp_output,
            model_class=model_class,
            auto_config=AutoConfig,
            auto_processor=AutoProcessor,
            dtype=dtype,
        )
        output_artifacts = _tree_artifacts(temp_output, exclude=("merge_manifest.json",))
        manifest = build_merge_manifest(
            base_model=base_model,
            adapter_path=adapter,
            output_path=output,
            model_type=config.model_type,
            status="completed",
            started_at=started_at,
            finished_at=_utc_now(),
            atomic_publish=True,
            merge_device="cpu",
            dtype=str(dtype),
            max_shard_size=args.max_shard_size,
            memory=memory,
            stale_staging_cleaned=stale_staging_cleaned,
            canonical=validation["canonical"],
            recipe=validation["summary"].get("recipe"),
            processor_source=str(processor_source),
            inputs={
                "adapter_config": validation["config"],
                "adapter_weight": validation["weight"],
                "train_summary": validation["summary_artifact"],
                "training_contract": validation["contract_artifact"],
                "merge_preflight": _artifact(preflight) if preflight else None,
                "base_weight_files": (
                    ((validation["current_preflight"] or {}).get("checks") or {})
                    .get("model", {})
                    .get("weight_files")
                ),
            },
            dependencies=_package_versions(),
            verification=verification,
            output_artifacts=output_artifacts,
        )
        _write_json(temp_output / "merge_manifest.json", manifest)
        publish_attempted = True
        os.replace(temp_output, output)
        temp_output = None
        if record_dir:
            try:
                _write_json(record_dir / "merge_manifest.json", manifest)
            except Exception as exc:
                _best_effort_print(
                    f"警告：merged 已成功原子发布，但 attempt manifest 复制失败：{exc}",
                    error=True,
                )
        _best_effort_print(json.dumps(manifest, ensure_ascii=False))
    except BaseException as exc:
        committed_after_signal = (
            publish_attempted
            and output.is_dir()
            and (output / "merge_manifest.json").is_file()
            and (temp_output is None or not temp_output.exists())
        )
        if committed_after_signal:
            _best_effort_print(
                "警告：收到发布后的中断或记录异常；merged 已完整原子发布，按成功处理。",
                error=True,
            )
            return
        failure = {
            "schema_version": "shopping-sft-merge-failure-v1",
            "status": "failed",
            "started_at": started_at,
            "failed_at": _utc_now(),
            "base_model": str(base_model),
            "adapter": str(adapter),
            "output": str(output),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if record_dir:
            try:
                _write_json(record_dir / "merge_failure.json", failure)
            except Exception as record_exc:
                _best_effort_print(
                    f"警告：无法写入 merge_failure.json：{record_exc}", error=True
                )
        if temp_output and temp_output.is_dir():
            try:
                shutil.rmtree(temp_output)
            except OSError as cleanup_exc:
                _best_effort_print(
                    f"警告：临时 merged 目录清理失败：{temp_output}：{cleanup_exc}",
                    error=True,
                )
        raise


if __name__ == "__main__":
    main()
