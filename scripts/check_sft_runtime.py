#!/usr/bin/env python3
"""在加载模型权重或开始训练前校验 canonical SFT 运行条件。"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from shopping_grpo.collection.data_gate import DATA_GATE_VERSION, DEFAULT_POLICY
from shopping_grpo.collection.sft import (
    COLLECTION_SCHEMA_VERSION,
    TEACHER_SELECTION_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TRANSFORMERS_REVISION = "7ea2320c76117e6742364808a666ef6f2fb40a67"
EXPECTED_PACKAGE_VERSIONS = {
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "transformers": "5.15.0.dev0",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "swanlab": "0.9.1",
}
EXPECTED_OPTIONAL_PACKAGE_VERSIONS = {
    "liger-kernel": "0.8.1",
    "bitsandbytes": "0.50.0",
    "flash-attn": "2.8.3.post1",
    "flash-linear-attention": "0.5.2",
}
EXPECTED_MODEL_SNAPSHOT = "15852e8c16360a2fea060d615a32b45270f8a8fc"
EXPECTED_MODEL_WEIGHT_BYTES = 4_548_221_488
EXPECTED_MODEL_TENSOR_BYTES = 4_548_144_832
EXPECTED_MODEL_WEIGHT_SHA256 = "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
EXPECTED_MODEL_METADATA_SHA256 = {
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "config.json": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "model.safetensors.index.json": (
        "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9"
    ),
    "preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": (
        "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"
    ),
    "video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}
CANONICAL_MIN_GPU_MEMORY_GIB = 94.0
CANONICAL_MIN_FREE_GPU_MEMORY_GIB = 92.0
PREFLIGHT_SCHEMA = "shopping-sft-preflight-v3"
EXPECTED_SFT_SCHEMA = "shopping-sft-dataset-v3"
EXPECTED_EVALUATION_SCHEMA = "shopping-evaluation-dataset-v2.2"
EXPECTED_SFT_ENVIRONMENT = "shopsimulator-environment-v2.4"
EXPECTED_SFT_REWARD = "shopsimulator-reward-v3.2"
EXPECTED_SFT_TERMINATION = "shopping-termination-v3.1"
EXPECTED_SFT_OBSERVATION = "shopping-observation-v2"
EXPECTED_SFT_TOOL_SCHEMA = "shopping-tools-v2"
EXPECTED_TEACHER_SELECTION = TEACHER_SELECTION_VERSION
EXPECTED_EVALUATION_ENVIRONMENT = "shopsimulator-environment-v2.4"
EXPECTED_EVALUATION_REWARD = "shopsimulator-reward-v4"
BLIND_TASK_IDS = ROOT / "src/shopping_grpo/resources/blind_final_task_ids.json"
FORBIDDEN_TRAINING_KEYS = {
    "reasoning_content",
    "teacher_reasoning",
    "terminal_result",
    "reward",
    "reward_valid",
    "reward_breakdown",
}
EXPECTED_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


def parse_args():
    parser = argparse.ArgumentParser(description="检查 SFT 数据、本地模型、依赖、GPU 与磁盘")
    parser.add_argument(
        "--model",
        default=os.environ.get("BASE_MODEL", str(ROOT / "models/Qwen3.5-2B")),
        help="本地 Qwen3.5-2B 模型目录；预检不会联网下载模型",
    )
    parser.add_argument("--train", type=Path, default=ROOT / "data/sft/train.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/sft/validation.jsonl")
    parser.add_argument(
        "--all-data",
        type=Path,
        default=ROOT / "data/sft/all.jsonl",
        help="canonical train/validation 合集",
    )
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/sft/metadata.json")
    parser.add_argument(
        "--evaluation-tasks", type=Path, default=ROOT / "data/evaluation/tasks.jsonl"
    )
    parser.add_argument(
        "--evaluation-metadata", type=Path, default=ROOT / "data/evaluation/metadata.json"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/models/sft-lora"
    )
    parser.add_argument("--report", type=Path, default=None, help="可选 JSON 报告路径")
    parser.add_argument(
        "--compare-report",
        type=Path,
        default=None,
        help="恢复训练时要求当前关键指纹与首次训练预检完全一致",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--data-only",
        action="store_true",
        help="只校验数据及留出集隔离；无需模型、PyTorch 或 NVIDIA GPU",
    )
    mode_group.add_argument(
        "--runtime-only",
        action="store_true",
        help="只校验 Linux、锁定依赖、CUDA/BF16/SDPA 与磁盘；模型可尚未下载",
    )
    parser.add_argument(
        "--min-gpu-memory-gib",
        type=float,
        default=CANONICAL_MIN_GPU_MEMORY_GIB,
        help="canonical BF16 LoRA 的最低可见显存；显式变体可自行降低",
    )
    parser.add_argument(
        "--min-free-gpu-memory-gib",
        type=float,
        default=CANONICAL_MIN_FREE_GPU_MEMORY_GIB,
        help="训练启动前第一张可见 GPU 的最低空闲显存",
    )
    parser.add_argument(
        "--min-free-disk-gib",
        type=float,
        default=50.0,
        help="模型输出所在文件系统的最低可用空间",
    )
    parser.add_argument(
        "--storage-path",
        type=Path,
        action="append",
        default=[],
        help="额外检查一个将写入运行记录、adapter 或 merged 模型的路径；可重复",
    )
    parser.add_argument(
        "--tokenize-data",
        action="store_true",
        help="使用本地 processor 完整渲染 train/validation，但不加载模型权重",
    )
    parser.add_argument("--max-length", type=int, default=30000)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--min-kept-ratio", type=float, default=0.9)
    parser.add_argument(
        "--recipe-variant",
        default="canonical",
        help="记录显式训练变体；canonical 以外的值不会被当作忠实复现",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="flash_attention_2",
        help="Attention backend required by this SFT run.",
    )
    parser.add_argument(
        "--allow-non-linux",
        action="store_true",
        help="仅供开发机静态检查；正式训练应使用 Linux",
    )
    parser.add_argument(
        "--allow-multiple-gpus",
        action="store_true",
        help="允许多个 GPU 对当前进程可见；canonical 复现应只暴露一张卡",
    )
    parser.add_argument(
        "--allow-model-variant",
        action="store_true",
        help="允许使用其他 Qwen3.5 尺寸；仍校验模型结构、分片完整性和 LoRA targets。",
    )
    parser.add_argument(
        "--allow-data-gate-policy-variant",
        action="store_true",
        help="允许已冻结数据集使用不同版本的数据门 policy；仍要求报告 passed、无缺口且哈希一致。",
    )
    parser.add_argument(
        "--allow-fp16",
        action="store_true",
        help="允许不支持 BF16 的 GPU；这不属于 canonical BF16 复现",
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="仅供 CPU merge 预检；仍检查锁定的 PyTorch/CUDA wheel，但不要求可用 GPU",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str, errors: list[str]) -> dict:
    if not path.is_file():
        errors.append(f"缺少{label}：{path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"无法读取{label} {path}：{exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label}必须是 JSON object：{path}")
        return {}
    return value


def _read_jsonl(path: Path, label: str, errors: list[str]) -> tuple[list[dict], set[int]]:
    rows: list[dict] = []
    task_ids: set[int] = set()
    if not path.is_file():
        errors.append(f"缺少{label}：{path}")
        return rows, task_ids
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL 行不是 object")
                if "task_id" not in value:
                    raise ValueError("缺少 task_id")
                task_id = int(value["task_id"])
                task_ids.add(task_id)
                rows.append(value)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        errors.append(f"{label}格式错误 {path}:{line_number if 'line_number' in locals() else '?'}：{exc}")
    return rows, task_ids


def _forbidden_keys(value) -> set[str]:
    found = set()
    if isinstance(value, dict):
        found.update(FORBIDDEN_TRAINING_KEYS & set(value))
        for nested in value.values():
            found.update(_forbidden_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_forbidden_keys(nested))
    return found


def _validate_sft_rows(rows: list[dict], label: str, errors: list[str]) -> dict:
    trajectory_ids: set[str] = set()
    duplicate_trajectory_ids: set[str] = set()
    seen_task_ids: set[int] = set()
    duplicate_task_ids: set[int] = set()
    tool_schema_hashes: set[str] = set()
    malformed = 0
    for row in rows:
        messages = row.get("messages")
        tools = row.get("tools")
        trajectory_id = row.get("trajectory_id")
        task_id = int(row["task_id"])
        if task_id in seen_task_ids:
            duplicate_task_ids.add(task_id)
        seen_task_ids.add(task_id)
        if not isinstance(messages, list) or not messages:
            malformed += 1
            continue
        if not isinstance(tools, list):
            malformed += 1
            continue
        declared_tool_names = set()
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(name, str) or not name:
                malformed += 1
                continue
            declared_tool_names.add(name)
        if not declared_tool_names:
            errors.append(f"{label}轨迹 {trajectory_id} 的工具 schema 为空")
        tool_schema_hashes.add(
            hashlib.sha256(
                json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
        )
        if not any(isinstance(message, dict) and message.get("role") == "assistant" for message in messages):
            malformed += 1
            continue
        if not isinstance(trajectory_id, str) or not trajectory_id:
            malformed += 1
            continue
        if trajectory_id in trajectory_ids:
            duplicate_trajectory_ids.add(trajectory_id)
        trajectory_ids.add(trajectory_id)

        forbidden = _forbidden_keys(row)
        if forbidden:
            errors.append(f"{label}轨迹 {trajectory_id} 含审计专用字段：{sorted(forbidden)}")

        issued_calls = {}
        returned_calls = set()
        pending_calls = []
        last_tool_name = None
        call_error = False
        for message in messages:
            if not isinstance(message, dict):
                call_error = True
                continue
            if message.get("role") == "assistant":
                tool_calls = message.get("tool_calls") or []
                if len(tool_calls) > 1:
                    call_error = True
                for call in tool_calls:
                    if not isinstance(call, dict):
                        call_error = True
                        continue
                    call_id = call.get("id")
                    function = call.get("function")
                    if not isinstance(call_id, str) or not call_id or not isinstance(function, dict):
                        call_error = True
                        continue
                    if call_id in issued_calls:
                        call_error = True
                    name = function.get("name")
                    arguments = function.get("arguments")
                    try:
                        parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    except json.JSONDecodeError:
                        parsed_arguments = None
                    if not isinstance(name, str) or not isinstance(parsed_arguments, dict):
                        call_error = True
                    if name not in declared_tool_names:
                        call_error = True
                    issued_calls[call_id] = name
                    pending_calls.append(call_id)
                    last_tool_name = name
            elif message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                if (
                    not isinstance(call_id, str)
                    or call_id not in issued_calls
                    or call_id in returned_calls
                    or not pending_calls
                    or pending_calls[0] != call_id
                    or message.get("name") != issued_calls.get(call_id)
                ):
                    call_error = True
                else:
                    returned_calls.add(call_id)
                    pending_calls.pop(0)
        if call_error or pending_calls or set(issued_calls) != returned_calls:
            errors.append(f"{label}轨迹 {trajectory_id} 的工具调用与返回不完整配对")
        if last_tool_name != "buy_now":
            errors.append(f"{label}轨迹 {trajectory_id} 的最终动作不是 buy_now")
        final_is_purchase = False
        if len(messages) >= 2:
            final_assistant = messages[-2]
            final_tool = messages[-1]
            final_calls = (
                final_assistant.get("tool_calls")
                if isinstance(final_assistant, dict)
                and final_assistant.get("role") == "assistant"
                else None
            )
            if isinstance(final_calls, list) and len(final_calls) == 1:
                final_call = final_calls[0]
                final_function = final_call.get("function") if isinstance(final_call, dict) else None
                final_is_purchase = (
                    isinstance(final_function, dict)
                    and final_function.get("name") == "buy_now"
                    and isinstance(final_tool, dict)
                    and final_tool.get("role") == "tool"
                    and final_tool.get("name") == "buy_now"
                    and final_tool.get("tool_call_id") == final_call.get("id")
                )
        if not final_is_purchase:
            errors.append(f"{label}轨迹 {trajectory_id} 不是以 buy_now tool return 完整终止")
    if malformed:
        errors.append(f"{label}有 {malformed} 行不满足 SFT schema")
    if duplicate_trajectory_ids:
        errors.append(
            f"{label}存在重复 trajectory_id：{sorted(duplicate_trajectory_ids)[:5]}"
        )
    if duplicate_task_ids:
        errors.append(f"{label}存在重复 task_id：{sorted(duplicate_task_ids)[:10]}")
    if len(tool_schema_hashes) != 1:
        errors.append(f"{label}包含 {len(tool_schema_hashes)} 个不同工具 schema")
    return {
        "rows": len(rows),
        "task_count": len({int(row["task_id"]) for row in rows if "task_id" in row}),
        "trajectory_count": len(trajectory_ids),
        "trajectory_ids": trajectory_ids,
        "tool_schema_sha256": next(iter(tool_schema_hashes), None),
    }


def _validate_data_gate(
    metadata: dict,
    metadata_path: Path,
    expected_rows: int,
    errors: list[str],
    allow_policy_variant: bool = False,
) -> dict:
    record = metadata.get("data_gate")
    if not isinstance(record, dict):
        errors.append("SFT metadata 缺少 data_gate 审计记录")
        return {}
    if record.get("schema_version") != DATA_GATE_VERSION:
        errors.append(f"SFT metadata data_gate.schema_version 必须为 {DATA_GATE_VERSION}")
    if record.get("status") != "passed":
        errors.append("SFT metadata data_gate.status 必须为 passed")

    raw_report_path = record.get("path")
    if not isinstance(raw_report_path, str) or not raw_report_path.strip():
        errors.append("SFT metadata data_gate.path 必须指向审计报告")
        return {}
    report_path = Path(raw_report_path)
    if report_path.is_absolute():
        errors.append("SFT metadata data_gate.path 必须是相对 metadata 的路径")
        return {}
    metadata_root = metadata_path.parent.resolve()
    report_path = (metadata_root / report_path).resolve()
    try:
        report_path.relative_to(metadata_root)
    except ValueError:
        errors.append("SFT metadata data_gate.path 不得离开 canonical 数据目录")
        return {}

    report = _load_json(report_path, "Teacher data gate report", errors)
    if not report:
        return {"path": str(report_path)}
    if report.get("schema_version") != DATA_GATE_VERSION:
        errors.append(f"Teacher data gate report schema 必须为 {DATA_GATE_VERSION}")
    if report.get("status") != "passed":
        errors.append("Teacher data gate report status 必须为 passed")
    if report.get("policy") != DEFAULT_POLICY and not allow_policy_variant:
        errors.append("Teacher data gate report policy 与 canonical 数据门不一致")
    if report.get("rows") != expected_rows:
        errors.append(
            "Teacher data gate report 行数与 canonical SFT 不一致："
            f"expected={expected_rows} actual={report.get('rows')}"
        )
    if report.get("unique_task_ids") != expected_rows:
        errors.append("Teacher data gate report 必须覆盖互不重复的全部 canonical task_id")
    if report.get("deficits"):
        errors.append("Teacher data gate report 仍包含未满足的数据门缺口")
    audit = report.get("audit")
    if not isinstance(audit, dict):
        errors.append("Teacher data gate report 缺少可复现 audit 记录")
    else:
        if audit.get("collection_schema_version") != COLLECTION_SCHEMA_VERSION:
            errors.append(
                "Teacher data gate report collection schema 必须为 "
                f"{COLLECTION_SCHEMA_VERSION}"
            )
        if audit.get("teacher_selection") != EXPECTED_TEACHER_SELECTION:
            errors.append(
                "Teacher data gate report teacher selection 必须为 "
                f"{EXPECTED_TEACHER_SELECTION}"
            )
        if audit.get("search_contract") != "shopsimulator-multifield-bm25-v2.1":
            errors.append("Teacher data gate report 必须使用 Search v2.1")
        if audit.get("audited_rows") != expected_rows:
            errors.append("Teacher data gate report audit 行数与 canonical SFT 不一致")
        for key in ("input_sha256", "products_sha256", "search_index_sha256"):
            value = audit.get(key)
            if not isinstance(value, str) or len(value) != 64:
                errors.append(f"Teacher data gate report audit.{key} 缺少有效 SHA-256")

    actual_sha256 = sha256_file(report_path)
    if record.get("sha256") != actual_sha256:
        errors.append(
            "Teacher data gate report SHA-256 不匹配："
            f"expected={record.get('sha256')} actual={actual_sha256}"
        )
    return {
        "path": str(report_path),
        "sha256": actual_sha256,
        "schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "rows": report.get("rows"),
    }


def validate_datasets(args, errors: list[str]) -> dict:
    metadata = _load_json(args.metadata, "SFT metadata", errors)
    evaluation_metadata = _load_json(args.evaluation_metadata, "evaluation metadata", errors)
    all_rows, all_ids = _read_jsonl(args.all_data, "SFT 全量集", errors)
    train_rows, train_ids = _read_jsonl(args.train, "SFT 训练集", errors)
    validation_rows, validation_ids = _read_jsonl(args.validation, "SFT 验证集", errors)
    evaluation_rows, evaluation_ids = _read_jsonl(
        args.evaluation_tasks, "冻结评估集", errors
    )

    all_summary = _validate_sft_rows(all_rows, "SFT 全量集", errors)
    train_summary = _validate_sft_rows(train_rows, "SFT 训练集", errors)
    validation_summary = _validate_sft_rows(validation_rows, "SFT 验证集", errors)

    for document, label, expected_schema, expected_environment, expected_reward in (
        (
            metadata,
            "SFT metadata",
            EXPECTED_SFT_SCHEMA,
            EXPECTED_SFT_ENVIRONMENT,
            EXPECTED_SFT_REWARD,
        ),
        (
            evaluation_metadata,
            "evaluation metadata",
            EXPECTED_EVALUATION_SCHEMA,
            EXPECTED_EVALUATION_ENVIRONMENT,
            EXPECTED_EVALUATION_REWARD,
        ),
    ):
        if document.get("schema_version") != expected_schema:
            errors.append(
                f"{label} schema 不匹配：expected={expected_schema} actual={document.get('schema_version')}"
            )
        if document.get("environment") != expected_environment:
            errors.append(f"{label} environment 必须为 {expected_environment}")
        if document.get("reward") != expected_reward:
            errors.append(f"{label} reward 必须为 {expected_reward}")

    for key, expected in (
        ("termination", EXPECTED_SFT_TERMINATION),
        ("observation", EXPECTED_SFT_OBSERVATION),
        ("tool_schema", EXPECTED_SFT_TOOL_SCHEMA),
        ("teacher_selection", EXPECTED_TEACHER_SELECTION),
    ):
        if metadata.get(key) != expected:
            errors.append(f"SFT metadata {key} 必须为 {expected}")
    if metadata.get("status") != "current":
        errors.append("SFT metadata status 必须为 current")
    data_gate_summary = _validate_data_gate(
        metadata,
        args.metadata,
        len(all_rows),
        errors,
        allow_policy_variant=bool(
            getattr(args, "allow_data_gate_policy_variant", False)
        ),
    )

    for split_name, path, rows in (
        ("all", args.all_data, all_rows),
        ("train", args.train, train_rows),
        ("validation", args.validation, validation_rows),
    ):
        expected = metadata.get(split_name)
        if not isinstance(expected, dict):
            errors.append(f"SFT metadata 缺少 {split_name} 记录")
            continue
        if path.is_file():
            actual_hash = sha256_file(path)
            if actual_hash != expected.get("sha256"):
                errors.append(
                    f"{split_name} SHA-256 不匹配：expected={expected.get('sha256')} actual={actual_hash}"
                )
            try:
                expected_rows = int(expected.get("rows", -1))
            except (TypeError, ValueError):
                expected_rows = -1
                errors.append(f"SFT metadata {split_name}.rows 必须是整数")
            if len(rows) != expected_rows:
                errors.append(
                    f"{split_name} 行数不匹配：expected={expected.get('rows')} actual={len(rows)}"
                )

    if args.evaluation_tasks.is_file():
        actual_eval_hash = sha256_file(args.evaluation_tasks)
        expected_eval_hash = evaluation_metadata.get("task_sha256")
        if expected_eval_hash is None:
            expected_eval_hash = evaluation_metadata.get("sha256")
        if actual_eval_hash != expected_eval_hash:
            errors.append(
                "冻结评估集 SHA-256 不匹配："
                f"expected={expected_eval_hash} actual={actual_eval_hash}"
            )
        try:
            expected_evaluation_rows = int(evaluation_metadata.get("tasks", -1))
        except (TypeError, ValueError):
            expected_evaluation_rows = -1
            errors.append("evaluation metadata tasks 必须是整数")
        if len(evaluation_rows) != expected_evaluation_rows:
            errors.append(
                "冻结评估集行数不匹配："
                f"expected={evaluation_metadata.get('tasks')} actual={len(evaluation_rows)}"
            )

    train_validation_overlap = sorted(train_ids & validation_ids)
    final_overlap = sorted((train_ids | validation_ids) & evaluation_ids)
    if train_validation_overlap:
        errors.append(
            f"SFT train/validation task_id 重叠：{train_validation_overlap[:10]}"
        )
    if final_overlap:
        errors.append(f"SFT 与冻结评估集 task_id 重叠：{final_overlap[:10]}")
    split_ids = train_ids | validation_ids
    if all_ids != split_ids:
        missing = sorted(split_ids - all_ids)
        unexpected = sorted(all_ids - split_ids)
        errors.append(
            "SFT all 与 train/validation task_id 并集不一致："
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    all_trajectory_ids = all_summary["trajectory_ids"]
    trajectory_overlap = sorted(
        train_summary["trajectory_ids"] & validation_summary["trajectory_ids"]
    )
    if trajectory_overlap:
        errors.append(f"SFT train/validation trajectory_id 重叠：{trajectory_overlap[:5]}")
    split_trajectory_ids = (
        train_summary["trajectory_ids"] | validation_summary["trajectory_ids"]
    )
    if all_trajectory_ids != split_trajectory_ids:
        errors.append("SFT all 与 train/validation trajectory_id 并集不一致")
    if all_summary["tool_schema_sha256"] != train_summary["tool_schema_sha256"]:
        errors.append("SFT all/train 使用了不同的工具 schema")
    if train_summary["tool_schema_sha256"] != validation_summary["tool_schema_sha256"]:
        errors.append("SFT train/validation 使用了不同的工具 schema")

    for key in (
        "train_validation_overlap",
        "final_240_overlap",
        "final_240_asin_overlap",
        "final_240_family_overlap",
        "final_240_semantic_overlap",
    ):
        if metadata.get(key) != 0:
            errors.append(f"SFT metadata {key} 必须为 0")

    blind_document = _load_json(BLIND_TASK_IDS, "打包的 blind-final task IDs", errors)
    blind_ids = blind_document.get("task_ids")
    if not isinstance(blind_ids, list) or not all(isinstance(value, int) for value in blind_ids):
        errors.append("打包的 blind-final task IDs 格式错误")
        blind_id_set = set()
    else:
        blind_id_set = set(blind_ids)
        if blind_id_set != evaluation_ids:
            errors.append("data/evaluation 与打包的 blind-final task IDs 不一致")

    all_summary.pop("trajectory_ids", None)
    train_summary.pop("trajectory_ids", None)
    validation_summary.pop("trajectory_ids", None)

    return {
        "metadata": str(args.metadata.resolve()),
        "data_gate": data_gate_summary,
        "all": {
            **all_summary,
            "path": str(args.all_data.resolve()),
            "sha256": sha256_file(args.all_data) if args.all_data.is_file() else None,
        },
        "train": {
            **train_summary,
            "path": str(args.train.resolve()),
            "sha256": sha256_file(args.train) if args.train.is_file() else None,
        },
        "validation": {
            **validation_summary,
            "path": str(args.validation.resolve()),
            "sha256": sha256_file(args.validation) if args.validation.is_file() else None,
        },
        "evaluation": {
            "rows": len(evaluation_rows),
            "task_count": len(evaluation_ids),
            "path": str(args.evaluation_tasks.resolve()),
            "sha256": sha256_file(args.evaluation_tasks)
            if args.evaluation_tasks.is_file()
            else None,
        },
        "blind_final_task_count": len(blind_id_set),
        "train_validation_overlap": len(train_validation_overlap),
        "evaluation_overlap": len(final_overlap),
    }


def _weight_files(model_dir: Path, config: dict, errors: list[str]) -> list[Path]:
    del config
    index_candidates = sorted(model_dir.glob("*.safetensors.index.json")) + sorted(
        model_dir.glob("*.bin.index.json")
    )
    if index_candidates:
        index = _load_json(index_candidates[0], "模型权重索引", errors)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            errors.append(f"模型权重索引没有有效 weight_map：{index_candidates[0]}")
            return []
        names = sorted({str(value) for value in weight_map.values()})
        files = [model_dir / name for name in names]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            errors.append(f"模型权重分片缺失：{missing[:5]}")
        return [path for path in files if path.is_file()]

    files = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("pytorch_model*.bin"))
    if not files:
        errors.append(f"模型目录没有 safetensors/bin 权重：{model_dir}")
    return files


def _validate_safetensors_file(path: Path, errors: list[str]) -> dict:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(200)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            errors.append(f"模型权重仍是 Git LFS pointer，未下载真实内容：{path}")
            return {"valid": False, "tensor_count": 0}
        if size < 8:
            errors.append(f"safetensors 文件过小：{path}")
            return {"valid": False, "tensor_count": 0}
        stream.seek(0)
        raw_header_length = stream.read(8)
        header_length = struct.unpack("<Q", raw_header_length)[0]
        if header_length < 2 or header_length > min(256 * 1024 * 1024, size - 8):
            errors.append(f"safetensors header 长度非法：{path} header={header_length}")
            return {"valid": False, "tensor_count": 0}
        try:
            header = json.loads(stream.read(header_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"safetensors header 无法解析：{path}：{exc}")
            return {"valid": False, "tensor_count": 0}
    if not isinstance(header, dict):
        errors.append(f"safetensors header 不是 object：{path}")
        return {"valid": False, "tensor_count": 0}
    data_bytes = size - 8 - header_length
    tensor_count = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        tensor_count += 1
        if not isinstance(metadata, dict):
            errors.append(f"safetensors tensor metadata 非法：{path}:{name}")
            continue
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > data_bytes
        ):
            errors.append(f"safetensors data_offsets 越界：{path}:{name}")
    if tensor_count == 0:
        errors.append(f"safetensors 不含 tensor：{path}")
    return {
        "valid": tensor_count > 0,
        "header_bytes": header_length,
        "tensor_count": tensor_count,
        "data_bytes": data_bytes,
    }


def validate_model(args, errors: list[str], warnings: list[str]) -> dict:
    model_dir = Path(args.model).expanduser()
    if not model_dir.is_absolute():
        model_dir = (ROOT / model_dir).resolve()
    else:
        model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        errors.append(
            f"BASE_MODEL 必须指向完整的本地模型目录，且预检不会联网下载：{model_dir}"
        )
        return {"path": str(model_dir), "exists": False}

    config_path = model_dir / "config.json"
    config = _load_json(config_path, "模型 config.json", errors)
    model_type = str(config.get("model_type", ""))
    if not model_type.startswith("qwen3_5"):
        errors.append(f"期望 Qwen3.5 模型，实际 model_type={model_type or '<missing>'}")
    canonical_model = not getattr(args, "allow_model_variant", False)
    expected_architecture = ["Qwen3_5ForConditionalGeneration"]
    if config.get("architectures") != expected_architecture:
        errors.append(
            "Qwen3.5-2B architectures 不匹配："
            f"expected={expected_architecture} actual={config.get('architectures')}"
        )
    text_config = config.get("text_config")
    expected_text_config = {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 24,
        "vocab_size": 248320,
    }
    if not isinstance(text_config, dict):
        errors.append("Qwen3.5-2B config 缺少 text_config")
    elif canonical_model:
        for key, expected_value in expected_text_config.items():
            if text_config.get(key) != expected_value:
                errors.append(
                    f"Qwen3.5-2B text_config.{key} 不匹配："
                    f"expected={expected_value} actual={text_config.get(key)}"
                )

    if not (model_dir / "tokenizer_config.json").is_file():
        errors.append(f"模型目录缺少 tokenizer_config.json：{model_dir}")
    processor_candidates = (
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer.model",
    )
    if not any((model_dir / name).is_file() for name in processor_candidates):
        errors.append(f"模型目录缺少 processor/tokenizer 资产：{model_dir}")

    critical_metadata = {}
    for name, expected_sha256 in EXPECTED_MODEL_METADATA_SHA256.items():
        path = model_dir / name
        if not path.is_file():
            if canonical_model:
                errors.append(f"固定 Qwen3.5-2B snapshot 缺少关键 metadata：{name}")
            continue
        actual_sha256 = sha256_file(path)
        critical_metadata[name] = actual_sha256
        if canonical_model and actual_sha256 != expected_sha256:
            errors.append(
                f"Qwen3.5-2B metadata 哈希不匹配：{name} "
                f"expected={expected_sha256} actual={actual_sha256}"
            )

    weights = _weight_files(model_dir, config, errors)
    weight_records = []
    for path in weights:
        record = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".safetensors":
            record["safetensors"] = _validate_safetensors_file(path, errors)
        weight_records.append(record)
    if canonical_model and len(weight_records) != 1:
        errors.append(
            f"固定 Qwen3.5-2B snapshot 应只有 1 个权重文件，实际为 {len(weight_records)}"
        )
    elif canonical_model and weight_records[0]["bytes"] != EXPECTED_MODEL_WEIGHT_BYTES:
        errors.append(
            "Qwen3.5-2B 权重大小不匹配："
            f"expected={EXPECTED_MODEL_WEIGHT_BYTES} actual={weight_records[0]['bytes']}"
        )
    elif canonical_model and weight_records[0]["sha256"] != EXPECTED_MODEL_WEIGHT_SHA256:
        errors.append(
            "Qwen3.5-2B 权重 SHA-256 不匹配："
            f"expected={EXPECTED_MODEL_WEIGHT_SHA256} actual={weight_records[0]['sha256']}"
        )

    index_candidates = sorted(model_dir.glob("*.safetensors.index.json")) + sorted(
        model_dir.glob("*.bin.index.json")
    )
    target_matches = {}
    if index_candidates:
        index_document = _load_json(index_candidates[0], "模型权重索引", errors)
        index_total_size = (index_document.get("metadata") or {}).get("total_size")
        if canonical_model and index_total_size != EXPECTED_MODEL_TENSOR_BYTES:
            errors.append(
                "Qwen3.5-2B index total_size 不匹配："
                f"expected={EXPECTED_MODEL_TENSOR_BYTES} actual={index_total_size}"
            )
        parameter_names = list((index_document.get("weight_map") or {}).keys())
        for target in EXPECTED_LORA_TARGETS:
            matches = [
                name
                for name in parameter_names
                if f".{target}." in name or name.endswith(f".{target}")
            ]
            target_matches[target] = len(matches)
            if not matches:
                errors.append(f"模型权重索引未发现 LoRA target module：{target}")
    else:
        warnings.append("模型没有权重索引，无法在预检阶段核对 LoRA target modules")

    weight_paths = {path.resolve() for path in weights}
    metadata_hashes = {}
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file() or path.resolve() in weight_paths:
            continue
        relative = path.relative_to(model_dir)
        if ".cache" in relative.parts or path.suffix in {".safetensors", ".bin"}:
            continue
        metadata_hashes[relative.as_posix()] = sha256_file(path)

    loader_check = "not_run"
    flash_attention_2_support = "not_requested"
    try:
        from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

        loaded_config = AutoConfig.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
        AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
        if getattr(args, "attention_implementation", "sdpa") == "flash_attention_2":
            model_class = AutoModelForMultimodalLM._model_mapping[type(loaded_config)]
            if getattr(model_class, "_supports_flash_attn", False):
                flash_attention_2_support = "passed"
            else:
                flash_attention_2_support = "failed"
                errors.append(
                    f"{model_class.__name__} does not declare Flash Attention 2 support"
                )
        loader_check = "passed"
    except ImportError:
        warnings.append("尚未安装 Transformers，模型 loader 检查将在 setup_sft.sh 后执行")
    except Exception as exc:
        loader_check = "failed"
        errors.append(f"本地 AutoConfig/AutoProcessor 加载失败：{exc}")

    return {
        "path": str(model_dir),
        "exists": True,
        "model_type": model_type,
        "expected_snapshot": EXPECTED_MODEL_SNAPSHOT if canonical_model else None,
        "expected_metadata_sha256": EXPECTED_MODEL_METADATA_SHA256 if canonical_model else None,
        "model_variant_allowed": bool(getattr(args, "allow_model_variant", False)),
        "critical_metadata_sha256": critical_metadata,
        "architectures": config.get("architectures"),
        "metadata_sha256": metadata_hashes,
        "local_loader_check": loader_check,
        "flash_attention_2_support": flash_attention_2_support,
        "lora_target_parameter_counts": target_matches,
        "weight_files": weight_records,
        "weight_bytes": sum(record["bytes"] for record in weight_records),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _transformers_revision() -> str | None:
    try:
        distribution = importlib.metadata.distribution("transformers")
    except importlib.metadata.PackageNotFoundError:
        return None
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    try:
        return json.loads(raw).get("vcs_info", {}).get("commit_id")
    except json.JSONDecodeError:
        return None


def _existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _nvidia_smi_snapshot() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return {}
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            rows.append(
                {
                    "index": fields[0],
                    "name": fields[1],
                    "driver_version": fields[2],
                    "memory_total_mib": fields[3],
                    "memory_free_mib": fields[4],
                }
            )
    return {"gpus": rows}


def _runtime_package_names(args) -> list[str]:
    package_names = list(EXPECTED_PACKAGE_VERSIONS)
    if "liger" in args.recipe_variant:
        package_names.append("liger-kernel")
    if "qlora" in args.recipe_variant:
        package_names.append("bitsandbytes")
    if args.attention_implementation == "flash_attention_2":
        package_names.append("flash-attn")
    if "fla" in args.recipe_variant:
        package_names.append("flash-linear-attention")
    return package_names


def _optional_package_version_matches(package: str, actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    if package == "flash-attn" and actual.startswith("2.8.3+"):
        return True
    return False


def validate_runtime(args, errors: list[str], warnings: list[str]) -> dict:
    system = platform.system()
    if system != "Linux" and not args.allow_non_linux:
        errors.append(f"正式 SFT 仅支持 Linux，当前系统为 {system}")
    if sys.version_info[:2] != (3, 12):
        errors.append(
            f"canonical SFT 要求 Python 3.12，当前为 {platform.python_version()}"
        )
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        errors.append(f"canonical 锁文件要求 Linux x86_64，当前架构为 {machine}")
    libc_name, libc_version = platform.libc_ver()
    if system == "Linux":
        if libc_name != "glibc":
            errors.append(
                f"canonical SFT 要求 glibc >= 2.28，当前 libc={libc_name or '<unknown>'}"
            )
        else:
            try:
                libc_tuple = tuple(int(part) for part in libc_version.split(".")[:2])
            except ValueError:
                libc_tuple = (0, 0)
            if libc_tuple < (2, 28):
                errors.append(f"glibc 至少需要 2.28，当前为 {libc_version or '<unknown>'}")

    package_names = _runtime_package_names(args)
    packages = {
        name: _package_version(name)
        for name in package_names
    }
    missing = [name for name in EXPECTED_PACKAGE_VERSIONS if not packages[name]]
    if missing:
        errors.append(f"缺少 SFT 依赖：{', '.join(missing)}；请先运行 bash scripts/setup_sft.sh")
    for package, expected in EXPECTED_PACKAGE_VERSIONS.items():
        actual = packages.get(package)
        if actual is not None and actual != expected:
            errors.append(f"{package} 版本不匹配：expected={expected} actual={actual}")
    for package, expected in EXPECTED_OPTIONAL_PACKAGE_VERSIONS.items():
        if package not in package_names:
            continue
        actual = packages.get(package)
        if actual is None:
            errors.append(
                f"变体 {args.recipe_variant} 缺少 {package}；请运行 "
                "SFT_INSTALL_ACCELERATED=1 bash scripts/setup_sft.sh"
            )
        elif not _optional_package_version_matches(package, actual, expected):
            errors.append(f"{package} 版本不匹配：expected={expected} actual={actual}")

    import_smoke = {}
    modules = {
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
        "peft": "peft",
        "accelerate": "accelerate",
        "swanlab": "swanlab",
    }
    if "liger-kernel" in package_names:
        modules["liger-kernel"] = "liger_kernel"
    if "bitsandbytes" in package_names:
        modules["bitsandbytes"] = "bitsandbytes"
    if "flash-attn" in package_names:
        modules["flash-attn"] = "flash_attn"
    if "flash-linear-attention" in package_names:
        modules["flash-linear-attention"] = "fla"
    for package, module_name in modules.items():
        try:
            importlib.import_module(module_name)
            import_smoke[package] = "passed"
        except Exception as exc:
            import_smoke[package] = f"failed: {type(exc).__name__}: {exc}"
            errors.append(f"{package} 导入失败：{exc}")

    transformers_revision = _transformers_revision()
    if packages["transformers"] and transformers_revision != EXPECTED_TRANSFORMERS_REVISION:
        errors.append(
            "Transformers revision 不匹配："
            f"expected={EXPECTED_TRANSFORMERS_REVISION} actual={transformers_revision or '<unknown>'}"
        )

    cuda = {}
    try:
        import torch

        cuda = {
            "torch_cuda": torch.version.cuda,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "check_skipped": bool(args.skip_gpu_check),
        }
        if torch.version.cuda and not str(torch.version.cuda).startswith("13."):
            errors.append(
                f"canonical 锁文件使用 CUDA 13 PyTorch wheel，当前 torch CUDA={torch.version.cuda}"
            )
        devices = []
        if args.skip_gpu_check:
            cuda["available"] = None
            cuda["device_count"] = None
            cuda["bf16_supported"] = None
            if args.attention_implementation == "flash_attention_2":
                cuda["bf16_flash_attention_2_smoke_test"] = "skipped_no_gpu"
        else:
            cuda_available = bool(torch.cuda.is_available())
            cuda.update(
                {
                    "available": cuda_available,
                    "cudnn": torch.backends.cudnn.version(),
                    "device_count": torch.cuda.device_count(),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported())
                    if cuda_available
                    else False,
                }
            )
            if cuda_available:
                for index in range(torch.cuda.device_count()):
                    properties = torch.cuda.get_device_properties(index)
                    memory_gib = properties.total_memory / 1024**3
                    free_bytes, _ = torch.cuda.mem_get_info(index)
                    devices.append(
                        {
                            "index": index,
                            "name": properties.name,
                            "capability": [properties.major, properties.minor],
                            "total_memory_gib": round(memory_gib, 2),
                            "free_memory_gib": round(free_bytes / 1024**3, 2),
                        }
                    )
                if torch.cuda.device_count() != 1 and not args.allow_multiple_gpus:
                    errors.append(
                        "canonical SFT 要求只暴露一张 GPU；请设置 CUDA_VISIBLE_DEVICES=<index>"
                    )
                if devices and devices[0]["total_memory_gib"] < args.min_gpu_memory_gib:
                    errors.append(
                        f"可见 GPU 显存 {devices[0]['total_memory_gib']} GiB 低于 canonical 要求 "
                        f"{args.min_gpu_memory_gib} GiB"
                    )
                if devices and devices[0]["free_memory_gib"] < args.min_free_gpu_memory_gib:
                    errors.append(
                        f"第一张可见 GPU 当前空闲显存 {devices[0]['free_memory_gib']} GiB 低于要求 "
                        f"{args.min_free_gpu_memory_gib} GiB；请停止模型服务和其他 GPU 进程"
                    )
                if not cuda["bf16_supported"] and not args.allow_fp16:
                    errors.append("当前 GPU 不支持 BF16，无法忠实复现 canonical SFT")
                if cuda["bf16_supported"]:
                    sample = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
                    float((sample * sample).sum().item())
                    query = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.bfloat16)
                    torch.nn.functional.scaled_dot_product_attention(query, query, query)
                    torch.cuda.synchronize()
                    del sample, query
                    cuda["bf16_smoke_test"] = "passed"
                    cuda["bf16_sdpa_smoke_test"] = "passed"
                    if args.attention_implementation == "flash_attention_2":
                        properties = torch.cuda.get_device_properties(0)
                        if properties.major < 8:
                            errors.append(
                                "Flash Attention 2 requires compute capability >= 8.0"
                            )
                        else:
                            try:
                                from flash_attn import flash_attn_func

                                flash_query = torch.randn(
                                    1,
                                    128,
                                    2,
                                    64,
                                    device="cuda",
                                    dtype=torch.bfloat16,
                                    requires_grad=True,
                                )
                                flash_output = flash_attn_func(
                                    flash_query,
                                    flash_query,
                                    flash_query,
                                    causal=True,
                                )
                                flash_output.sum().backward()
                                torch.cuda.synchronize()
                                del flash_query, flash_output
                                cuda["bf16_flash_attention_2_smoke_test"] = "passed"
                            except Exception as exc:
                                errors.append(
                                    "Flash Attention 2 CUDA smoke test failed: "
                                    f"{type(exc).__name__}: {exc}"
                                )
            else:
                errors.append(
                    "PyTorch 无法使用 CUDA；请检查 NVIDIA Driver 与 CUDA 13 wheel 兼容性"
                )
        cuda["devices"] = devices
    except ImportError:
        pass
    except Exception as exc:  # CUDA 初始化错误需要进入持久化报告。
        errors.append(f"CUDA 初始化失败：{exc}")

    storage_records = []
    storage_paths = [args.output_root, *args.storage_path]
    seen_storage_paths = set()
    for requested_path in storage_paths:
        resolved_path = requested_path.expanduser().resolve()
        if resolved_path in seen_storage_paths:
            continue
        seen_storage_paths.add(resolved_path)
        disk_parent = _existing_parent(resolved_path)
        writable = True
        write_error = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=".shopping-sft-write-check-",
                dir=disk_parent,
            ):
                pass
        except OSError as exc:
            writable = False
            write_error = str(exc)
            errors.append(f"目标路径不可写：{resolved_path}（现有父目录 {disk_parent}）：{exc}")
        disk = shutil.disk_usage(disk_parent)
        free_gib = disk.free / 1024**3
        if free_gib < args.min_free_disk_gib:
            errors.append(
                f"目标 {resolved_path} 所在文件系统可用空间 {free_gib:.1f} GiB "
                f"低于要求 {args.min_free_disk_gib:.1f} GiB"
            )
        elif free_gib < 80:
            warnings.append(
                f"目标 {resolved_path} 可用磁盘少于 80 GiB；保留 uv 缓存、模型、"
                "checkpoint 和 merged 权重时可能偏紧"
            )
        storage_records.append(
            {
                "requested_path": str(resolved_path),
                "existing_parent": str(disk_parent),
                "writable": writable,
                "write_error": write_error,
                "free_gib": round(free_gib, 2),
                "total_gib": round(disk.total / 1024**3, 2),
            }
        )

    return {
        "platform": {
            "system": system,
            "release": platform.release(),
            "machine": platform.machine(),
            "libc": {"name": libc_name, "version": libc_version},
            "python": platform.python_version(),
        },
        "packages": packages,
        "import_smoke": import_smoke,
        "transformers_revision": transformers_revision,
        "cuda": cuda,
        "nvidia_smi": _nvidia_smi_snapshot(),
        "disk": storage_records[0] if storage_records else {},
        "storage": storage_records,
    }


def validate_tokenization(args, errors: list[str]) -> dict:
    """用真实本地 chat template 做完整数据渲染，但不加载权重。"""

    try:
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        from shopping_grpo.training.sft.dataset import audit_supervised_examples
    except ImportError as exc:
        errors.append(f"无法执行 tokenizer-only 检查：{exc}")
        return {}

    model_dir = Path(args.model).expanduser()
    if not model_dir.is_absolute():
        model_dir = (ROOT / model_dir).resolve()
    else:
        model_dir = model_dir.resolve()
    try:
        config = AutoConfig.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
        is_multimodal = str(getattr(config, "model_type", "")).startswith("qwen3_5")
        if is_multimodal:
            processor = AutoProcessor.from_pretrained(
                str(model_dir), trust_remote_code=True, local_files_only=True
            )
            tokenizer = processor.tokenizer
            template = processor if getattr(processor, "chat_template", None) else tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir), trust_remote_code=True, local_files_only=True
            )
            template = tokenizer
        train_stats = audit_supervised_examples(
            args.train,
            tokenizer=tokenizer,
            chat_template=template,
            max_length=args.max_length,
        )
        validation_stats = audit_supervised_examples(
            args.validation,
            tokenizer=tokenizer,
            chat_template=template,
            max_length=args.max_length,
        )
    except Exception as exc:
        errors.append(f"tokenizer-only 数据渲染失败：{exc}")
        return {}

    if not train_stats.get("kept"):
        errors.append("tokenizer-only 检查后训练集没有可用样本")
    if not validation_stats.get("kept"):
        errors.append("tokenizer-only 检查后验证集没有可用样本")
    for split_name, stats in (("train", train_stats), ("validation", validation_stats)):
        kept_ratio = stats["kept"] / stats["total"] if stats["total"] else 0.0
        stats["kept_ratio"] = round(kept_ratio, 6)
        if kept_ratio < args.min_kept_ratio:
            errors.append(
                f"tokenizer-only {split_name} kept ratio {kept_ratio:.3f} "
                f"低于要求 {args.min_kept_ratio:.3f}"
            )
    updates_per_epoch = math.ceil(
        math.ceil(train_stats["kept"] / args.train_batch_size)
        / args.gradient_accumulation_steps
    )
    expected_optimizer_steps = math.ceil(args.epochs * updates_per_epoch)
    return {
        "max_length": args.max_length,
        "train": train_stats,
        "validation": validation_stats,
        "expected_optimizer_steps": expected_optimizer_steps,
        "formula": {
            "epochs": args.epochs,
            "per_device_train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "visible_processes": 1,
        },
    }


def _git_snapshot() -> dict:
    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=15,
        )
        relative_paths = [
            Path(value.decode("utf-8"))
            for value in completed.stdout.split(b"\0")
            if value
        ]
        source_files = sorted(
            path for path in (ROOT / relative for relative in relative_paths) if path.is_file()
        )
    except (FileNotFoundError, subprocess.SubprocessError, UnicodeDecodeError):
        excluded = {".git", "outputs", "models", "checkpoints", "logs", "swanlog", "wandb"}
        source_files = sorted(
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and not (excluded & set(path.relative_to(ROOT).parts))
            and not any(part.startswith(".venv") for part in path.relative_to(ROOT).parts)
            and "__pycache__" not in path.parts
        )
    digest = hashlib.sha256()
    for path in source_files:
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "source_tree_sha256": digest.hexdigest(),
        "source_file_count": len(source_files),
        "pyproject_sha256": sha256_file(ROOT / "pyproject.toml"),
        "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
    }


def _resume_contract(report: dict) -> dict:
    checks = report.get("checks") or {}
    datasets = checks.get("datasets") or {}
    model = checks.get("model") or {}
    runtime = checks.get("runtime") or {}
    tokenization = checks.get("tokenization") or {}
    return {
        "recipe": report.get("recipe"),
        "source": {
            key: (report.get("source") or {}).get(key)
            for key in ("source_tree_sha256", "pyproject_sha256", "uv_lock_sha256")
        },
        "datasets": {
            split: {
                key: (datasets.get(split) or {}).get(key)
                for key in ("sha256", "rows", "tool_schema_sha256")
            }
            for split in ("all", "train", "validation", "evaluation")
        },
        "model": {
            key: model.get(key)
            for key in (
                "model_type",
                "architectures",
                "metadata_sha256",
                "weight_files",
                "lora_target_parameter_counts",
            )
        },
        "runtime": {
            "packages": runtime.get("packages"),
            "transformers_revision": runtime.get("transformers_revision"),
            "torch_cuda": (runtime.get("cuda") or {}).get("torch_cuda"),
        },
        "tokenization": {
            "max_length": tokenization.get("max_length"),
            "train": tokenization.get("train"),
            "validation": tokenization.get("validation"),
            "expected_optimizer_steps": tokenization.get("expected_optimizer_steps"),
        },
    }


def compare_resume_contract(report: dict, reference_path: Path, errors: list[str]) -> dict:
    reference = _load_json(reference_path, "首次训练预检报告", errors)
    if not reference:
        return {"reference": str(reference_path), "matched": False}
    current_contract = _resume_contract(report)
    reference_contract = _resume_contract(reference)
    matched = current_contract == reference_contract
    if not matched:
        errors.append(
            "当前模型、数据、源码、依赖或 recipe 与首次训练不一致；"
            "拒绝把 checkpoint 接入同一 run。请创建新的 SFT_RUN_ID。"
        )
    return {
        "reference": str(reference_path.resolve()),
        "matched": matched,
        "current": current_contract,
        "expected": reference_contract,
    }


def _recipe_record(args) -> dict:
    effective = {
        "variant": args.recipe_variant,
        "attention_implementation": args.attention_implementation,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "min_gpu_memory_gib": args.min_gpu_memory_gib,
        "min_free_gpu_memory_gib": args.min_free_gpu_memory_gib,
        "min_free_disk_gib": args.min_free_disk_gib,
        "min_kept_ratio": args.min_kept_ratio,
        "gpu_check_skipped": args.skip_gpu_check,
    }
    expected = {
        "variant": "canonical",
        "attention_implementation": "flash_attention_2",
        "max_length": 30000,
        "epochs": 3,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "min_gpu_memory_gib": CANONICAL_MIN_GPU_MEMORY_GIB,
        "min_free_gpu_memory_gib": CANONICAL_MIN_FREE_GPU_MEMORY_GIB,
        "min_free_disk_gib": 50.0,
        "min_kept_ratio": 0.9,
        "gpu_check_skipped": False,
    }
    deviations = {
        key: {"expected": expected[key], "actual": value}
        for key, value in effective.items()
        if value != expected[key]
    }
    return {
        "name": args.recipe_variant,
        "canonical": not deviations,
        "effective": effective,
        "deviations": deviations,
    }


def build_report(args) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    checks = {}
    if args.data_only:
        checks["datasets"] = validate_datasets(args, errors)
    elif args.runtime_only:
        checks["runtime"] = validate_runtime(args, errors, warnings)
    else:
        checks["datasets"] = validate_datasets(args, errors)
        checks["model"] = validate_model(args, errors, warnings)
        checks["runtime"] = validate_runtime(args, errors, warnings)
        if args.tokenize_data:
            checks["tokenization"] = validate_tokenization(args, errors)
    report = {
        "schema_version": PREFLIGHT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "recipe": _recipe_record(args),
        "checks": checks,
        "source": _git_snapshot(),
        "warnings": warnings,
        "errors": errors,
    }
    if args.compare_report:
        report["resume_contract"] = compare_resume_contract(
            report, args.compare_report, errors
        )
        report["status"] = "passed" if not errors else "failed"
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    report = build_report(args)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
