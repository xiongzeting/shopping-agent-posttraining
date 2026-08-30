#!/usr/bin/env python3
"""在固定 ShopSimulator benchmark 上评测 OpenAI-compatible 本地或远端模型。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

from shopping_grpo.environment.manifest import validate_manifest, validate_runtime_files
from shopping_grpo.evaluation.blind_guard import (
    guard_blind_final,
    validate_canonical_benchmark_files,
)
from shopping_grpo.evaluation.artifacts import write_json_atomic, write_jsonl_atomic
from shopping_grpo.evaluation.manifest import build_run_manifest, sha256_file
from shopping_grpo.evaluation.pipeline import evaluate_trajectories
from shopping_grpo.evaluation.rollout import (
    EVALUATION_TERMINATION_VERSION,
    EVALUATION_TOOL_SCHEMAS,
    EVALUATION_TOOL_VERSION,
    SYSTEM_PROMPT,
    MultiKeyOpenAIChatClient,
    OpenAIChatClient,
    collect_tasks,
    load_tasks,
)
from shopping_grpo.local_env import load_project_env
from shopping_grpo.runtime_contract import (
    CONTEXT_WINDOW_TOKENS,
    GENERATION_RESERVE_TOKENS,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="评测 Base、SFT 或 GRPO Shopping Agent")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--task-subset",
        type=Path,
        help="Run only this JSONL subset after validating the complete canonical Benchmark v2.2.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="统一保存轨迹、四面板结果和运行清单的目录",
    )
    parser.add_argument(
        "--benchmark-metadata",
        type=Path,
        default=ROOT / "data/evaluation/metadata.json",
    )
    parser.add_argument(
        "--benchmark-slices",
        type=Path,
        default=ROOT / "data/evaluation/slices.jsonl",
    )
    parser.add_argument(
        "--environment-manifest",
        type=Path,
        default=ROOT / "data/environment.json",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--model", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument(
        "--api-key",
        help="默认读取 SHOPPING_TEACHER_API_KEY 或 OPENAI_API_KEY；本地 vLLM 可传 EMPTY",
    )
    parser.add_argument("--actor-label", help="报告中的模型标签；默认使用 --model")
    parser.add_argument("--rubrics", type=Path, help="可选的冻结 Rubric JSONL")
    parser.add_argument("--judges", type=Path, help="可选的离线 Judge JSONL")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="本地 tokenizer/processor 路径；用于不提供 vLLM /tokenize 的推理服务。",
    )
    parser.add_argument(
        "--tokenizer-json-path",
        type=Path,
        help="Low-memory tokenizers tokenizer.json for conservative local counting.",
    )
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=GENERATION_RESERVE_TOKENS,
        help="单次模型生成上限；防止未调用工具时耗尽完整上下文。",
    )
    parser.add_argument("--context-window", type=int, default=CONTEXT_WINDOW_TOKENS)
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument(
        "--context-compaction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="上下文接近上限时压缩较早的完整交互组；Harness v3 默认开启。",
    )
    parser.add_argument("--observation-token-budget", type=int, default=2560)
    parser.add_argument("--observation-detail-token-budget", type=int, default=3072)
    parser.add_argument("--observation-generic-token-budget", type=int, default=512)
    parser.add_argument(
        "--observation-candidate-memory-token-budget",
        type=int,
        default=1024,
    )
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _json_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _local_token_counters(path):
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    load_kwargs = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(path, **load_kwargs)
    if getattr(config, "model_type", "") == "qwen3_5":
        try:
            renderer = AutoProcessor.from_pretrained(path, **load_kwargs)
            tokenizer = renderer.tokenizer
            if not getattr(renderer, "chat_template", None):
                renderer = tokenizer
        except ImportError:
            # Final-240 is text-only.  A lightweight evaluator should not need
            # torch/torchvision merely because Qwen3.5 also ships a vision
            # processor; its tokenizer owns the same text chat template.
            tokenizer = AutoTokenizer.from_pretrained(path, **load_kwargs)
            renderer = tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, **load_kwargs)
        renderer = tokenizer
    tokenizer_lock = threading.RLock()

    def chat_counter(messages, tools):
        normalized_messages = json.loads(json.dumps(messages, ensure_ascii=False))
        for message in normalized_messages:
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    parsed_arguments = json.loads(arguments)
                    function["arguments"] = (
                        parsed_arguments if isinstance(parsed_arguments, dict) else {}
                    )
        with tokenizer_lock:
            token_ids = renderer.apply_chat_template(
                normalized_messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        if hasattr(token_ids, "input_ids"):
            token_ids = token_ids.input_ids
        elif isinstance(token_ids, dict):
            token_ids = token_ids["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return len(token_ids)

    def text_counter(text):
        with tokenizer_lock:
            return len(tokenizer(str(text), add_special_tokens=False)["input_ids"])

    return chat_counter, text_counter


def _lightweight_token_counters(path):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer_lock = threading.RLock()

    def text_counter(text):
        with tokenizer_lock:
            return len(tokenizer.encode(str(text)).ids)

    def chat_counter(messages, tools):
        payload = json.dumps(
            {"messages": messages, "tools": tools},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return text_counter(payload)

    return chat_counter, text_counter


def main():
    load_project_env()
    args = parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps 必须为正数")
    if args.workers < 1:
        raise SystemExit("--workers 必须为正数")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens 必须为正数")
    if args.context_window <= args.max_tokens + args.context_safety_margin:
        raise SystemExit("--context-window 必须大于 --max-tokens 与安全余量之和")
    if args.tokenizer_path is not None and args.tokenizer_json_path is not None:
        raise SystemExit("--tokenizer-path and --tokenizer-json-path are mutually exclusive")
    api_key = (
        args.api_key
        or os.environ.get("SHOPPING_TEACHER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    api_keys_json = os.environ.get("SHOPPING_TEACHER_API_KEYS")
    api_keys = []
    if api_keys_json:
        try:
            api_keys = [str(value) for value in json.loads(api_keys_json) if str(value)]
        except (TypeError, json.JSONDecodeError) as exc:
            raise SystemExit("SHOPPING_TEACHER_API_KEYS must be a JSON string array") from exc
    if not api_key and not api_keys:
        raise SystemExit(
            "--api-key、SHOPPING_TEACHER_API_KEY 或 OPENAI_API_KEY 至少提供一个"
        )
    # Always validate the complete frozen benchmark before a formal rollout.
    # A task subset changes only which validated rows are executed; it must not
    # turn off the Final-240 hash, schema, blind-ID, or slice-coverage checks.
    try:
        benchmark_metadata, task_slices = validate_canonical_benchmark_files(
            tasks_path=args.benchmark,
            metadata_path=args.benchmark_metadata,
            slices_path=args.benchmark_slices,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Final-240 benchmark 校验失败：{exc}") from exc
    try:
        environment_manifest = validate_manifest(
            json.loads(args.environment_manifest.read_text(encoding="utf-8"))
        )
        validate_runtime_files(environment_manifest, ROOT)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Final-240 环境清单校验失败：{exc}") from exc
    if environment_manifest["environment_version"] != benchmark_metadata["environment"]:
        raise SystemExit("Final-240 benchmark 与环境清单版本不一致")
    if environment_manifest["reward"]["version"] != benchmark_metadata["reward"]:
        raise SystemExit("Final-240 benchmark 与 Reward 版本不一致")

    canonical_tasks = load_tasks(args.benchmark)
    canonical_task_ids = {int(task["task_id"]) for task in canonical_tasks}
    tasks = load_tasks(args.task_subset) if args.task_subset else canonical_tasks
    task_ids = [int(task["task_id"]) for task in tasks]
    if not task_ids:
        raise SystemExit("--task-subset must contain at least one task")
    if len(task_ids) != len(set(task_ids)):
        raise SystemExit("--task-subset contains duplicate task IDs")
    active_task_slices = (
        {task_id: task_slices[task_id] for task_id in task_ids}
        if all(task_id in task_slices for task_id in task_ids)
        else None
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = args.run_dir / "trajectories.jsonl"
    token_counter = None
    observation_token_counter = None
    if args.tokenizer_path is not None:
        token_counter, observation_token_counter = _local_token_counters(args.tokenizer_path)
    elif args.tokenizer_json_path is not None:
        token_counter, observation_token_counter = _lightweight_token_counters(
            args.tokenizer_json_path
        )
    tool_choice = os.environ.get("SHOPPING_TOOL_CHOICE", "auto")
    missing_tool_call_retries = int(
        os.environ.get("SHOPPING_MISSING_TOOL_CALL_RETRIES", "0")
    )
    client_kwargs = dict(
        model=args.model,
        base_url=args.llm_base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction,
        observation_token_budget=args.observation_token_budget,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_candidate_memory_token_budget=(
            args.observation_candidate_memory_token_budget
        ),
        observation_search_top_k=args.observation_search_top_k,
        seed=args.seed,
        tool_choice=tool_choice,
        missing_tool_call_retries=missing_tool_call_retries,
        token_counter=token_counter,
        observation_token_counter=observation_token_counter,
    )
    if len(api_keys) > 1:
        per_key_concurrency = int(os.environ.get("SHOPPING_API_KEY_CONCURRENCY", "10"))
        if args.workers > len(api_keys) * per_key_concurrency:
            raise SystemExit("workers exceed SHOPPING_TEACHER_API_KEYS capacity")
        client = MultiKeyOpenAIChatClient(
            api_keys=api_keys,
            per_key_concurrency=per_key_concurrency,
            client_kwargs=client_kwargs,
        )
    else:
        client = OpenAIChatClient(
            api_key=(api_keys[0] if api_keys else api_key),
            **client_kwargs,
        )
    collect_tasks(
        tasks,
        client=client,
        output_path=trajectories_path,
        base_url=args.base_url,
        max_steps=args.max_steps,
        workers=args.workers,
        evaluation_extensions=True,
    )
    rubric_rows = _read_jsonl(args.rubrics) if args.rubrics else []
    judge_rows = _read_jsonl(args.judges) if args.judges else []
    actor = {
        "label": args.actor_label or args.model,
        "model": args.model,
        "tokenizer": (
            args.tokenizer_path.name
            if args.tokenizer_path
            else args.tokenizer_json_path.name if args.tokenizer_json_path else None
        ),
    }
    artifacts = evaluate_trajectories(
        expected_task_ids=task_ids,
        trajectories=_read_jsonl(trajectories_path),
        actor=actor,
        task_slices=active_task_slices,
        rubric_bundles=rubric_rows,
        judge_results=judge_rows,
    )
    artifact_paths = {
        "evaluations": args.run_dir / "evaluations.jsonl",
        "summary": args.run_dir / "summary.json",
    }
    write_jsonl_atomic(artifact_paths["evaluations"], artifacts["evaluations"])
    write_json_atomic(artifact_paths["summary"], artifacts["summary"])

    output_manifest = {
        "trajectories.jsonl": sha256_file(trajectories_path),
        **{path.name: sha256_file(path) for path in artifact_paths.values()},
    }
    run_manifest = build_run_manifest(
        run_id=args.run_dir.name,
        actor=actor,
        task_manifest={
            "benchmark": "ShopBench-LH Final-240 / Benchmark v2.2",
            "tasks": len(task_ids),
            "canonical_tasks": benchmark_metadata["tasks"],
            "task_sha256": benchmark_metadata["task_sha256"],
            "slice_sha256": benchmark_metadata["slice_sha256"],
            "source_pool": benchmark_metadata["source_pool"],
            "subset": args.task_subset is not None,
            "subset_sha256": sha256_file(args.task_subset) if args.task_subset else None,
        },
        environment={
            "manifest_sha256": sha256_file(args.environment_manifest),
            "environment": environment_manifest["environment_version"],
            "reward": environment_manifest["reward"]["version"],
            "termination": EVALUATION_TERMINATION_VERSION,
            "observation": benchmark_metadata["observation"],
            "tool_schema": EVALUATION_TOOL_VERSION,
        },
        protocol={
            "max_steps": args.max_steps,
            "workers": args.workers,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "context_window": args.context_window,
            "context_safety_margin": args.context_safety_margin,
            "context_compaction": args.context_compaction,
            "observation_token_budget": args.observation_token_budget,
            "observation_detail_token_budget": args.observation_detail_token_budget,
            "observation_generic_token_budget": args.observation_generic_token_budget,
            "observation_candidate_memory_token_budget": (
                args.observation_candidate_memory_token_budget
            ),
            "observation_search_top_k": args.observation_search_top_k,
            "missing_tasks_count_as_failures": True,
        },
        code={
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "tool_schema_sha256": _json_sha256(EVALUATION_TOOL_SCHEMAS),
        },
        judge={
            "enabled": bool(args.judges),
            "rubric_source_sha256": sha256_file(args.rubrics) if args.rubrics else None,
            "judge_source_sha256": sha256_file(args.judges) if args.judges else None,
            "role": "diagnostic_only",
        },
        outputs=output_manifest,
    )
    write_json_atomic(args.run_dir / "run_manifest.json", run_manifest)
    print(json.dumps(artifacts["summary"], ensure_ascii=True))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    main()
