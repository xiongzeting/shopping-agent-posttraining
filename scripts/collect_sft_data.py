#!/usr/bin/env python3
"""Collect resumable Teacher rollouts and build leak-free SFT JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path

from shopping_grpo.collection.sft import (
    acceptance_reasons,
    build_collection_artifacts,
    task_ids_from_jsonl,
)
from shopping_grpo.evaluation.rollout import (
    CollectionInfrastructureError,
    OpenAIChatClient,
    _is_infrastructure_failure,
    append_jsonl,
    collect_for_task,
    collect_tasks,
    completed_task_attempts,
    load_tasks,
    rollout_interrupted,
    SYSTEM_PROMPT,
)
from shopping_grpo.runtime_contract import CONTEXT_WINDOW_TOKENS, GENERATION_RESERVE_TOKENS
from shopping_grpo.local_env import load_project_env


def batch_paths(output_dir: Path) -> dict[str, Path]:
    """Keep raw source data and every reproducible derivative in one directory."""

    return {
        "raw": output_dir / "raw.jsonl",
        "accepted": output_dir / "accepted.jsonl",
        "rejected": output_dir / "rejected.jsonl",
        "stats": output_dir / "reject_stats.json",
        "sft": output_dir / "sft.jsonl",
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "metadata": output_dir / "metadata.json",
    }


def parse_args() -> argparse.Namespace:
    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sft-collection"),
    )
    parser.add_argument(
        "--held-out-tasks",
        type=Path,
        default=Path("data/evaluation/tasks.jsonl"),
        help="These task IDs are never collected or written to SFT outputs.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help=(
            "Local tokenizer/processor used for context accounting when the remote "
            "Teacher endpoint does not provide vLLM /tokenize."
        ),
    )
    parser.add_argument(
        "--tokenizer-json-path",
        type=Path,
        help="Low-memory tokenizers tokenizer.json for conservative context accounting.",
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-accepted", type=int, default=None)
    parser.add_argument("--attempts-per-task", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SHOPSIM_BASE_URL", "http://127.0.0.1:5700"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "SHOPPING_TEACHER_MODEL",
            os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        ),
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get(
            "SHOPPING_TEACHER_BASE_URL",
            os.environ.get("OPENAI_BASE_URL"),
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get(
            "SHOPPING_TEACHER_API_KEY",
            os.environ.get("OPENAI_API_KEY"),
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=GENERATION_RESERVE_TOKENS)
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--reasoning-effort", choices=("high", "max"), default="high")
    parser.add_argument("--context-window", type=int, default=CONTEXT_WINDOW_TOKENS)
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument(
        "--context-compaction",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    return parser.parse_args()


TEACHER_PROMPT_VERSION = "shopping-teacher-prompt-v4-convergence-repair"

TEACHER_BASE_GUIDANCE = """

你正在生成一条供小模型模仿的 SFT Teacher 轨迹。目标不是最少调用工具，也不是刻意增加步骤，而是完成“最短的充分证据轨迹”：任务真正需要的搜索恢复、候选比较、关键证据、规格选择和最终价格确认都不能省略；与用户约束无关、没有信息增益的页面和动作也不能加入。

Teacher 采集规则：
1. 不得假设或猜测隐藏的 gold 商品、ASIN 或规格。只能根据用户需求和当前可见 observation 做决定；采集任务存在严格成功目标也不构成购买证据。
2. 搜索结果只用于发现候选。标题、摘要、排序和价格区间不能替代详情、结构化属性、规格可用性和完整 variant 实际价格的核验。第一个看似匹配的候选不自动等于最佳候选。
3. 每次行动前判断还缺少哪一项决定性证据。商品详情会直接提供非空的 `features` 和 `attributes`，无需进入信息子页；只有会带来新候选、新约束证据、新规格状态或最终价格的动作才值得执行。
4. 轨迹长度必须由任务难度自然产生。容易题证据充分后及时购买；困难题即使步骤已经较多，也不能因为追求短轨迹而跳过必要的搜索改写、候选比较或证据核验。严禁重复查询、无目的往返或访问无关页面凑步数。
5. 工具被本地动作守卫拒绝时，不要重复同一调用。阅读错误和最新 observation，改用当前页面合法动作继续；一次可恢复错误不代表任务失败，也不得故意触发 Guard 来增加多样性。
6. 本批训练任务来自存在可购买目标的隔离任务池。只有经过充分且实质不同的搜索与候选核验，仍没有任何满足品类和预算门槛的可购买候选时，才使用 `finish_without_purchase`；不得为了得到成功样本而错误购买。
7. 购买前必须做一次显式的内部约束清单：品类、预算、品牌/型号、每项核心功能、每个必要规格轴和最终 variant 价格都要分别有当前轨迹证据。任何必要项缺失、只得到相近语义、规格值不完全匹配或最终价格超限时，都不得购买。不要把“浅灰色”当成“灰色”，也不要用缺少脚踏、重力轮、尺寸、套装内容或指定版本的近似 option 替代目标规格。
8. `select_option.value` 必须逐字复制当前 observation 的 `available_options` 中目标 label 对应的稳定 `option_id`；不得填写 label、凭记忆改写 ID 或使用历史页面 ID。若目标 option 当前不可见，先用合法的返回、打开候选或重新搜索动作恢复页面状态，绝不能重复提交已被 Guard 拒绝的参数。
9. 搜索与比较必须收敛。每次搜索、返回或重开候选前，先判断它会新增哪项证据；若最近两次探索没有新增候选、约束证据或规格状态，必须改变检索维度、核验另一个真实候选，或在确认没有可接受商品时调用 `finish_without_purchase`。该动作统一按 `early_abstain` 终止，不会因检索充分而获得单独的停止类型。不得在同一组查询、候选和页面之间循环。
10. 不得用普通 assistant 文本结束轨迹。找到完全满足且证据闭合的候选后必须调用 `buy_now`；经过充分探索确认没有可接受候选时必须调用 `finish_without_purchase`。分析文字不能替代终局工具调用，也不得在已经能够购买时继续说“再看看其他候选”。
"""


TEACHER_REPAIR_GUIDANCE = """

本批轨迹重点修复“检索后无法收敛、接受近似商品、分析后不调用终局工具”三类行为：
11. 把每次搜索和候选核验视为消除一项不确定性。若查询、ASIN、页面和已知证据没有变化，不得重新执行同一分支；应改变检索维度、换一个真实候选或根据当前证据收敛决策。不要通过重复搜索、重复打开和无目的返回来延长轨迹。
12. 对看似接近的候选严格区分“完全满足”和“部分替代”。颜色近似、缺少任一核心功能、型号或落地形式不同、尺寸或材质不完全一致、套餐内容缺失、variant 价格超限，都属于硬约束失败，必须放弃该候选，不能用整体相似度覆盖单项失败。
13. 购买前在内部逐项对齐“用户要求 → 当前商品证据 → 当前已选 option → 完整 variant 实际价格”。只有每项硬约束均有当前轨迹证据时才购买；不得凭标题暗示、历史页面记忆或自己改写的 option 值补全证据。
14. 一旦完全匹配候选、必要规格和最终价格已经闭合，下一次 assistant 行为必须直接调用 `buy_now`。不得继续搜索、继续比较或输出推荐性自然语言。若当前动作不合法，则只依据最新 observation 恢复到合法页面或合法 option，再尽快完成终局工具调用。
15. 可以保留自然出现的弱搜索、近似候选或短暂错误方向及其成功修复，以展示如何恢复；但不得故意制造非法调用、重复循环或固定步数模板。最终轨迹必须干净、收敛并完成 gold purchase。
"""


TEACHER_STRATEGY_GUIDANCE = {
    "loop_recovery": """

本题优先展示自然的检索收敛：弱查询或近似候选没有补齐硬约束时，明确指出缺失证据，改变一个有意义的检索维度并转向新候选。不得再次打开相同候选或复用等价查询；找到证据闭合的目标后立即停止探索并购买。
""",
    "near_miss_rejection": """

本题优先展示近似候选拒绝。若存在多个可信候选，应逐项核验硬约束，明确淘汰至少一个缺少功能、规格、variant、颜色、材质、形式或预算条件的近似商品，再购买完整满足的候选。不能为了完成任务而放宽任何硬约束。
""",
    "terminal_tool_commit": """

本题优先训练决策到工具动作的转换。证据和 option 闭合前认真核验；闭合后不再输出解释性自然语言、不再继续搜索，下一步直接调用 `buy_now` 并以环境终局结束。
""",
    "option_grounding": """

本题优先训练规格与 option 的精确落地。只选择当前 observation 明确可见的完整 option 值，逐轴形成最终 variant，并用该 variant 的实际价格判断预算。不得自己缩写、改写、拼接或猜测规格名称。
""",
    "focused_verification": """

本题采用聚焦核验策略。先识别决定购买的品类、预算、品牌/型号、核心功能和规格约束，只为尚未被当前 observation 证明的约束选择最直接的信息面。若一个候选已被可靠证据完整证明，应及时完成规格和购买；不得机械地同时查看 Features、Description、Reviews 和 Attributes，也不得为了形成固定流程访问无关页面。""",
    "search_reformulation": """

本题由任务分析器标记为检索恢复型。第一次搜索只用于判断候选空间；若结果没有同时体现正确品类和关键区分约束，或高位结果含糊、重复、类别偏离，不得直接从这些结果购买，必须使用实质不同的查询重新搜索。新查询应改变品牌、型号、核心功能、规格或品类表达中的至少一项，不能只调整词序或做同义重复。若新结果仍无可靠候选，可继续缩短查询、换核心约束或翻页；一旦出现证据充分的最佳候选就停止改写。""",
    "candidate_comparison": """

本题由任务分析器标记为候选混淆型。不要在打开第一个看似符合的商品后立即购买。搜索结果中存在两个或更多真实可行候选时，必须打开至少两个不同商品，分别核验决定性约束、规格可用性和实际价格，再选择整体最符合需求的一个。若当前结果只有一个可行候选，应通过一次实质不同的搜索或翻页确认候选空间，而不是拿明显无关商品凑比较；确认没有第二个合理候选后才可回到最佳候选。""",
    "evidence_verification": """

本题由任务分析器标记为证据核验型。标题和搜索摘要不能作为品牌、型号、材质、功能或兼容性的最终证据。针对每一项会改变购买结论的未决约束，选择最权威的信息面：结构化字段、Description、Features 或 Attributes；Reviews 只能补充体验，不得证明官方规格。不同页面证据冲突、表述含糊或只能证明部分约束时，必须继续核验或比较其他候选；证据已经闭合后不要访问无关页面。""",
    "price_semantics": """

本题由任务分析器标记为价格语义型。明确区分预算上限、价格区间、约数、起售价和具体 variant 的最终价格。搜索页价格和未完成规格时的价格只能用于初筛，不能证明最终可购买价格；完成所有必要规格选择后必须重新确认完整 variant 的实际价格。若最佳语义匹配项超预算，应比较预算内替代项，绝不能通过忽略规格溢价完成购买。""",
    "multi_option": """

本题由任务分析器标记为多规格组合型。识别用户要求涉及的每个必要规格轴，逐轴选择当前页面明确可用且语义准确的 option；一次 `select_option` 不代表组合已经完成。形成完整 variant 后，重新检查可用性和最终价格。若任一必要规格缺失、不可用、语义不符或导致超预算，应返回搜索并比较其他候选，不得用相近但错误的规格完成购买。""",
}


def collection_system_prompt(teacher_strategy: str | None = None) -> str:
    if teacher_strategy is not None and teacher_strategy not in TEACHER_STRATEGY_GUIDANCE:
        raise ValueError(f"unknown Teacher strategy: {teacher_strategy}")
    strategy_suffix = TEACHER_STRATEGY_GUIDANCE.get(teacher_strategy or "", "")
    return (
        SYSTEM_PROMPT.rstrip()
        + TEACHER_BASE_GUIDANCE
        + TEACHER_REPAIR_GUIDANCE
        + strategy_suffix
    )


def teacher_prompt_manifest() -> dict:
    """Return hashes for every exact prompt variant used by the collector."""

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    return {
        "schema_version": TEACHER_PROMPT_VERSION,
        "base_sha256": digest(collection_system_prompt()),
        "strategy_sha256": {
            strategy: digest(collection_system_prompt(strategy))
            for strategy in sorted(TEACHER_STRATEGY_GUIDANCE)
        },
    }


def collect_until_target(
    *,
    tasks,
    target_accepted,
    client,
    client_factory=None,
    output_path,
    base_url,
    max_steps,
    attempts_per_task,
    workers=1,
    excluded_task_ids=(),
    system_prompt=None,
    system_prompt_factory=None,
    evaluation_extensions=True,
):
    """Collect concurrently without scheduling more possible successes than needed."""

    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    accepted_task_ids = _accepted_task_ids(output_path, excluded_task_ids)
    accepted = len(accepted_task_ids)
    completed = completed_task_attempts(output_path)
    candidates = [
        (task, attempt_index)
        for attempt_index in range(int(attempts_per_task))
        for task in tasks
        if (int(task["task_id"]), attempt_index) not in completed
    ]
    candidate_iter = iter(candidates)
    pending = {}
    written = []
    infrastructure_failed = False

    def submit_available(executor):
        remaining = int(target_accepted) - accepted
        max_pending = min(workers, max(remaining, 0))
        while len(pending) < max_pending:
            try:
                task, attempt_index = next(candidate_iter)
            except StopIteration:
                return
            if int(task["task_id"]) in accepted_task_ids:
                continue
            task_system_prompt = (
                system_prompt_factory(task)
                if system_prompt_factory is not None
                else system_prompt or collection_system_prompt()
            )
            future = executor.submit(
                collect_for_task,
                task,
                client=client_factory() if client_factory else client,
                base_url=base_url,
                max_steps=max_steps,
                attempt_index=attempt_index,
                system_prompt=task_system_prompt,
                evaluation_extensions=evaluation_extensions,
            )
            pending[future] = (task, attempt_index)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        submit_available(executor)
        while pending:
            completed_futures, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed_futures:
                pending.pop(future)
                trajectory = future.result()
                append_jsonl(output_path, [trajectory])
                written.append(trajectory)
                accepted_now = bool(acceptance_reasons(trajectory)[0])
                if accepted_now:
                    accepted_task_ids.add(int(trajectory["task_id"]))
                    accepted = len(accepted_task_ids)
                infrastructure_failed |= _is_infrastructure_failure(trajectory)
            if not infrastructure_failed:
                submit_available(executor)

    if infrastructure_failed:
        raise CollectionInfrastructureError(
            "collection infrastructure failure; stopped before the next task"
        )
    return written, accepted


def _accepted_count(raw_path: Path, excluded_task_ids=()) -> int:
    return len(_accepted_task_ids(raw_path, excluded_task_ids))


def _accepted_task_ids(raw_path: Path, excluded_task_ids=()) -> set[int]:
    raw_path = Path(raw_path)
    if not raw_path.exists():
        return set()
    excluded = {int(task_id) for task_id in excluded_task_ids}
    with raw_path.open(encoding="utf-8") as handle:
        accepted = set()
        for line in handle:
            if not line.strip():
                continue
            trajectory = json.loads(line)
            if int(trajectory["task_id"]) in excluded:
                continue
            if acceptance_reasons(trajectory)[0]:
                accepted.add(int(trajectory["task_id"]))
        return accepted


def _validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.target_accepted is not None and args.target_accepted < 1:
        raise SystemExit("--target-accepted must be at least 1")
    if args.attempts_per_task < 1:
        raise SystemExit("--attempts-per-task must be at least 1")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.workers > 1 and args.target_accepted is None:
        raise SystemExit("--workers > 1 requires --target-accepted")
    if args.tokenizer_path is not None and args.tokenizer_json_path is not None:
        raise SystemExit("--tokenizer-path and --tokenizer-json-path are mutually exclusive")
    if not 0 <= args.validation_ratio < 1:
        raise SystemExit("--validation-ratio must be in [0, 1)")
    if not args.build_only and not args.llm_base_url:
        raise SystemExit("--llm-base-url or SHOPPING_TEACHER_BASE_URL is required")
    if not args.build_only and not args.api_key:
        raise SystemExit("--api-key or SHOPPING_TEACHER_API_KEY is required")
    if not args.build_only and args.tasks is None:
        raise SystemExit("--tasks is required unless --build-only is used")


@lru_cache(maxsize=4)
def _local_token_counters(path: str):
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    load_kwargs = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(path, **load_kwargs)
    if getattr(config, "model_type", "") == "qwen3_5":
        renderer = AutoProcessor.from_pretrained(path, **load_kwargs)
        tokenizer = renderer.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, **load_kwargs)
        renderer = tokenizer

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
        return len(tokenizer(str(text), add_special_tokens=False)["input_ids"])

    return chat_counter, text_counter


@lru_cache(maxsize=4)
def _lightweight_token_counters(path: str):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(path)
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


def _make_client(args: argparse.Namespace) -> OpenAIChatClient:
    token_counter = None
    observation_token_counter = None
    if args.tokenizer_path is not None:
        token_counter, observation_token_counter = _local_token_counters(
            str(args.tokenizer_path)
        )
    elif args.tokenizer_json_path is not None:
        token_counter, observation_token_counter = _lightweight_token_counters(
            str(args.tokenizer_json_path)
        )
    return OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        context_window=args.context_window or None,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction,
        observation_token_budget=args.observation_token_budget or None,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_candidate_memory_token_budget=(
            args.observation_candidate_memory_token_budget
        ),
        observation_search_top_k=args.observation_search_top_k,
        token_counter=token_counter,
        observation_token_counter=observation_token_counter,
    )


def _collection_config(args: argparse.Namespace) -> dict:
    """Record reproducibility settings without ever serializing the API key."""

    return {
        "tasks": str(args.tasks),
        "held_out_tasks": str(args.held_out_tasks),
        "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
        "tokenizer_json_path": (
            str(args.tokenizer_json_path) if args.tokenizer_json_path else None
        ),
        "model": args.model,
        "llm_base_url": args.llm_base_url,
        "shopsim_base_url": args.base_url,
        "limit": args.limit,
        "target_accepted": args.target_accepted,
        "attempts_per_task": args.attempts_per_task,
        "workers": args.workers,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
        "max_steps": args.max_steps,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort,
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
        "teacher_prompt": teacher_prompt_manifest(),
    }


def main() -> int:
    args = parse_args()
    _validate_args(args)
    paths = batch_paths(args.output_dir)
    held_out_ids = task_ids_from_jsonl(args.held_out_tasks)
    exit_code = 0
    collection_config = None

    if not args.build_only:
        collection_config = _collection_config(args)
        signal.signal(signal.SIGTERM, rollout_interrupted)
        signal.signal(signal.SIGINT, rollout_interrupted)
        tasks = [
            task for task in load_tasks(args.tasks) if int(task["task_id"]) not in held_out_ids
        ]
        if args.limit is not None:
            tasks = tasks[: args.limit]
        try:
            if args.target_accepted is None:
                client = _make_client(args)
                written = collect_tasks(
                    tasks,
                    client=client,
                    output_path=paths["raw"],
                    base_url=args.base_url,
                    max_steps=args.max_steps,
                    attempts_per_task=args.attempts_per_task,
                    system_prompt_factory=lambda task: collection_system_prompt(
                        task.get("teacher_strategy")
                    ),
                    evaluation_extensions=True,
                )
                print(f"collected_raw={len(written)}")
            else:
                written, accepted = collect_until_target(
                    tasks=tasks,
                    target_accepted=args.target_accepted,
                    client=None,
                    client_factory=lambda: _make_client(args),
                    output_path=paths["raw"],
                    base_url=args.base_url,
                    max_steps=args.max_steps,
                    attempts_per_task=args.attempts_per_task,
                    workers=args.workers,
                    excluded_task_ids=held_out_ids,
                    system_prompt_factory=lambda task: collection_system_prompt(
                        task.get("teacher_strategy")
                    ),
                    evaluation_extensions=True,
                )
                print(f"collected_raw={len(written)} accepted_total={accepted}")
        except CollectionInfrastructureError as exc:
            print(f"collection paused: {exc}")
            exit_code = 2

    if not paths["raw"].exists():
        raise SystemExit(f"raw trajectory file does not exist: {paths['raw']}")
    summary = build_collection_artifacts(
        raw_path=paths["raw"],
        output_dir=args.output_dir,
        held_out_task_ids=held_out_ids,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        collection_config=collection_config,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
