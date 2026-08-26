"""使用 OpenAI-compatible 工具调用运行 Shopping Agent 评测轨迹。

这里负责把模型回复、工具守卫和 ShopSimulator 串成一条可回放记录；它是评测采集
入口，不负责修改环境仓库或训练模型。
"""

import json
import os
import queue
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http.client import IncompleteRead, RemoteDisconnected
from urllib.error import URLError
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

from shopping_grpo.environment.actions import (
    action_guard_tool_message,
    action_reject_reason,
    clickable_buttons,
    product_ids,
    resolve_action_parameters,
    selected_option_ids,
)
from shopping_grpo.environment.candidate_memory import new_candidate_memory
from shopping_grpo.environment.context import (
    ContextBudgetError,
    VllmChatTokenCounter,
    VllmTextTokenCounter,
    compact_chat_messages,
)
from shopping_grpo.environment.projection import project_observation
from shopping_grpo.environment.client import ShopAgentEnv, ShopEnvironmentError, ShopHttpError
from shopping_grpo.environment.product_id import is_product_id
from shopping_grpo.environment.tools import (
    SHOP_TOOL_SCHEMAS,
    tool_call_to_action,
)
from shopping_grpo.environment.observation import (
    add_step_budget_notice,
    render_structured_observation,
)
from shopping_grpo.training.grpo.adapter.runtime import calculate_step_penalty


SYSTEM_PROMPT = """你是一个购物 Agent，负责在 ShopSimulator 中替用户完成一次单轮购物任务。

用户的完整需求只会在开头给出。不得向用户追问、确认、告别，也不要假设存在用户对话工具。你只能调用提供的标准工具与商店交互。目标是在有限45步内按以下优先级完成任务：优先购买满足全部要求的商品；经过有效探索仍没有全满足商品时，才可在所有硬约束均满足的候选中选择整体最符合软偏好的商品；若找不到满足全部硬约束且可接受的商品，应合理结束，绝不能违反硬约束错误购买。硬约束包括品类，以及用户明确表示必须满足、不可违反或未被软化的明确条件；无效循环是指重复相同或无实质变化的搜索、反复打开或核验相同商品、在相同页面间往返，或重复被拒绝的动作，却没有获得新候选、新商品信息、新规格选择或新的需求证据。

执行规则：
1. 动作合法性与历史比较。每轮只提供最新页面真实可执行的工具；未出现在本轮列表中的工具当前不可用，不得猜测或强行调用。搜索首页只提供搜索和合理结束；搜索结果页只提供打开当前商品、当前可见导航和合理结束；商品详情页只提供当前未选规格、当前可见购买/返回和合理结束。普通页面按钮、ASIN 和规格只能依据最新 observation；历史 observation 只用于比较商品，不代表当前可点击目标。环境结束前，每个 assistant 回合必须立即且仅返回一个工具调用；禁止输出自然语言、分析、解释、需求复述或候选清单，选定下一动作后直接调用工具。
2. 页面状态、候选记忆与参数。打开商品后，直接依据最新商品详情 observation 中展示的标题、品牌、品类、实际价格、商品属性、规格状态和当前已选/可选规格，自行核验当前完整 variant；规格变化后必须重新检查实际价格。Harness 会在内部最多保存 4 个已核验候选，普通阶段不会展示或开放这些历史候选；只有系统明确进入“6步无进展候选收敛阶段”时，候选才会临时显示为同一页可打开商品，此时必须用 `open_product` 从该页选择一个候选，随后严格按当轮仅暴露的规格或终局工具完成决策。无参数工具（翻页、返回、购买）必须传严格的 `{}`；`search_products` 使用 `query`、`open_product` 使用当前页 `asin`、`select_option` 使用最新 `available_options` 中目标 label 对应的稳定 `option_id`。不要重复选择已经在 `selected_options` 中的 option_id。
3. 搜索与候选探索。查询应简洁，优先使用品类和最有区分度的品牌、型号、核心功能或规格，不要机械复制整段需求。结果不理想时，缩短查询、更换真正不同的关键词或翻页；出现有希望的商品时应打开核验。不要重复相同查询，也不要只做同义改写却反复得到相同候选。
4. 硬约束、软偏好与候选比较。必须按用户原话区分不可违反的硬约束（Hard）与可折中的软偏好（Soft）。品类按商品的实际用途判断，不要求与平台目录叶子节点逐字一致，但品类始终是 Hard。“必须、务必、一定要、仅限”等明确必需表达，以及“绝对不要、不能带、禁止、不得有、不要带有”等明确排除表达属于 Hard；“不需要、不用、无需、不要求”通常表示无偏好，不等于禁止。未被软化的明确价格上限、下限或区间，以及直接提出且没有软化措辞的品牌、型号、核心功能和规格要求也按 Hard 处理，任何一项可核验 Hard 不满足都不得购买。“最好、优先、尽量、希望、偏好、左右、大约、大概、接近、预算、不要太”等属于 Soft；其中“约、左右、大概、预算”不是最低价格。价格必须按完整 variant 的实际价格和用户原话判断；近似价格按目标价上下20%的范围理解。“可搭配X”表示兼容能力，不等于排除需要X的商品。常见单位必须正确换算，例如 1kg=2斤、25kg=50斤、10斤=5kg。比较候选时先保证全部 Hard，再尽量满足 Soft；一份直接、结构化且与当前 variant 对应的证据已经足够，不得为同一条件寻找第二份证明。
5. 购买决策。打开商品并完成必要规格选择后，立即依据当前详情页判断。品类不符或任一可核验 Hard 不满足时不得购买；规格轴已经闭合且全部要求满足、最新 observation 显示 `Buy Now` 时，下一次 assistant 行为必须直接调用 `buy_now`，不得继续搜索、比较、重复核验或输出文字。尚有 Hard 不满足时，立刻离开并转向其他候选。充分探索后仍无完全匹配商品时，可以购买全部 Hard 满足、在已核验候选中整体最符合 Soft 且可接受的候选；若不存在这样的候选，才调用 `finish_without_purchase`。判断只能使用用户需求和模型可见公开信息。
6. 主动结束。经过多次有实质差异的搜索和多个候选核验，仍没有可接受商品，并且当前没有明显值得继续核验的候选时，调用 `finish_without_purchase`。不得过早结束，也不要为了增加搜索次数继续无效探索。
7. 防止循环和非法动作。不要连续重复同一动作，也不要在相同结果、商品或子页之间无目的往返；后续操作应带来新候选、新商品信息、新规格选择或新的需求证据。如需改写搜索词，必须实质改变品类、品牌、型号、核心功能或规格中的至少一项，不得只做同义替换、语序调整或增删虚词。不要调用 `think` 工具。若 tool 返回“本地动作守卫拒绝，未执行”，依据错误消息和最新 observation 改为一个合法动作，不要重复被拒绝的调用。不要在任务结束前输出最终答复或推荐总结；只有环境报告任务结束后才停止。
"""


EVALUATION_TOOL_SCHEMAS = list(SHOP_TOOL_SCHEMAS)
EVALUATION_TOOL_VERSION = "shopping-evaluation-tools-v2.3"
EVALUATION_TERMINATION_VERSION = "shopping-termination-v3.2"

CANDIDATE_PHASE_CHOOSE = "choose_candidate"
CANDIDATE_PHASE_SELECT = "select_option"
CANDIDATE_PHASE_TERMINAL = "terminal_decision"


MAX_BLOCKED_TOOL_CALLS = 3
MAX_MISSING_TOOL_CALL_CORRECTIONS = 2
MISSING_TOOL_CALL_CORRECTION = (
    "上一回复未包含工具调用，因此未执行。"
    "禁止继续输出分析；现在必须且只能调用一个合法工具。"
)
MODEL_COMPLETION_RETRIES = 2
MODEL_RETRY_DELAY_SECONDS = 1
ASSISTANT_FINAL_REWARD = -0.80
GUARD_REJECTION_REWARD = -0.80
REPEAT_LOOP_REWARD = -0.60


def _assistant_final_result(step_count=0):
    step_count = max(0, int(step_count))
    step_penalty = calculate_step_penalty(step_count)
    reward = round(float(ASSISTANT_FINAL_REWARD) + step_penalty, 10)
    return {
        "done": True,
        "over": True,
        "reward": reward,
        "termination_reason": "assistant_final",
        "reward_valid": True,
        "reward_detail": {
            "reward_version": "shopsimulator-reward-v4",
            "query_constraint_version": "shopping-query-constraints-v1",
            "reward_type": "assistant_final",
            "reward_valid": True,
            "termination_reason": "assistant_final",
            "termination_subreason": "assistant_final",
            "target_asin_match": False,
            "hard_gates": {},
            "weighted_score": 0.0,
            "evidence_coverage": 0.0,
            "dimension_scores": {},
            "constraint_results": [],
            "terminal_utility": reward,
            "base_terminal_utility": ASSISTANT_FINAL_REWARD,
            "step_count": step_count,
            "step_penalty": step_penalty,
            "step_penalty_version": "shopping-step-penalty-v1",
            "purchase_success": False,
            "sampling_invalid": False,
        },
    }


def _guard_rejection_result(step_count=0):
    step_count = max(0, int(step_count))
    step_penalty = calculate_step_penalty(step_count)
    reward = round(float(GUARD_REJECTION_REWARD) + step_penalty, 10)
    return {
        "done": True,
        "over": True,
        "reward": reward,
        "termination_reason": "invalid_action_limit",
        "reward_valid": True,
        "reward_detail": {
            "reward_version": "shopsimulator-reward-v4",
            "query_constraint_version": "shopping-query-constraints-v1",
            "reward_type": "guard_rejection",
            "reward_valid": True,
            "termination_reason": "invalid_action_limit",
            "termination_subreason": "too_many_guard_rejections",
            "target_asin_match": False,
            "hard_gates": {},
            "weighted_score": 0.0,
            "evidence_coverage": 0.0,
            "dimension_scores": {},
            "constraint_results": [],
            "terminal_utility": reward,
            "base_terminal_utility": GUARD_REJECTION_REWARD,
            "step_count": step_count,
            "step_penalty": step_penalty,
            "step_penalty_version": "shopping-step-penalty-v1",
            "purchase_success": False,
            "sampling_invalid": False,
        },
    }


def _repeat_loop_result(step_count=0, *, subreason="no_progress_loop"):
    """Build the unchanged Reward-v4 Loop terminal used by forced recovery."""
    step_count = max(0, int(step_count))
    step_penalty = calculate_step_penalty(step_count)
    reward = round(float(REPEAT_LOOP_REWARD) + step_penalty, 10)
    return {
        "done": True,
        "over": True,
        "reward": reward,
        "termination_reason": "repeat_loop",
        "termination_subreason": subreason,
        "reward_valid": True,
        "reward_detail": {
            "reward_version": "shopsimulator-reward-v4",
            "query_constraint_version": "shopping-query-constraints-v1",
            "reward_type": "repeat_loop",
            "reward_valid": True,
            "termination_reason": "repeat_loop",
            "termination_subreason": subreason,
            "target_asin_match": False,
            "hard_gates": {},
            "weighted_score": 0.0,
            "evidence_coverage": 0.0,
            "dimension_scores": {},
            "constraint_results": [],
            "terminal_utility": reward,
            "base_terminal_utility": REPEAT_LOOP_REWARD,
            "step_count": step_count,
            "step_penalty": step_penalty,
            "step_penalty_version": "shopping-step-penalty-v1",
            "purchase_success": False,
            "sampling_invalid": False,
        },
    }


def _candidate_selection_observation(candidate_memory):
    """Project saved public candidates as one temporary search-like page."""
    entries = [
        entry
        for entry in (candidate_memory or {}).get("entries", [])
        if isinstance(entry, dict) and is_product_id(str(entry.get("asin") or ""))
    ]
    lines = [
        "[SHOPPING_OBSERVATION_V2]",
        "page_type: candidate_selection",
        "候选收敛阶段：已连续6步没有获得实质进展，无进展计数已清零。",
        "以下已核验候选现在临时视为同一页商品；本轮只能调用 open_product，asin 必须取自下列候选。",
        "一旦打开某个候选，该候选将被锁定，之后不能返回候选页或改选其他候选；必须先比较候选摘要，再打开整体最符合要求的一个。",
        "格式: rank|asin|price|brand|category|selected_options|title|public_evidence",
    ]
    actions = []
    for rank, entry in enumerate(entries, start=1):
        asin = str(entry.get("asin") or "").strip().upper()
        actions.append(asin)
        selected = entry.get("selected_options") or {}
        selected_text = ",".join(
            f"{axis}={value}" for axis, value in sorted(selected.items())
        )
        evidence = ",".join(str(value) for value in (entry.get("evidence") or []))
        fields = (
            rank,
            asin,
            entry.get("price") or "-",
            entry.get("brand") or "-",
            entry.get("category") or "-",
            selected_text or "-",
            entry.get("title") or "-",
            evidence or "-",
        )
        lines.append("|".join(str(value).replace("\n", " ") for value in fields))
    if actions:
        lines.append(f'调用示例: open_product(asin="{actions[0]}")')
    lines.extend(
        [
            "",
            "搜索功能是否可用: False",
            "可点击的按钮: " + json.dumps(actions, ensure_ascii=False),
        ]
    )
    return "\n".join(lines)


def _candidate_phase_notice(observation, phase):
    """Prepend the exact forced-stage contract without changing page evidence."""
    if phase == CANDIDATE_PHASE_SELECT:
        notice = (
            "候选规格阶段：已打开强制收敛候选，本阶段不提供搜索、返回或其他候选。"
            "只能调用 select_option 选择当前商品的一个合适规格。"
        )
    elif phase == CANDIDATE_PHASE_TERMINAL:
        notice = (
            "候选终局阶段：规格选择已完成或当前候选没有可选规格。"
            "如果当前候选满足全部 Hard 且可以接受，立即调用 buy_now 购买；"
            "只有当前候选存在无法接受的 Hard 违反时，才调用 finish_without_purchase 放弃。"
            "放弃后任务立即失败结束。"
        )
    else:
        return str(observation or "")
    return f"{notice}\n\n{observation}"


def _has_unselected_option(observation):
    status = re.search(
        r"规格状态:\s*(\d+)\s*/\s*(\d+)\s*个规格轴已选择",
        str(observation or ""),
    )
    if status:
        selected_axes, total_axes = map(int, status.groups())
        return selected_axes < total_axes
    selected = selected_option_ids(observation)
    return any(
        re.fullmatch(r"opt_[0-9a-f]{16}", button.casefold())
        and button.casefold() not in selected
        for button in clickable_buttons(observation)
    )


def _candidate_context_anchor(initial_messages):
    """Keep only the governing system prompt and the original shopping request."""
    systems = [
        json.loads(json.dumps(message, ensure_ascii=False))
        for message in initial_messages
        if message.get("role") == "system"
    ]
    original_users = [
        message for message in initial_messages if message.get("role") == "user"
    ]
    if original_users:
        systems.append(
            json.loads(json.dumps(original_users[-1], ensure_ascii=False))
        )
    return systems


def _candidate_phase_context(anchor, observation):
    """Build a fresh model context containing no pre-recovery page history."""
    context = json.loads(json.dumps(anchor, ensure_ascii=False))
    context.append(
        {
            "role": "user",
            "content": str(observation or ""),
        }
    )
    return context


def rollout_interrupted(signum, frame):
    """将终止信号转为 KeyboardInterrupt，使 collect_for_task 的 finally 释放环境。"""
    raise KeyboardInterrupt


class OpenAIChatClient:
    def __init__(
        self,
        model,
        base_url,
        api_key,
        temperature=0.0,
        top_p=1.0,
        timeout=60,
        max_tokens=512,
        thinking=False,
        reasoning_effort="high",
        context_window=None,
        context_safety_margin=512,
        context_compaction_enable=False,
        observation_token_budget=2560,
        observation_detail_token_budget=3072,
        observation_generic_token_budget=512,
        observation_candidate_memory_token_budget=1024,
        observation_search_top_k=20,
        seed=20260806,
        tool_choice="auto",
        missing_tool_call_retries=0,
        token_counter=None,
        observation_token_counter=None,
        transport=None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.timeout = timeout
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.thinking = bool(thinking)
        self.reasoning_effort = reasoning_effort
        self.context_window = int(context_window) if context_window else None
        self.context_safety_margin = int(context_safety_margin)
        self.context_compaction_enable = bool(context_compaction_enable)
        self.observation_token_budget = (
            int(observation_token_budget) if observation_token_budget else None
        )
        self.observation_search_top_k = int(observation_search_top_k)
        self.observation_detail_token_budget = int(observation_detail_token_budget)
        self.observation_generic_token_budget = int(observation_generic_token_budget)
        self.observation_candidate_memory_token_budget = int(
            observation_candidate_memory_token_budget
        )
        self.seed = int(seed)
        self.tool_choice = str(tool_choice)
        self.missing_tool_call_retries = int(missing_tool_call_retries)
        if self.tool_choice not in {"auto", "required"}:
            raise ValueError("tool_choice must be auto or required")
        if self.missing_tool_call_retries < 0:
            raise ValueError("missing_tool_call_retries cannot be negative")
        if self.context_window is not None:
            if self.context_window <= self.max_tokens + self.context_safety_margin:
                raise ValueError("context_window must exceed max_tokens plus context_safety_margin")
            self.token_counter = token_counter or VllmChatTokenCounter(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        else:
            self.token_counter = token_counter
        if self.observation_token_budget is not None:
            if self.observation_token_budget < 64:
                raise ValueError("observation_token_budget must be at least 64")
            if self.observation_candidate_memory_token_budget < 64:
                raise ValueError(
                    "observation_candidate_memory_token_budget must be at least 64"
                )
            self.observation_token_counter = observation_token_counter or VllmTextTokenCounter(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        else:
            self.observation_token_counter = observation_token_counter
        self.last_context_event = None
        self.last_context_tokens = None
        self.last_call_metrics = None
        self.transport = transport

    def complete(self, messages, tools):
        """请求模型下一轮回复，并在上下文超限时按配置压缩历史。"""
        self.last_context_event = None
        self.last_context_tokens = None
        self.last_call_metrics = None
        request_messages = messages
        if self.context_window is not None:
            input_budget = self.context_window - self.max_tokens - self.context_safety_margin
            original_tokens = int(self.token_counter(messages, tools))
            self.last_context_tokens = original_tokens
            if original_tokens > input_budget:
                if not self.context_compaction_enable:
                    raise ContextBudgetError(
                        f"prompt uses {original_tokens} tokens, above input budget {input_budget}"
                    )
                request_messages, stats = compact_chat_messages(
                    messages,
                    tools,
                    count_tokens=self.token_counter,
                    max_input_tokens=input_budget,
                )
                self.last_context_tokens = int(stats.final_tokens)
                if stats.removed_groups:
                    self.last_context_event = stats.to_dict()
        payload = {
            "model": self.model,
            "messages": request_messages,
            "tools": tools,
            "tool_choice": self.tool_choice,
            # 约束单个 assistant 回合的输出；--max-model-len 只限制上下文，
            # 不能防止模型在未调用工具时持续生成纯文本。
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
        if self.thinking:
            # DeepSeek tool-call thinking requires reasoning_content in later messages.
            payload.update(
                {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": self.reasoning_effort,
                }
            )
        else:
            payload.update({"temperature": self.temperature, "top_p": self.top_p})
            if self.model.casefold().startswith("deepseek-v4"):
                if "/zen/go/v1" in self.base_url.casefold():
                    # OpenCode Go exposes only low/high/max reasoning effort for
                    # DeepSeek V4; its current route ignores the standard thinking
                    # toggle. Keep reasoning at the lowest supported level and
                    # prevent parallel calls so one shopping turn remains one action.
                    payload["reasoning"] = {"effort": "low"}
                    payload["parallel_tool_calls"] = False
                else:
                    payload["thinking"] = {"type": "disabled"}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # 避免 Cloudflare 将 Python urllib 默认客户端识别为自动化流量。
            "User-Agent": "shopping-grpo/0.1 (OpenAI-compatible evaluation)",
        }
        url = f"{self.base_url}/chat/completions"
        started = time.monotonic()
        network_attempts = 0
        for missing_attempt in range(self.missing_tool_call_retries + 1):
            for attempt in range(MODEL_COMPLETION_RETRIES + 1):
                network_attempts += 1
                try:
                    if self.transport is not None:
                        response = self.transport(url, payload, headers, self.timeout)
                    else:
                        request = Request(
                            url,
                            data=json.dumps(payload).encode("utf-8"),
                            headers=headers,
                            method="POST",
                        )
                        with urlopen(request, timeout=self.timeout) as raw:
                            response = json.loads(raw.read().decode("utf-8"))
                    break
                except (IncompleteRead, RemoteDisconnected, TimeoutError, URLError):
                    if attempt >= MODEL_COMPLETION_RETRIES:
                        raise
                    time.sleep(MODEL_RETRY_DELAY_SECONDS * (attempt + 1))
            assistant = _response_message(response)
            tool_calls = assistant.get("tool_calls") or []
            if (
                self.tool_choice == "required"
                and not tool_calls
                and missing_attempt < self.missing_tool_call_retries
            ):
                time.sleep(MODEL_RETRY_DELAY_SECONDS * (missing_attempt + 1))
                continue
            usage = response.get("usage") if isinstance(response, dict) else None
            self.last_call_metrics = {
                "latency_seconds": time.monotonic() - started,
                "attempts": network_attempts,
                "missing_tool_call_retries": missing_attempt,
                "usage": _plain(usage) if isinstance(usage, dict) else {},
            }
            return assistant

    def project_observation(self, tool_name, observation, parameters=None):
        if self.observation_token_budget is None:
            return str(observation), None
        visible, meta = project_observation(
            tool_name=tool_name,
            observation=observation,
            parameters=parameters,
            count_tokens=self.observation_token_counter,
            token_budget=self.observation_token_budget,
            detail_token_budget=self.observation_detail_token_budget,
            generic_token_budget=self.observation_generic_token_budget,
            candidate_memory_token_budget=(
                self.observation_candidate_memory_token_budget
            ),
            search_top_k=self.observation_search_top_k,
        )
        return visible, meta.to_dict()


class MultiKeyOpenAIChatClient:
    """Distribute requests across isolated client slots without exposing keys."""

    def __init__(self, *, api_keys, per_key_concurrency, client_kwargs):
        keys = [str(key) for key in api_keys if str(key)]
        if len(keys) < 2:
            raise ValueError("MultiKeyOpenAIChatClient requires at least two API keys")
        per_key_concurrency = int(per_key_concurrency)
        if per_key_concurrency < 1:
            raise ValueError("per_key_concurrency must be positive")
        self._pools = []
        for key in keys:
            pool = queue.Queue(maxsize=per_key_concurrency)
            for _ in range(per_key_concurrency):
                pool.put(OpenAIChatClient(api_key=key, **client_kwargs))
            self._pools.append(pool)
        self._cursor = 0
        self._cursor_lock = threading.Lock()
        self._local = threading.local()
        exemplar = self._pools[0].queue[0]
        for name in (
            "model",
            "context_window",
            "max_tokens",
            "context_safety_margin",
            "observation_token_budget",
        ):
            setattr(self, name, getattr(exemplar, name))

    def _next_pool(self):
        with self._cursor_lock:
            start = self._cursor
            self._cursor = (self._cursor + 1) % len(self._pools)
        for offset in range(len(self._pools)):
            pool = self._pools[(start + offset) % len(self._pools)]
            try:
                return pool, pool.get_nowait()
            except queue.Empty:
                continue
        pool = self._pools[start]
        return pool, pool.get()

    def complete(self, messages, tools):
        pool, client = self._next_pool()
        try:
            assistant = client.complete(messages, tools)
            self._local.last_call_metrics = client.last_call_metrics
            self._local.last_context_tokens = client.last_context_tokens
            self._local.last_context_event = client.last_context_event
            return assistant
        finally:
            pool.put(client)

    @property
    def last_call_metrics(self):
        return getattr(self._local, "last_call_metrics", None)

    @property
    def last_context_tokens(self):
        return getattr(self._local, "last_context_tokens", None)

    @property
    def last_context_event(self):
        return getattr(self._local, "last_context_event", None)

    def project_observation(self, tool_name, observation, parameters=None):
        client = self._pools[0].queue[0]
        return client.project_observation(tool_name, observation, parameters)


class ToolExecutionError(RuntimeError):
    def __init__(self, step, original):
        super().__init__(str(original))
        self.step = step
        self.original = original


class CollectionInfrastructureError(RuntimeError):
    """环境租约未恢复时，阻止采集器继续污染后续任务。"""


def load_tasks(path):
    tasks = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = _task_id(row)
            if task_id is None:
                raise ValueError("task row is missing task_id")
            row = dict(row)
            row["task_id"] = int(task_id)
            tasks.append(row)
    return tasks


def completed_task_attempts(path):
    path = Path(path)
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "task_id" in row:
                done.add((int(row["task_id"]), int(row.get("attempt_index", 0))))
    return done


def collect_for_task(
    task,
    client,
    env_factory=ShopAgentEnv,
    base_url="http://127.0.0.1:5700",
    max_steps=45,
    tools=None,
    attempt_index=0,
    system_prompt=SYSTEM_PROMPT,
    evaluation_extensions=False,
):
    """执行一个任务并返回完整轨迹；所有异常都会被写入轨迹后再释放环境。"""
    started = time.monotonic()
    context_window = getattr(client, "context_window", None)
    max_tokens = int(getattr(client, "max_tokens", 0) or 0)
    safety_margin = int(getattr(client, "context_safety_margin", 0) or 0)
    trajectory = {
        "trajectory_id": str(uuid4()),
        "task_id": int(task["task_id"]),
        "attempt_index": int(attempt_index),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "messages": [],
        "steps": [],
        "blocked_tool_calls": [],
        "missing_tool_call_corrections": [],
        "tool_call_truncations": [],
        "context_compactions": [],
        "context_turn_tokens": [],
        "model_calls": [],
        "timing": {},
        "context_budget": {
            "context_window": context_window,
            "reserved_completion_tokens": max_tokens,
            "safety_margin_tokens": safety_margin,
            "max_input_tokens": (
                int(context_window) - max_tokens - safety_margin
                if context_window is not None
                else None
            ),
        },
        "candidate_memory": new_candidate_memory(
            max_entries=4 if evaluation_extensions else 6,
            stable_candidate_ids=evaluation_extensions
        ),
        "candidate_forced_phase": None,
        "candidate_recovery_events": [],
        "candidate_context_resets": [],
        "initial_result": {},
        "terminal_result": {},
        "final_reward": 0.0,
        "done": False,
        "error": None,
        "release_error": None,
    }
    env = env_factory(base_url=base_url)
    try:
        # reset 建立任务状态；后续每一轮只允许一个工具调用。
        initial = env.reset(task["task_id"])
        if evaluation_extensions and hasattr(env, "configure_candidate_recovery"):
            env.configure_candidate_recovery()
        if initial.get("observation_state") is not None:
            latest_observation = render_structured_observation(
                initial["observation_state"],
                candidate_memory=trajectory["candidate_memory"],
                step_count=0,
                show_candidate_memory=not evaluation_extensions,
            )
        else:
            latest_observation = initial.get("instruction", initial.get("observation", ""))
        trajectory["initial_result"] = initial
        messages = _initial_messages(task, initial, system_prompt=system_prompt)
        trajectory["messages"] = messages
        audit_messages = trajectory["messages"]
        candidate_context_anchor = _candidate_context_anchor(messages)

        def append_message(message):
            messages.append(message)
            if messages is not audit_messages:
                audit_messages.append(message)

        tool_schemas = tools or (
            EVALUATION_TOOL_SCHEMAS
            if evaluation_extensions
            else SHOP_TOOL_SCHEMAS
        )
        consecutive_blocked_calls = 0
        consecutive_missing_tool_calls = 0
        latest_observation_truncated = False

        while len(trajectory["steps"]) < int(max_steps):
            # 先请求模型，再校验动作；工具结果会追加到 messages，成为下一轮上下文。
            active_tool_schemas = (
                _active_tool_schemas(
                    tool_schemas,
                    len(trajectory["steps"]),
                    latest_observation=latest_observation,
                    candidate_memory=trajectory["candidate_memory"],
                    candidate_phase=trajectory["candidate_forced_phase"],
                )
                if evaluation_extensions
                else list(tool_schemas)
            )
            assistant = client.complete(messages, active_tool_schemas)
            _capture_client_call(
                trajectory,
                client,
                step_index=len(trajectory["steps"]),
            )
            assistant, dropped_tool_calls = _enforce_serial_tool_call(assistant)
            if dropped_tool_calls:
                trajectory["tool_call_truncations"].append(
                    {
                        "message_index": len(messages),
                        "kept_tool_call_id": assistant["tool_calls"][0].get("id"),
                        "dropped_tool_calls": dropped_tool_calls,
                    }
                )
            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                append_message(assistant)
                if consecutive_missing_tool_calls < MAX_MISSING_TOOL_CALL_CORRECTIONS:
                    consecutive_missing_tool_calls += 1
                    correction = {
                        "role": "user",
                        "content": MISSING_TOOL_CALL_CORRECTION,
                    }
                    append_message(correction)
                    trajectory["missing_tool_call_corrections"].append(
                        {
                            "step_index": len(trajectory["steps"]),
                            "correction_attempt": consecutive_missing_tool_calls,
                            "assistant_message_index": len(messages) - 2,
                            "correction_message_index": len(messages) - 1,
                        }
                    )
                    continue
                trajectory["status"] = "assistant_final"
                terminal_result = _assistant_final_result(len(trajectory["steps"]))
                trajectory["terminal_result"] = terminal_result
                trajectory["final_reward"] = terminal_result["reward"]
                trajectory["done"] = True
                break
            consecutive_missing_tool_calls = 0
            tool_call = tool_calls[0]
            try:
                name, arguments = _tool_call_name_args(tool_call)
                reason = action_reject_reason(
                    name,
                    arguments,
                    latest_observation,
                    tool_schemas=(
                        active_tool_schemas
                        if evaluation_extensions
                        else tool_schemas
                    ),
                    candidate_memory=trajectory["candidate_memory"],
                    step_count=len(trajectory["steps"]),
                    evaluation_extensions=evaluation_extensions,
                )
            except Exception as exc:
                reason = f"invalid_tool_call:{exc.__class__.__name__}"
            if reason:
                consecutive_blocked_calls += 1
                trajectory["blocked_tool_calls"].append(
                    {
                        "step_index": len(trajectory["steps"]),
                        "tool_call": tool_call,
                        "reason": reason,
                        "consecutive_count": consecutive_blocked_calls,
                        "latest_observation_truncated": latest_observation_truncated,
                    }
                )
                append_message(assistant)
                append_message(
                    action_guard_tool_message(
                        tool_call,
                        reason,
                        latest_observation,
                        candidate_memory=trajectory["candidate_memory"],
                        tool_schemas=active_tool_schemas,
                    )
                )
                if consecutive_blocked_calls >= MAX_BLOCKED_TOOL_CALLS:
                    trajectory["status"] = "invalid_action_limit"
                    terminal_result = _guard_rejection_result(
                        len(trajectory["steps"])
                    )
                    trajectory["terminal_result"] = terminal_result
                    trajectory["final_reward"] = terminal_result["reward"]
                    trajectory["done"] = True
                    break
                continue
            if len(trajectory["steps"]) >= int(max_steps):
                trajectory["status"] = "max_steps"
                return trajectory
            append_message(assistant)
            if (
                evaluation_extensions
                and trajectory["candidate_forced_phase"]
                and name == "finish_without_purchase"
            ):
                terminal_result = _repeat_loop_result(
                    len(trajectory["steps"]) + 1,
                    subreason="no_progress_loop",
                )
                step = {
                    "step_index": len(trajectory["steps"]),
                    "tool_call": tool_call,
                    "tool_name": name,
                    "parameters": arguments,
                    "env_action": None,
                    "observation": (
                        "[SHOPPING_OBSERVATION_V2]\npage_type: terminal\n"
                        "候选强制收敛阶段选择不购买，按 repeat_loop 终止。"
                    ),
                    "reward": terminal_result["reward"],
                    "done": True,
                    "result": terminal_result,
                    "tool_latency_seconds": 0.0,
                }
                trajectory["steps"].append(step)
                append_message(_tool_message(tool_call, step))
                trajectory["status"] = "done"
                trajectory["terminal_result"] = terminal_result
                trajectory["final_reward"] = terminal_result["reward"]
                trajectory["done"] = True
                return trajectory
            # 只有通过当前 observation 守卫的调用才会触碰环境并消耗一个执行步骤。
            step = _execute_tool_call(
                env,
                tool_call,
                len(trajectory["steps"]),
                latest_observation,
                candidate_memory=trajectory["candidate_memory"],
                evaluation_extensions=evaluation_extensions,
                forced_candidate_open=(
                    trajectory["candidate_forced_phase"]
                    == CANDIDATE_PHASE_CHOOSE
                ),
            )
            raw_observation = step["observation"]
            current_phase = trajectory["candidate_forced_phase"]
            if current_phase == CANDIDATE_PHASE_CHOOSE:
                trajectory["candidate_forced_phase"] = (
                    CANDIDATE_PHASE_SELECT
                    if _has_unselected_option(raw_observation)
                    else CANDIDATE_PHASE_TERMINAL
                )
            elif current_phase == CANDIDATE_PHASE_SELECT:
                trajectory["candidate_forced_phase"] = CANDIDATE_PHASE_TERMINAL

            progress = ((step.get("result") or {}).get("progress") or {})
            recovery_required = bool(progress.get("candidate_recovery_required"))
            if recovery_required and current_phase is None:
                candidate_count = len(
                    trajectory["candidate_memory"].get("entries") or []
                )
                event = {
                    "step_index": len(trajectory["steps"]),
                    "candidate_count": candidate_count,
                    "trigger_progress": dict(progress),
                }
                if candidate_count == 0:
                    terminal_result = _repeat_loop_result(
                        len(trajectory["steps"]) + 1,
                        subreason="no_progress_loop",
                    )
                    step["result"] = terminal_result
                    step["reward"] = terminal_result["reward"]
                    step["done"] = True
                    step["observation"] = (
                        "[SHOPPING_OBSERVATION_V2]\npage_type: terminal\n"
                        "连续6步无实质进展且没有已核验候选，按 repeat_loop 终止。"
                    )
                    event["outcome"] = "repeat_loop_without_candidate"
                    trajectory["candidate_recovery_events"].append(event)
                    raw_observation = step["observation"]
                else:
                    reset_result = (
                        env.reset_no_progress()
                        if hasattr(env, "reset_no_progress")
                        else {"no_progress_steps": 0, "consecutive_repeats": 0}
                    )
                    progress["no_progress_steps"] = 0
                    progress["consecutive_repeats"] = 0
                    progress["candidate_recovery_required"] = False
                    progress["candidate_recovery_triggered"] = True
                    trajectory["candidate_forced_phase"] = CANDIDATE_PHASE_CHOOSE
                    event.update(
                        {
                            "outcome": "candidate_selection",
                            "reset_result": reset_result,
                        }
                    )
                    trajectory["candidate_recovery_events"].append(event)
                    raw_observation = _candidate_selection_observation(
                        trajectory["candidate_memory"]
                    )

            if trajectory["candidate_forced_phase"] == CANDIDATE_PHASE_CHOOSE:
                model_observation = raw_observation
            else:
                model_observation = add_step_budget_notice(
                    raw_observation,
                    step_count=len(trajectory["steps"]) + 1,
                    max_steps=max_steps,
                    candidate_count=0,
                    no_progress_steps=progress.get("no_progress_steps", 0),
                )
            step["observation"] = model_observation
            projector = getattr(client, "project_observation", None)
            if (
                projector is not None
                and trajectory["candidate_forced_phase"] != CANDIDATE_PHASE_CHOOSE
            ):
                visible_observation, projection = projector(
                    step["tool_name"],
                    model_observation,
                    step["parameters"],
                )
                step["observation"] = visible_observation
                if projection is not None:
                    step["raw_observation"] = raw_observation
                    step["projection"] = projection
            model_observation = _candidate_phase_notice(
                step["observation"],
                trajectory["candidate_forced_phase"],
            )
            step["observation"] = model_observation
            trajectory["steps"].append(step)
            consecutive_blocked_calls = 0
            latest_observation = step["observation"]
            latest_observation_truncated = bool((step.get("projection") or {}).get("truncated"))
            append_message(_tool_message(tool_call, step))
            if trajectory["candidate_forced_phase"] and not step["done"]:
                previous_message_count = len(messages)
                messages = _candidate_phase_context(
                    candidate_context_anchor,
                    latest_observation,
                )
                trajectory["candidate_context_resets"].append(
                    {
                        "step_index": step["step_index"],
                        "phase": trajectory["candidate_forced_phase"],
                        "discarded_model_messages": previous_message_count,
                        "new_model_messages": len(messages),
                    }
                )
            if step["done"]:
                trajectory["status"] = "done"
                trajectory["terminal_result"] = step["result"]
                trajectory["final_reward"] = step["reward"]
                trajectory["done"] = True
                return trajectory
        else:
            trajectory["status"] = "max_steps"
        if trajectory["steps"] and not trajectory["done"]:
            trajectory["final_reward"] = trajectory["steps"][-1]["reward"]
    except ToolExecutionError as exc:
        trajectory["steps"].append(exc.step)
        trajectory["status"] = "error"
        trajectory["error"] = {
            "type": exc.original.__class__.__name__,
            "message": str(exc.original),
            "traceback": "".join(traceback.format_exception(exc.original)),
        }
    except Exception as exc:
        trajectory["status"] = "error"
        trajectory["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            env.release()
        except Exception as exc:
            trajectory["release_error"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            trajectory["status"] = "environment_release_failed"
        trajectory["timing"]["trajectory_duration_seconds"] = time.monotonic() - started
    return trajectory


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_tasks(
    tasks,
    client,
    output_path,
    base_url,
    max_steps=45,
    env_factory=ShopAgentEnv,
    attempts_per_task=1,
    attempt_start=0,
    workers=1,
    system_prompt=SYSTEM_PROMPT,
    system_prompt_factory=None,
    client_factory=None,
    evaluation_extensions=False,
):
    attempts_per_task = int(attempts_per_task)
    attempt_start = int(attempt_start)
    if attempts_per_task < 1:
        raise ValueError("attempts_per_task must be at least 1")
    if attempt_start < 0:
        raise ValueError("attempt_start must be non-negative")
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    done = completed_task_attempts(output_path)
    written = []
    jobs = [
        (task, attempt_index)
        for task in tasks
        for attempt_index in range(attempt_start, attempt_start + attempts_per_task)
        if (int(task["task_id"]), attempt_index) not in done
    ]

    def collect_job(task, attempt_index):
        task_system_prompt = (
            system_prompt_factory(task)
            if system_prompt_factory is not None
            else system_prompt
        )
        job_client = (
            client_factory(task, attempt_index)
            if client_factory is not None
            else client
        )
        return collect_for_task(
            task,
            client=job_client,
            env_factory=env_factory,
            base_url=base_url,
            max_steps=max_steps,
            attempt_index=attempt_index,
            system_prompt=task_system_prompt,
            evaluation_extensions=evaluation_extensions,
        )

    if workers == 1:
        for task, attempt_index in jobs:
            trajectory = collect_job(task, attempt_index)
            append_jsonl(output_path, [trajectory])
            written.append(trajectory)
            if _is_infrastructure_failure(trajectory):
                raise CollectionInfrastructureError(
                    "collection infrastructure failure; stopped before the next task"
                )
        return written

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(collect_job, task, attempt_index): (
                int(task["task_id"]),
                attempt_index,
            )
            for task, attempt_index in jobs
        }
        for future in as_completed(futures):
            trajectory = future.result()
            append_jsonl(output_path, [trajectory])
            written.append(trajectory)
            if _is_infrastructure_failure(trajectory):
                for pending in futures:
                    pending.cancel()
                task_id, attempt_index = futures[future]
                raise CollectionInfrastructureError(
                    "collection infrastructure failure at "
                    f"task_id={task_id}, attempt_index={attempt_index}"
                )
    return written


def _is_infrastructure_failure(trajectory):
    """环境或模型服务不可用时中断；普通任务失败仍保留并继续。"""
    if trajectory.get("release_error"):
        return True
    error = trajectory.get("error") or {}
    error_type = error.get("type")
    if error_type in {
        IncompleteRead.__name__,
        URLError.__name__,
        RemoteDisconnected.__name__,
        TimeoutError.__name__,
    }:
        return True
    if error_type == ShopHttpError.__name__:
        return True
    return (
        error_type == ShopEnvironmentError.__name__
        and "Unable to get available environment resource" in error.get("message", "")
    )


def _phase_tool_schemas(tool_schemas, step_count, *, candidate_memory=None):
    """Compatibility shim: step/candidate phases no longer hide tools."""
    return list(tool_schemas)


def _active_tool_schemas(
    tool_schemas,
    step_count,
    *,
    latest_observation,
    candidate_memory=None,
    candidate_phase=None,
):
    """Expose only tools executable from the latest public page."""
    phased = _phase_tool_schemas(
        tool_schemas,
        step_count,
        candidate_memory=candidate_memory,
    )
    forced_names = {
        CANDIDATE_PHASE_CHOOSE: {"open_product"},
        CANDIDATE_PHASE_SELECT: {"select_option"},
        CANDIDATE_PHASE_TERMINAL: {"buy_now", "finish_without_purchase"},
    }.get(candidate_phase)
    if forced_names is not None:
        return [
            tool
            for tool in phased
            if (tool.get("function") or {}).get("name") in forced_names
        ]
    observation = str(latest_observation or "")
    # Legacy/unstructured environments do not expose the stable action footer.
    # Keep the complete supplied contract there; Final-240 uses structured state.
    if "搜索功能是否可用:" not in observation or "可点击的按钮:" not in observation:
        return phased

    buttons = clickable_buttons(observation)
    normalized_buttons = {button.casefold() for button in buttons}
    selected_ids = selected_option_ids(observation)
    page_allowed = {"finish_without_purchase"}
    if "搜索功能是否可用: True" in observation:
        page_allowed.add("search_products")
    if product_ids(observation):
        page_allowed.add("open_product")
    if "next >" in normalized_buttons:
        page_allowed.add("next_page")
    if "< prev" in normalized_buttons:
        page_allowed.add("prev_page")
    if "back to search" in normalized_buttons:
        page_allowed.add("back_to_search")
    if "buy now" in normalized_buttons:
        page_allowed.add("buy_now")
    if any(
        re.fullmatch(r"opt_[0-9a-f]{16}", button.casefold())
        and button.casefold() not in selected_ids
        for button in buttons
    ):
        page_allowed.add("select_option")

    return [
        tool
        for tool in phased
        if (tool.get("function") or {}).get("name") in page_allowed
    ]


def _capture_client_call(trajectory, client, *, step_index):
    call_metrics = getattr(client, "last_call_metrics", None)
    if isinstance(call_metrics, dict):
        trajectory["model_calls"].append(
            {"step_index": int(step_index), **call_metrics}
        )
    context_tokens = getattr(client, "last_context_tokens", None)
    if context_tokens is not None:
        trajectory["context_turn_tokens"].append(
            {"step_index": int(step_index), "input_tokens": int(context_tokens)}
        )
    context_event = getattr(client, "last_context_event", None)
    if context_event:
        trajectory["context_compactions"].append(
            {"step_index": int(step_index), **context_event}
        )


def _execute_tool_call(
    env,
    tool_call,
    step_index,
    latest_observation="",
    *,
    candidate_memory=None,
    evaluation_extensions=False,
    forced_candidate_open=False,
):
    name, arguments = _tool_call_name_args(tool_call)
    env_arguments = resolve_action_parameters(name, arguments, latest_observation)
    if forced_candidate_open and name == "open_product":
        candidate_asins = {
            str(entry.get("asin") or "").strip().upper()
            for entry in (candidate_memory or {}).get("entries", [])
            if isinstance(entry, dict)
        }
        asin = str(env_arguments.get("asin") or "").strip().upper()
        if asin not in candidate_asins:
            raise ValueError("forced candidate ASIN is not present in candidate memory")
        action = f"reopen[{asin}]"
    else:
        action = tool_call_to_action(name, env_arguments)
    result = {"instruction": arguments.get("note", ""), "reward": 0.0, "done": False}
    step = {
        "step_index": step_index,
        "tool_call": tool_call,
        "tool_name": name,
        "parameters": arguments,
        "env_action": action,
        "observation": "",
        "reward": 0.0,
        "done": False,
        "result": {},
    }
    if action is not None:
        started = time.monotonic()
        try:
            result = env.step(action)
        except Exception as exc:
            step["tool_latency_seconds"] = time.monotonic() - started
            step["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
            raise ToolExecutionError(step, exc) from exc
        step["tool_latency_seconds"] = time.monotonic() - started
    else:
        step["tool_latency_seconds"] = 0.0
    if result.get("observation_state") is not None:
        observation = render_structured_observation(
            result["observation_state"],
            candidate_memory=candidate_memory,
            step_count=step_index + 1,
            show_candidate_memory=not evaluation_extensions,
        )
    else:
        observation = result.get("instruction", result.get("observation", ""))
    step.update(
        {
            "observation": observation,
            "reward": float(result.get("reward", 0.0)),
            "done": bool(result.get("done", False)),
            "result": result,
        }
    )
    return step


def _tool_message(tool_call, step):
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": step["tool_name"],
        "content": step["observation"],
    }


def _enforce_serial_tool_call(assistant):
    """每轮只把一个工具调用交给环境，防止在旧 observation 上批量点击。"""
    tool_calls = assistant.get("tool_calls") or []
    if len(tool_calls) <= 1:
        return assistant, []
    serial_assistant = dict(assistant)
    serial_assistant["tool_calls"] = [tool_calls[0]]
    return serial_assistant, list(tool_calls[1:])


def _initial_messages(task, initial, system_prompt=SYSTEM_PROMPT):
    prompt = task.get("prompt")
    if prompt:
        messages = [dict(message) for message in prompt]
    else:
        messages = [{"role": "system", "content": system_prompt}]
    if not any(message.get("role") == "system" for message in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": initial.get("instruction", "")})
    return messages


def _task_id(row):
    if "task_id" in row:
        return row["task_id"]
    extra = row.get("extra_info") or {}
    if "task_id" in extra:
        return extra["task_id"]
    kwargs = extra.get("interaction_kwargs") or {}
    return kwargs.get("task_id")


def _tool_call_name_args(tool_call):
    function = tool_call.get("function") or {}
    name = function.get("name")
    raw_args = function.get("arguments") or "{}"
    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    return name, parsed_args


def _response_message(response):
    choice = response["choices"][0]
    message = choice["message"]
    return _plain(message)


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return {k: _plain(v) for k, v in value.__dict__.items() if not k.startswith("_")}
    return value


def client_from_env(
    model=None,
    base_url=None,
    api_key=None,
    temperature=0.0,
    top_p=1.0,
    timeout=60,
    max_tokens=512,
    thinking=False,
    reasoning_effort="high",
    context_window=None,
    context_safety_margin=512,
    context_compaction_enable=False,
    observation_token_budget=2560,
    observation_detail_token_budget=3072,
    observation_generic_token_budget=512,
    observation_candidate_memory_token_budget=1024,
    observation_search_top_k=20,
    seed=20260806,
):
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("api_key or OPENAI_API_KEY is required")
    return OpenAIChatClient(
        model=model or os.environ.get("OPENAI_MODEL", "deepseek-chat"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        max_tokens=max_tokens,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        context_window=context_window,
        context_safety_margin=context_safety_margin,
        context_compaction_enable=context_compaction_enable,
        observation_token_budget=observation_token_budget,
        observation_detail_token_budget=observation_detail_token_budget,
        observation_generic_token_budget=observation_generic_token_budget,
        observation_candidate_memory_token_budget=(
            observation_candidate_memory_token_budget
        ),
        observation_search_top_k=observation_search_top_k,
        seed=seed,
    )
