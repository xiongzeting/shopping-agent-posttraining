"""veRL 0.8 ToolAgentLoop 的 ShopSimulator 轨迹生命周期适配。

这个类不重新实现模型生成，而是在三个边界插入项目约束：上下文超长时压缩、工具
返回后投影 observation、trajectory 结束时计算奖励并释放环境。
"""

from __future__ import annotations

import json

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

from shopping_grpo.environment.context import ContextBudgetError, compact_token_trajectory
from shopping_grpo.environment.projection import (
    ObservationProjectionError,
    project_observation,
)
from shopping_grpo.training.grpo.adapter.runtime import (
    current_runtime_state,
    record_observation_projection,
    reward_breakdown,
    task_id_from_kwargs,
    terminal_reward,
)
from shopping_grpo.training.grpo.adapter.session import ShopSimulatorSession


class ShoppingToolAgentLoop(ToolAgentLoop):
    """Vanilla ToolAgentLoop with deterministic ShopSimulator termination and release."""

    def __init__(
        self,
        *args,
        base_url="http://127.0.0.1:5700",
        timeout=60,
        max_steps=45,
        required_environment_version=None,
        reward_mode="native",
        context_window_tokens=30000,
        context_generation_reserve_tokens=768,
        context_safety_margin_tokens=512,
        context_input_budget_tokens=28720,
        context_preserve_recent_groups=1,
        context_compaction_enable=False,
        observation_token_budget=2560,
        observation_detail_token_budget=3072,
        observation_generic_token_budget=512,
        observation_candidate_memory_token_budget=1024,
        observation_search_top_k=20,
        env_factory=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_url = base_url
        self.timeout = int(timeout)
        self.max_steps = int(max_steps)
        self.required_environment_version = required_environment_version
        self.reward_mode = str(reward_mode)
        self.context_window_tokens = int(context_window_tokens)
        self.context_generation_reserve_tokens = int(context_generation_reserve_tokens)
        self.context_safety_margin_tokens = int(context_safety_margin_tokens)
        self.context_input_budget_tokens = int(context_input_budget_tokens)
        self.context_preserve_recent_groups = int(context_preserve_recent_groups)
        self.context_compaction_enable = bool(context_compaction_enable)
        self.observation_token_budget = int(observation_token_budget)
        self.observation_detail_token_budget = int(observation_detail_token_budget)
        self.observation_generic_token_budget = int(observation_generic_token_budget)
        self.observation_candidate_memory_token_budget = int(
            observation_candidate_memory_token_budget
        )
        self.observation_search_top_k = int(observation_search_top_k)
        maximum_context_input = (
            self.context_window_tokens
            - self.context_generation_reserve_tokens
            - self.context_safety_margin_tokens
        )
        if not 0 < self.context_input_budget_tokens <= maximum_context_input:
            raise ValueError(
                "context_input_budget_tokens must be positive and fit the model context window"
            )
        self.context_input_budget = self.context_input_budget_tokens
        if self.context_preserve_recent_groups < 1:
            raise ValueError("context_preserve_recent_groups must be positive")
        if min(
            self.observation_token_budget,
            self.observation_detail_token_budget,
            self.observation_generic_token_budget,
            self.observation_candidate_memory_token_budget,
        ) < 64:
            raise ValueError("all observation token budgets must be at least 64")
        if self.observation_search_top_k < 1:
            raise ValueError("observation_search_top_k must be positive")
        if self.reward_mode not in {"native", "constraint_aware"}:
            raise ValueError(f"unknown shopping reward mode: {self.reward_mode!r}")
        self.env_factory = env_factory

    async def _handle_generating_state(
        self,
        agent_data,
        sampling_params,
        ignore_termination=False,
    ):
        """在每次生成前执行上下文预算检查，并限制本轮最大输出长度。"""
        runtime_state = current_runtime_state.get()
        current_input_tokens = len(agent_data.prompt_ids)
        if runtime_state is not None:
            runtime_state["context_max_input_tokens"] = max(
                runtime_state["context_max_input_tokens"],
                current_input_tokens,
            )
        if self.context_compaction_enable:
            try:
                prompt_ids, response_mask, response_logprobs, stats = compact_token_trajectory(
                    agent_data.prompt_ids,
                    agent_data.response_mask,
                    agent_data.response_logprobs,
                    max_input_tokens=self.context_input_budget,
                    preserve_recent_groups=self.context_preserve_recent_groups,
                )
            except ContextBudgetError as exc:
                if runtime_state is not None:
                    runtime_state["terminate"] = True
                    runtime_state["termination_reason"] = "context_budget_exhausted"
                    runtime_state["error"] = f"context_budget_exhausted:{exc}"
                    runtime_state["infrastructure_invalid"] = True
                return AgentState.TERMINATED
        else:
            prompt_ids = agent_data.prompt_ids
            response_mask = agent_data.response_mask
            response_logprobs = agent_data.response_logprobs
            stats = None
        if (
            not self.context_compaction_enable
            and current_input_tokens
            > self.context_window_tokens
            - self.context_generation_reserve_tokens
            - self.context_safety_margin_tokens
        ):
            if runtime_state is not None:
                runtime_state["terminate"] = True
                runtime_state["termination_reason"] = "context_hard_limit_exceeded"
                runtime_state["error"] = runtime_state["termination_reason"]
                runtime_state["infrastructure_invalid"] = True
            return AgentState.TERMINATED
        if stats is not None and stats.removed_tokens:
            # routed-experts 的额外状态无法随 token 一起安全裁剪，因此直接判为无效。
            if agent_data.routed_experts is not None:
                if runtime_state is not None:
                    runtime_state["terminate"] = True
                    runtime_state["termination_reason"] = "context_compaction_unsupported_routed_experts"
                    runtime_state["error"] = runtime_state["termination_reason"]
                    runtime_state["infrastructure_invalid"] = True
                return AgentState.TERMINATED
            agent_data.prompt_ids = prompt_ids
            agent_data.response_mask = response_mask
            agent_data.response_logprobs = response_logprobs
            if runtime_state is not None:
                runtime_state["context_compactions"] += 1
                runtime_state["context_tokens_removed"] += stats.removed_tokens
        bounded_sampling_params = dict(sampling_params)
        if "max_tokens" in bounded_sampling_params:
            bounded_sampling_params["max_tokens"] = min(
                int(bounded_sampling_params["max_tokens"]),
                self.context_generation_reserve_tokens,
            )
        prompt_length_before_generation = len(agent_data.prompt_ids)
        next_state = await super()._handle_generating_state(
            agent_data,
            bounded_sampling_params,
            ignore_termination=ignore_termination,
        )
        if len(agent_data.prompt_ids) == prompt_length_before_generation:
            if runtime_state is not None:
                timed_out = bool(
                    agent_data.extra_fields.get("shopping_generation_timeout")
                )
                runtime_state["terminate"] = True
                runtime_state["termination_reason"] = (
                    "vllm_generation_timeout" if timed_out else "vllm_generation_aborted"
                )
                runtime_state["error"] = runtime_state["termination_reason"]
                runtime_state["infrastructure_invalid"] = True
            return AgentState.TERMINATED
        return next_state

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        """把工具适配器拿到的原始 observation 压缩成模型真正可见的版本。"""
        response, reward, step = await super()._call_tool(
            tool_call,
            tools_kwargs,
            agent_data,
        )
        state = current_runtime_state.get()
        if state is None:
            return response, reward, step
        raw_observation = state.pop("_pending_raw_observation", None)
        if raw_observation is None:
            return response, reward, step
        try:
            # 解析失败或投影破坏动作契约时，停止当前样本而不是把坏 observation
            # 继续喂给模型。
            parameters = json.loads(tool_call.arguments or "{}")
            visible_observation, projection = project_observation(
                tool_name=tool_call.name,
                observation=raw_observation,
                parameters=parameters,
                count_tokens=lambda text: len(
                    self.tokenizer.encode(text, add_special_tokens=False)
                ),
                token_budget=self.observation_token_budget,
                detail_token_budget=self.observation_detail_token_budget,
                generic_token_budget=self.observation_generic_token_budget,
                candidate_memory_token_budget=(
                    self.observation_candidate_memory_token_budget
                ),
                search_top_k=self.observation_search_top_k,
            )
            if len(visible_observation) > self.max_tool_response_length:
                raise ObservationProjectionError(
                    "projected observation exceeds veRL character fallback limit"
                )
        except Exception as exc:
            state["terminate"] = True
            state["termination_reason"] = "observation_projection_failed"
            state["error"] = f"observation_projection_failed:{exc.__class__.__name__}:{exc}"
            state["infrastructure_invalid"] = True
            state["observation_footer_failures"] += 1
            response.text = "Error: observation projection failed; trajectory is invalid."
            return response, 0.0, step

        projection_meta = projection.to_dict()
        state["latest_observation_raw"] = raw_observation
        state["latest_observation"] = visible_observation
        record_observation_projection(state, projection_meta)
        if isinstance(step, dict):
            step["projection"] = projection_meta
        response.text = visible_observation
        return response, reward, step

    async def _handle_processing_tools_state(self, agent_data):
        """与 Evaluation 一致：多工具回合只执行第一个，并记录其余调用。"""
        runtime_state = current_runtime_state.get()
        if len(agent_data.tool_calls) > 1:
            if runtime_state is not None:
                runtime_state["tool_call_truncation_count"] += (
                    len(agent_data.tool_calls) - 1
                )
            agent_data.tool_calls = agent_data.tool_calls[:1]
        next_state = await super()._handle_processing_tools_state(agent_data)
        runtime_state = current_runtime_state.get()
        if runtime_state is not None and runtime_state.get("terminate"):
            return AgentState.TERMINATED
        return next_state

    async def run(self, sampling_params, **kwargs):
        """启动 session、运行父类 AgentLoop，并在 finally 中释放环境租约。"""
        task_id = task_id_from_kwargs(kwargs)
        session = ShopSimulatorSession(
            base_url=self.base_url,
            timeout=self.timeout,
            max_steps=self.max_steps,
            required_environment_version=self.required_environment_version,
            env_factory=self.env_factory,
        )
        state = await session.start(task_id)
        try:
            output = await super().run(sampling_params, **kwargs)
            if (
                not state["done"]
                and not state["infrastructure_invalid"]
                and state.get("termination_reason") == "too_many_guard_rejections"
            ):
                # Repeated illegal actions are a verifiable model failure, not
                # an infrastructure/reward failure. Preserve them as a shaped
                # negative signal so GRPO can learn to recover after a guard
                # rejection instead of silently dropping the whole group.
                state["reward_version"] = "shopsimulator-reward-v4"
                state["reward_type"] = "guard_rejection"
                state["reward_valid"] = True
                state["reward_unverifiable"] = False
                state["termination_reason"] = "guard_rejection"
                state["error"] = None
                state["terminate"] = True
            elif not state["done"] and not state["error"]:
                # Direct natural-language termination is a model error, not an
                # infrastructure failure. Keep it as a valid negative GRPO
                # trajectory so the Actor can learn to call a terminal tool.
                state["reward_version"] = "shopsimulator-reward-v4"
                state["reward_type"] = "assistant_final"
                state["reward_valid"] = True
                state["reward_unverifiable"] = False
                state["termination_reason"] = "assistant_final"
                state["terminate"] = True
            # 父类结束后统一从环境状态结算，避免把中途异常当作正常终局奖励。
            breakdown = reward_breakdown(state)
            output.reward_score = terminal_reward(state, mode=self.reward_mode)
            shopping_info = {
                "task_id": task_id,
                "steps": len(state["steps"]),
                "done": bool(state["done"]),
                "termination_reason": state["termination_reason"],
                "error": state["error"],
                "infrastructure_invalid": bool(state["infrastructure_invalid"]),
                "action_attempts": int(state["action_attempt_count"]),
                "repeat_actions": int(state["repeat_action_count"]),
                "reward_mode": self.reward_mode,
                "reward_version": state.get("reward_version"),
                "reward_type": state.get("reward_type"),
                "reward_valid": bool(state.get("reward_valid", True)),
                "reward_unverifiable": bool(state.get("reward_unverifiable")),
                "reward": breakdown,
                "context_compactions": int(state["context_compactions"]),
                "context_tokens_removed": int(state["context_tokens_removed"]),
                "context_max_input_tokens": int(state["context_max_input_tokens"]),
                "observation_projection_count": int(state["observation_projection_count"]),
                "observation_truncated_count": int(state["observation_truncated_count"]),
                "observation_raw_tokens": int(state["observation_raw_tokens"]),
                "observation_visible_tokens": int(state["observation_visible_tokens"]),
                "observation_max_raw_tokens": int(state["observation_max_raw_tokens"]),
                "observation_max_visible_tokens": int(
                    state["observation_max_visible_tokens"]
                ),
                "observation_visible_asin_count": int(
                    state["observation_visible_asin_count"]
                ),
                "observation_visible_button_count": int(
                    state["observation_visible_button_count"]
                ),
                "observation_any_truncated": bool(state["observation_any_truncated"]),
                "observation_footer_failures": int(state["observation_footer_failures"]),
                "guard_rejections": int(state["guard_rejection_count"]),
                "guard_rejections_after_truncation": int(
                    state["guard_rejection_after_truncation_count"]
                ),
                "action_attempts_after_truncation": int(
                    state["action_attempt_after_truncation_count"]
                ),
                "tool_call_truncations": int(
                    state["tool_call_truncation_count"]
                ),
                "candidate_memory_version": state["candidate_memory"]["version"],
                "candidate_memory_entries": len(
                    state["candidate_memory"]["entries"]
                ),
                "candidate_memory_updates": int(
                    state["candidate_memory"]["updates"]
                ),
                "candidate_memory_search_updates": int(
                    state["candidate_memory"]["search_updates"]
                ),
                "candidate_memory_evictions": int(
                    state["candidate_memory"]["evictions"]
                ),
            }
            # veRL 0.8 only exposes fields nested under reward_extra_info to
            # extract_reward(); keep the flat copy for trajectory diagnostics.
            output.extra_fields["shopping"] = shopping_info
            output.extra_fields.setdefault("reward_extra_info", {})[
                "shopping"
            ] = shopping_info
            print(
                "SHOPPING_TRAJECTORY_TERMINAL "
                + json.dumps(shopping_info, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
            return output
        finally:
            await session.close()
