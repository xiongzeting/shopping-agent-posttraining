import json
import tempfile
import unittest
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

from shopping_grpo.environment.actions import (
    action_guard_tool_message,
    action_reject_reason,
)
from shopping_grpo.environment.client import ShopEnvironmentError
from shopping_grpo.environment.observation import (
    LOOP_RECOVERY_NOTICE_PREFIX,
    STEP_BUDGET_NOTICE_PREFIX,
    add_step_budget_notice,
)
from shopping_grpo.evaluation.rollout import (
    CollectionInfrastructureError,
    EVALUATION_TOOL_SCHEMAS,
    MultiKeyOpenAIChatClient,
    MISSING_TOOL_CALL_CORRECTION,
    OpenAIChatClient,
    SYSTEM_PROMPT,
    collect_tasks,
    collect_for_task,
    completed_task_attempts,
    load_tasks,
    rollout_interrupted,
    _active_tool_schemas,
    CANDIDATE_PHASE_CHOOSE,
    CANDIDATE_PHASE_SELECT,
    CANDIDATE_PHASE_TERMINAL,
    _phase_tool_schemas,
)


class FakeEnv:
    def __init__(self, **kwargs):
        self.actions = []
        self.released = False

    def reset(self, task_id):
        return {"env_idx": 0, "instruction": f"Instruction: task {task_id}"}

    def step(self, action):
        self.actions.append(action)
        if action == "search[乳胶枕]":
            return {
                "instruction": "搜索结果\n1|100000000001|乳胶枕|100.00",
                "reward": 0.0,
                "done": False,
            }
        if action == "click[100000000001]":
            return {
                "instruction": 'detail\n\n可点击的按钮: ["Buy Now"]',
                "reward": 0.0,
                "done": False,
            }
        return {
            "instruction": "done page",
            "reward": 1.0,
            "done": True,
            "over": True,
            "purchase": {"asin": "A1"},
            "reward_detail": {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1},
        }

    def release(self):
        self.released = True


class CandidateConvergenceContractTest(unittest.TestCase):
    def test_candidate_count_does_not_hide_tools(self):
        memory = {
            "entries": [
                {"candidate_id": f"C{index}", "asin": f"{index:012d}"}
                for index in range(1, 4)
            ]
        }

        active = _phase_tool_schemas(
            EVALUATION_TOOL_SCHEMAS,
            12,
            candidate_memory=memory,
        )
        names = {(tool.get("function") or {}).get("name") for tool in active}

        self.assertEqual(
            names,
            {(tool.get("function") or {}).get("name") for tool in EVALUATION_TOOL_SCHEMAS},
        )
        self.assertIn("内部最多保存 4 个", SYSTEM_PROMPT)
        self.assertNotIn("目前已经至少有3个候选", SYSTEM_PROMPT)
        self.assertIn("普通阶段不会展示或开放这些历史候选", SYSTEM_PROMPT)

    def test_forced_candidate_phases_expose_only_the_requested_tools(self):
        observation = (
            "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\n"
            "\n搜索功能是否可用: False\n可点击的按钮: []"
        )

        def names(phase):
            return {
                tool["function"]["name"]
                for tool in _active_tool_schemas(
                    EVALUATION_TOOL_SCHEMAS,
                    12,
                    latest_observation=observation,
                    candidate_memory={"entries": []},
                    candidate_phase=phase,
                )
            }

        self.assertEqual(names(CANDIDATE_PHASE_CHOOSE), {"open_product"})
        self.assertEqual(names(CANDIDATE_PHASE_SELECT), {"select_option"})
        self.assertEqual(
            names(CANDIDATE_PHASE_TERMINAL),
            {"buy_now", "finish_without_purchase"},
        )

    def test_structured_pages_expose_only_page_appropriate_tools(self):
        search_home = (
            "[SHOPPING_OBSERVATION_V2]\npage_type: search_home\n"
            "\n搜索功能是否可用: True\n可点击的按钮: []"
        )
        search_results = (
            "[SHOPPING_OBSERVATION_V2]\npage_type: search_results\n"
            "1|111111111111|100|品牌|类目|属性|商品\n"
            "\n搜索功能是否可用: False\n"
            '可点击的按钮: ["back to search", "next >", "111111111111"]'
        )
        option_id = "opt_0123456789abcdef"
        detail = (
            "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\n"
            "asin: 111111111111\nselected_options: {}\n"
            "\n搜索功能是否可用: False\n"
            f'可点击的按钮: ["back to search", "{option_id}", "buy now"]'
        )

        def names(observation, memory=None):
            return {
                (tool.get("function") or {}).get("name")
                for tool in _active_tool_schemas(
                    EVALUATION_TOOL_SCHEMAS,
                    5,
                    latest_observation=observation,
                    candidate_memory=memory or {"entries": []},
                )
            }

        self.assertEqual(
            names(search_home),
            {"search_products", "finish_without_purchase"},
        )
        self.assertEqual(
            names(search_results),
            {
                "open_product",
                "next_page",
                "back_to_search",
                "finish_without_purchase",
            },
        )
        self.assertEqual(
            names(
                detail,
                {
                    "entries": [
                        {"candidate_id": "C1", "asin": "111111111111"},
                        {"candidate_id": "C2", "asin": "222222222222"},
                    ]
                },
            ),
            {
                "select_option",
                "back_to_search",
                "buy_now",
                "finish_without_purchase",
            },
        )

    def test_candidate_count_keeps_page_tools_available(self):
        option_id = "opt_0123456789abcdef"
        detail = (
            "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\n"
            "asin: 333333333333\nselected_options: {}\n"
            "\n搜索功能是否可用: False\n"
            f'可点击的按钮: ["back to search", "{option_id}", "buy now"]'
        )
        memory = {
            "entries": [
                {"candidate_id": "C1", "asin": "111111111111"},
                {"candidate_id": "C2", "asin": "222222222222"},
                {"candidate_id": "C3", "asin": "333333333333"},
            ]
        }

        active = _active_tool_schemas(
            EVALUATION_TOOL_SCHEMAS,
            8,
            latest_observation=detail,
            candidate_memory=memory,
        )
        names = {(tool.get("function") or {}).get("name") for tool in active}

        self.assertEqual(
            names,
            {
                "select_option",
                "back_to_search",
                "buy_now",
                "finish_without_purchase",
            },
        )


class FailingEnv(FakeEnv):
    def step(self, action):
        self.actions.append(action)
        raise RuntimeError("env exploded")


class NonTerminalEnv(FakeEnv):
    def step(self, action):
        self.actions.append(action)
        return {"instruction": "keep going", "reward": 0.0, "done": False}


class ReleaseFailingEnv(FakeEnv):
    def release(self):
        raise OSError("ShopSimulator unavailable during release")


class UnavailableEnv(FakeEnv):
    def reset(self, task_id):
        raise ShopEnvironmentError(
            "Unable to get available environment resource, please try again later"
        )


class GuardRecoveryEnv(FakeEnv):
    """用于验证非法尝试被合法工具调用隔开时仍可恢复。"""

    def step(self, action):
        self.actions.append(action)
        if action == "search[乳胶枕]":
            return {
                "instruction": "搜索结果\n1|100000000001|乳胶枕|100.00",
                "reward": 0.0,
                "done": False,
            }
        if action == "click[100000000001]":
            return {
                "instruction": 'detail\n\n可点击的按钮: ["Features", "Buy Now"]',
                "reward": 0.0,
                "done": False,
            }
        if action == "click[Features]":
            return {
                "instruction": 'features\n\n可点击的按钮: ["< Prev"]',
                "reward": 0.0,
                "done": False,
            }
        if action == "click[< Prev]":
            return {
                "instruction": 'detail\n\n可点击的按钮: ["Features", "Buy Now"]',
                "reward": 0.0,
                "done": False,
            }
        if action == "click[Buy Now]":
            return {
                "instruction": "done page",
                "reward": 1.0,
                "done": True,
                "over": True,
                "purchase": {"asin": "A1"},
                "reward_detail": {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1},
            }
        raise AssertionError(f"unexpected action: {action}")


class MockClient:
    def __init__(self, messages):
        self.messages = list(messages)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append({"messages": messages, "tools": tools})
        return self.messages.pop(0)


class SnapshotClient(MockClient):
    def complete(self, messages, tools):
        self.requests.append(
            {
                "messages": json.loads(json.dumps(messages, ensure_ascii=False)),
                "tools": json.loads(json.dumps(tools, ensure_ascii=False)),
            }
        )
        return self.messages.pop(0)


class CandidateRecoveryEnv(FakeEnv):
    option_id = "opt_0123456789abcdef"
    asin = "111111111111"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.no_progress = 0
        self.recovery_configured = False
        self.reset_no_progress_calls = 0

    @staticmethod
    def _home_state():
        return {
            "observation_version": "shopping-observation-v2",
            "page_type": "search_home",
            "search_available": True,
            "actions": [],
        }

    @classmethod
    def _search_state(cls):
        return {
            "observation_version": "shopping-observation-v2",
            "page_type": "search_results",
            "search_available": False,
            "actions": ["back to search", cls.asin],
            "query": "候选",
            "normalized_query": "候选",
            "page": 1,
            "total_pages": 1,
            "total_results": 1,
            "rank_start": 1,
            "rank_end": 1,
            "products": [
                {
                    "rank": 1,
                    "asin": cls.asin,
                    "title": "黑色候选商品",
                    "brand": "测试品牌",
                    "category": "测试品类",
                    "price": 100,
                    "key_attributes": ["黑色"],
                }
            ],
        }

    @classmethod
    def _detail_state(cls, *, selected=False):
        actions = ["back to search", cls.option_id, "buy now"]
        selected_options = (
            {"颜色": {"option_id": cls.option_id, "label": "黑色"}}
            if selected
            else {}
        )
        return {
            "observation_version": "shopping-observation-v2",
            "page_type": "product_detail",
            "search_available": False,
            "actions": actions,
            "product": {
                "asin": cls.asin,
                "title": "黑色候选商品",
                "brand": "测试品牌",
                "category": "测试品类",
                "price": 100,
                "key_attributes": ["黑色"],
            },
            "selected_options": selected_options,
            "available_options": {
                "颜色": [{"option_id": cls.option_id, "label": "黑色"}]
            },
        }

    def reset(self, task_id):
        return {
            "env_idx": 0,
            "instruction": f"Instruction: task {task_id}",
            "observation_state": self._home_state(),
        }

    def configure_candidate_recovery(self):
        self.recovery_configured = True
        return {"candidate_recovery_enabled": True}

    def reset_no_progress(self):
        self.no_progress = 0
        self.reset_no_progress_calls += 1
        return {"no_progress_steps": 0, "consecutive_repeats": 0}

    def step(self, action):
        self.actions.append(action)
        if action == "search[候选]":
            return {
                "observation_state": self._search_state(),
                "reward": 0.0,
                "done": False,
                "progress": {"no_progress_steps": 0},
            }
        if action == f"click[{self.asin}]":
            return {
                "observation_state": self._detail_state(),
                "reward": 0.0,
                "done": False,
                "progress": {"no_progress_steps": 0},
            }
        if action == "click[Back to Search]":
            self.no_progress = 1
            return {
                "observation_state": self._home_state(),
                "reward": 0.0,
                "done": False,
                "progress": {"no_progress_steps": self.no_progress},
            }
        if action.startswith("search[q"):
            self.no_progress += 1
            progress = {"no_progress_steps": self.no_progress}
            if self.no_progress >= 6:
                progress["candidate_recovery_required"] = True
            return {
                "observation_state": self._home_state(),
                "reward": 0.0,
                "done": False,
                "progress": progress,
            }
        if action == f"reopen[{self.asin}]":
            return {
                "observation_state": self._detail_state(),
                "reward": 0.0,
                "done": False,
                "progress": {"no_progress_steps": 1},
            }
        if action == f"click[{self.option_id}]":
            return {
                "observation_state": self._detail_state(selected=True),
                "reward": 0.0,
                "done": False,
                "progress": {"no_progress_steps": 0},
            }
        if action == "click[Buy Now]":
            return {
                "instruction": "done",
                "reward": 1.0,
                "done": True,
                "over": True,
                "purchase": {"asin": self.asin},
                "reward_detail": {
                    "reward_type": "gold_purchase",
                    "reward_valid": True,
                },
            }
        raise AssertionError(f"unexpected action: {action}")


def assistant_tool(name, arguments, call_id="call_1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ],
    }


class RolloutTest(unittest.TestCase):
    def test_default_prompt_matches_reward_policy(self):
        """默认提示词应表达 Reward v4 的购买证据和停止门槛。"""
        self.assertIn("单轮购物任务", SYSTEM_PROMPT)
        self.assertIn("不得向用户追问", SYSTEM_PROMPT)
        self.assertIn("有限45步内", SYSTEM_PROMPT)
        self.assertNotIn("第45个普通动作执行后环境会立即 Max-step 终止", SYSTEM_PROMPT)
        self.assertIn("优先购买满足全部要求的商品", SYSTEM_PROMPT)
        self.assertIn("下一次 assistant 行为必须直接调用 `buy_now`", SYSTEM_PROMPT)
        self.assertEqual(SYSTEM_PROMPT.count("`buy_now`"), 1)
        self.assertIn("才可在所有硬约束均满足的候选中选择整体最符合软偏好的商品", SYSTEM_PROMPT)
        self.assertIn("若找不到满足全部硬约束且可接受的商品，应合理结束", SYSTEM_PROMPT)
        self.assertIn("硬约束包括品类，以及用户明确表示必须满足", SYSTEM_PROMPT)
        self.assertIn("无效循环是指重复相同或无实质变化的搜索", SYSTEM_PROMPT)
        self.assertIn("普通阶段不会展示或开放这些历史候选", SYSTEM_PROMPT)
        self.assertIn("最新页面真实可执行的工具", SYSTEM_PROMPT)
        self.assertNotIn("当前候选/步数阶段允许工具", SYSTEM_PROMPT)
        self.assertIn("未出现在本轮列表中的工具当前不可用", SYSTEM_PROMPT)
        self.assertIn("搜索结果页只提供打开当前商品", SYSTEM_PROMPT)
        self.assertIn("商品详情页只提供当前未选规格", SYSTEM_PROMPT)
        self.assertNotIn("每次工具返回后先阅读最新 observation", SYSTEM_PROMPT)
        self.assertIn("6步无进展候选收敛阶段", SYSTEM_PROMPT)
        self.assertIn("候选才会临时显示为同一页可打开商品", SYSTEM_PROMPT)
        self.assertIn("必须用 `open_product` 从该页选择一个候选", SYSTEM_PROMPT)
        self.assertIn("不要重复选择已经在 `selected_options` 中的 option_id", SYSTEM_PROMPT)
        self.assertIn("每个 assistant 回合必须立即且仅返回一个工具调用", SYSTEM_PROMPT)
        self.assertIn("禁止输出自然语言、分析、解释、需求复述或候选清单", SYSTEM_PROMPT)
        self.assertIn("区分不可违反的硬约束（Hard）与可折中的软偏好（Soft）", SYSTEM_PROMPT)
        self.assertIn("品类按商品的实际用途判断", SYSTEM_PROMPT)
        self.assertIn("任何一项可核验 Hard 不满足都不得购买", SYSTEM_PROMPT)
        self.assertIn("“不需要、不用、无需、不要求”通常表示无偏好，不等于禁止", SYSTEM_PROMPT)
        self.assertIn("“绝对不要、不能带、禁止、不得有、不要带有”", SYSTEM_PROMPT)
        self.assertIn("“约、左右、大概、预算”不是最低价格", SYSTEM_PROMPT)
        self.assertIn("按目标价上下20%的范围理解", SYSTEM_PROMPT)
        self.assertIn("1kg=2斤、25kg=50斤、10斤=5kg", SYSTEM_PROMPT)
        self.assertIn("“可搭配X”表示兼容能力", SYSTEM_PROMPT)
        self.assertIn("一份直接、结构化且与当前 variant 对应的证据已经足够", SYSTEM_PROMPT)
        self.assertNotIn("`features`、`attributes`", SYSTEM_PROMPT)
        for tool_name in (
            "view_features",
            "view_description",
            "view_reviews",
            "view_attributes",
        ):
            self.assertNotIn(tool_name, SYSTEM_PROMPT)
        self.assertIn("打开商品并完成必要规格选择后，立即依据当前详情页判断", SYSTEM_PROMPT)
        self.assertIn("下一次 assistant 行为必须直接调用 `buy_now`", SYSTEM_PROMPT)
        self.assertIn("尚有 Hard 不满足时，立刻离开并转向其他候选", SYSTEM_PROMPT)
        self.assertIn("不得继续搜索、比较、重复核验或输出文字", SYSTEM_PROMPT)
        self.assertNotIn("无需寻找任何预设商品", SYSTEM_PROMPT)
        self.assertNotIn("不得接触 Reward、Gold 或隐藏约束解析结果", SYSTEM_PROMPT)
        self.assertNotIn("目标 ASIN", SYSTEM_PROMPT)
        self.assertIn("充分探索后仍无完全匹配商品时", SYSTEM_PROMPT)
        self.assertIn("全部 Hard 满足", SYSTEM_PROMPT)
        self.assertIn("多次有实质差异的搜索和多个候选核验", SYSTEM_PROMPT)
        self.assertIn("没有明显值得继续核验的候选", SYSTEM_PROMPT)
        self.assertNotIn("统一按 `early_abstain` 终止", SYSTEM_PROMPT)
        self.assertIn("不要连续重复同一动作", SYSTEM_PROMPT)
        self.assertIn("必须实质改变品类、品牌、型号、核心功能或规格中的至少一项", SYSTEM_PROMPT)
        self.assertIn("不得只做同义替换、语序调整或增删虚词", SYSTEM_PROMPT)
        self.assertIn("不要调用 `think` 工具", SYSTEM_PROMPT)
        self.assertIn("不要在任务结束前输出最终答复", SYSTEM_PROMPT)
        self.assertNotIn("系统最多再给一次", SYSTEM_PROMPT)

    def test_guard_gives_a_return_only_instruction_on_information_subpage(self):
        """子页误操作后，守卫应明确引导模型先返回，不重复猜测按钮。"""
        message = action_guard_tool_message(
            assistant_tool("view_attributes", {}, "call_attributes"),
            "click_not_in_previous_observation",
            '详情页内容\n\n可点击的按钮: ["< Prev"]',
        )

        self.assertIn("当前页面只允许返回", message["content"])
        self.assertNotIn("信息子页", message["content"])
        self.assertIn("prev_page", message["content"])
        self.assertIn("不要再次提交刚才被拒绝的调用", message["content"])

    def test_hidden_open_product_uses_page_specific_guard_reason(self):
        detail = (
            "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\n"
            '可点击的按钮: ["back to search", "buy now"]'
        )
        active = [
            schema
            for schema in EVALUATION_TOOL_SCHEMAS
            if schema["function"]["name"] in {"back_to_search", "buy_now"}
        ]

        reason = action_reject_reason(
            "open_product",
            {"asin": "333333333333"},
            detail,
            tool_schemas=active,
            evaluation_extensions=True,
        )
        message = action_guard_tool_message(
            assistant_tool(
                "open_product",
                {"asin": "333333333333"},
                "call_historical_product",
            )["tool_calls"][0],
            reason,
            detail,
            tool_schemas=active,
        )

        self.assertEqual(reason, "click_not_in_previous_observation")
        self.assertIn("当前处于商品详情页", message["content"])
        self.assertIn("先调用 back_to_search({})", message["content"])
        self.assertNotIn("unknown_tool", message["content"])

    def test_guard_explains_legal_actions_when_search_is_unavailable(self):
        message = action_guard_tool_message(
            assistant_tool(
                "search_products",
                {"query": "黑色电饭煲"},
                "call_repeat_search",
            )["tool_calls"][0],
            "search_not_available_on_current_page",
            (
                "[SHOPPING_OBSERVATION_V2]\npage_type: search_results\n"
                "1|333333333333|349|品牌|电饭煲|5升|黑色电饭煲\n"
                '可点击的按钮: ["back to search", "333333333333"]'
            ),
        )

        self.assertIn("当前页面不能搜索", message["content"])
        self.assertIn('open_product(asin="333333333333")', message["content"])
        self.assertIn("back_to_search({})", message["content"])

    def test_guard_lists_current_option_after_stale_option_rejection(self):
        current_option = "opt_0123456789abcdef"
        message = action_guard_tool_message(
            assistant_tool(
                "select_option",
                {"value": "opt_ffffffffffffffff"},
                "call_stale_option",
            )["tool_calls"][0],
            "click_not_in_previous_observation",
            (
                "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\n"
                f'可点击的按钮: ["back to search", "{current_option}", "buy now"]'
            ),
        )

        self.assertIn("请求的规格 ID 已过期", message["content"])
        self.assertIn(
            f'select_option(value="{current_option}")',
            message["content"],
        )
        self.assertIn("buy_now({})", message["content"])

    def test_guard_excludes_selected_option_from_recovery_choices(self):
        selected = "opt_1111111111111111"
        unselected = "opt_2222222222222222"
        message = action_guard_tool_message(
            assistant_tool(
                "select_option",
                {"value": selected},
                "call_selected_option",
            )["tool_calls"][0],
            "option_already_selected",
            (
                "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\n"
                f'selected_options: {{"颜色": {{"option_id": "{selected}", "label": "黑色"}}}}\n'
                f'可点击的按钮: ["{selected}", "{unselected}", "buy now"]'
            ),
        )["content"]

        self.assertIn("该规格已经选中，禁止再次选择", message)
        self.assertNotIn(f'select_option(value="{selected}")', message)
        self.assertIn(f'select_option(value="{unselected}")', message)
        self.assertIn("buy_now({})", message)

    def test_collect_for_task_executes_openai_tool_calls_until_done(self):
        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )
        env = FakeEnv()

        traj = collect_for_task(
            {"task_id": 7},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=4,
        )

        self.assertEqual(traj["status"], "done")
        self.assertTrue(traj["trajectory_id"])
        self.assertEqual(env.actions, ["search[乳胶枕]", "click[100000000001]", "click[Buy Now]"])
        self.assertTrue(env.released)
        self.assertEqual(traj["steps"][0]["tool_call"]["function"]["name"], "search_products")
        self.assertEqual(traj["steps"][2]["env_action"], "click[Buy Now]")
        self.assertEqual(traj["terminal_result"]["purchase"]["asin"], "A1")
        self.assertTrue(any(message["role"] == "tool" for message in traj["messages"]))

    def test_collect_for_task_forwards_environment_no_progress_notice(self):
        class NoProgressEnv(FakeEnv):
            def step(self, action):
                self.actions.append(action)
                return {
                    "instruction": (
                        "same page"
                        "\n\n\u641c\u7d22\u529f\u80fd\u662f\u5426\u53ef\u7528: True"
                        "\n\u53ef\u70b9\u51fb\u7684\u6309\u94ae: []"
                    ),
                    "reward": 0.0,
                    "done": False,
                    "progress": {"no_progress_steps": 3},
                }

        client = MockClient(
            [
                assistant_tool(
                    "search_products",
                    {"query": "loop probe"},
                    "call_loop_probe",
                ),
                {"role": "assistant", "content": "stop"},
                {"role": "assistant", "content": "stop"},
                {"role": "assistant", "content": "stop"},
            ]
        )
        env = NoProgressEnv()

        collect_for_task(
            {"task_id": 8},
            client=client,
            env_factory=lambda **kwargs: env,
            max_steps=4,
        )

        latest_tool_message = next(
            message["content"]
            for message in reversed(client.requests[1]["messages"])
            if message.get("role") == "tool"
        )
        self.assertIn(LOOP_RECOVERY_NOTICE_PREFIX, latest_tool_message)

    def test_six_no_progress_steps_force_open_select_then_terminal_tools(self):
        env = CandidateRecoveryEnv()
        client = SnapshotClient(
            [
                assistant_tool("search_products", {"query": "候选"}, "search"),
                assistant_tool("open_product", {"asin": env.asin}, "open"),
                assistant_tool("back_to_search", {}, "back"),
                *[
                    assistant_tool("search_products", {"query": f"q{index}"}, f"q{index}")
                    for index in range(1, 6)
                ],
                assistant_tool("open_product", {"asin": env.asin}, "forced_open"),
                assistant_tool(
                    "select_option",
                    {"value": env.option_id},
                    "forced_select",
                ),
                assistant_tool("buy_now", {}, "forced_buy"),
            ]
        )

        trajectory = collect_for_task(
            {"task_id": 18},
            client=client,
            env_factory=lambda **kwargs: env,
            max_steps=20,
            evaluation_extensions=True,
        )

        self.assertTrue(env.recovery_configured)
        self.assertEqual(env.reset_no_progress_calls, 1)
        self.assertEqual(trajectory["terminal_result"]["reward_detail"]["reward_type"], "gold_purchase")
        self.assertIn(f"reopen[{env.asin}]", env.actions)
        self.assertEqual(
            [tool["function"]["name"] for tool in client.requests[8]["tools"]],
            ["open_product"],
        )
        self.assertEqual(
            [tool["function"]["name"] for tool in client.requests[9]["tools"]],
            ["select_option"],
        )
        self.assertEqual(
            {tool["function"]["name"] for tool in client.requests[10]["tools"]},
            {"buy_now", "finish_without_purchase"},
        )
        for request_index, phase_marker in (
            (8, "page_type: candidate_selection"),
            (9, "候选规格阶段"),
            (10, "候选终局阶段"),
        ):
            request_messages = client.requests[request_index]["messages"]
            self.assertEqual(
                [message["role"] for message in request_messages],
                ["system", "user", "user"],
            )
            self.assertIn("Instruction: task 18", request_messages[1]["content"])
            self.assertIn(phase_marker, request_messages[2]["content"])
            self.assertNotIn("search[q1]", json.dumps(request_messages, ensure_ascii=False))
        self.assertEqual(len(trajectory["candidate_context_resets"]), 3)
        self.assertGreater(len(trajectory["messages"]), 3)
        terminal_message = client.requests[10]["messages"][-1]["content"]
        self.assertIn("满足全部 Hard 且可以接受", terminal_message)
        self.assertIn("存在无法接受的 Hard 违反时", terminal_message)
        self.assertIn("放弃后任务立即失败结束", terminal_message)
        self.assertNotIn("按 repeat_loop 记分", terminal_message)
        forced_message = client.requests[8]["messages"][-1]["content"]
        self.assertIn("候选收敛阶段", forced_message)
        self.assertIn(env.asin, forced_message)
        self.assertIn(f'open_product(asin="{env.asin}")', forced_message)
        self.assertIn("该候选将被锁定", forced_message)
        self.assertIn("不能返回候选页或改选其他候选", forced_message)
        for request in client.requests[:8]:
            visible = "\n".join(
                str(message.get("content") or "")
                for message in request["messages"]
                if message.get("role") == "tool"
            )
            self.assertNotIn("[CANDIDATE_MEMORY_V2]", visible)

    def test_forced_candidate_with_complete_variant_skips_option_phase(self):
        class CompleteVariantCandidateRecoveryEnv(CandidateRecoveryEnv):
            def step(self, action):
                if action == f"reopen[{self.asin}]":
                    self.actions.append(action)
                    return {
                        "observation_state": self._detail_state(selected=True),
                        "reward": 0.0,
                        "done": False,
                        "progress": {"no_progress_steps": 1},
                    }
                return super().step(action)

        env = CompleteVariantCandidateRecoveryEnv()
        client = SnapshotClient(
            [
                assistant_tool("search_products", {"query": "候选"}, "search"),
                assistant_tool("open_product", {"asin": env.asin}, "open"),
                assistant_tool("back_to_search", {}, "back"),
                *[
                    assistant_tool("search_products", {"query": f"q{index}"}, f"q{index}")
                    for index in range(1, 6)
                ],
                assistant_tool("open_product", {"asin": env.asin}, "forced_open"),
                assistant_tool("buy_now", {}, "forced_buy"),
            ]
        )

        trajectory = collect_for_task(
            {"task_id": 21},
            client=client,
            env_factory=lambda **kwargs: env,
            max_steps=20,
            evaluation_extensions=True,
        )

        self.assertEqual(
            trajectory["terminal_result"]["reward_detail"]["reward_type"],
            "gold_purchase",
        )
        self.assertIn(f"reopen[{env.asin}]", env.actions)
        self.assertEqual(
            [tool["function"]["name"] for tool in client.requests[8]["tools"]],
            ["open_product"],
        )
        self.assertEqual(
            {tool["function"]["name"] for tool in client.requests[9]["tools"]},
            {"buy_now", "finish_without_purchase"},
        )
        terminal_message = client.requests[9]["messages"][-1]["content"]
        self.assertIn("规格选择已完成", terminal_message)
        self.assertNotIn("只能调用 select_option", terminal_message)

    def test_six_no_progress_steps_without_candidate_end_as_loop(self):
        class NoCandidateRecoveryEnv(FakeEnv):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.recovery_configured = False

            def reset(self, task_id):
                return {
                    "env_idx": 0,
                    "instruction": f"Instruction: task {task_id}",
                    "observation_state": CandidateRecoveryEnv._home_state(),
                }

            def configure_candidate_recovery(self):
                self.recovery_configured = True

            def step(self, action):
                self.actions.append(action)
                return {
                    "observation_state": CandidateRecoveryEnv._home_state(),
                    "reward": 0.0,
                    "done": False,
                    "progress": {
                        "no_progress_steps": 6,
                        "candidate_recovery_required": True,
                    },
                }

        env = NoCandidateRecoveryEnv()
        client = SnapshotClient(
            [assistant_tool("search_products", {"query": "无候选"}, "search")]
        )

        trajectory = collect_for_task(
            {"task_id": 19},
            client=client,
            env_factory=lambda **kwargs: env,
            max_steps=10,
            evaluation_extensions=True,
        )

        self.assertTrue(env.recovery_configured)
        self.assertTrue(trajectory["done"])
        self.assertEqual(trajectory["terminal_result"]["termination_reason"], "repeat_loop")
        self.assertEqual(
            trajectory["candidate_recovery_events"][0]["outcome"],
            "repeat_loop_without_candidate",
        )

    def test_forced_finish_without_purchase_is_scored_as_loop(self):
        env = CandidateRecoveryEnv()
        client = SnapshotClient(
            [
                assistant_tool("search_products", {"query": "候选"}, "search"),
                assistant_tool("open_product", {"asin": env.asin}, "open"),
                assistant_tool("back_to_search", {}, "back"),
                *[
                    assistant_tool("search_products", {"query": f"q{index}"}, f"q{index}")
                    for index in range(1, 6)
                ],
                assistant_tool("open_product", {"asin": env.asin}, "forced_open"),
                assistant_tool(
                    "select_option",
                    {"value": env.option_id},
                    "forced_select",
                ),
                assistant_tool(
                    "finish_without_purchase",
                    {"reason": "no_suitable_product"},
                    "forced_finish",
                ),
            ]
        )

        trajectory = collect_for_task(
            {"task_id": 20},
            client=client,
            env_factory=lambda **kwargs: env,
            max_steps=20,
            evaluation_extensions=True,
        )

        self.assertEqual(trajectory["terminal_result"]["termination_reason"], "repeat_loop")
        self.assertEqual(
            trajectory["terminal_result"]["reward_detail"]["reward_type"],
            "repeat_loop",
        )
        self.assertNotIn("finish[no_suitable_product]", env.actions)

    def test_late_step_notice_replaces_loop_recovery_notice(self):
        observation = (
            "[SHOPPING_OBSERVATION_V2]\n"
            "page_type: search_results\n"
            "query: 乳胶枕\n"
            "\n搜索功能是否可用: True\n"
            '可点击的按钮: ["next >"]'
        )

        visible = add_step_budget_notice(
            observation,
            step_count=37,
            max_steps=45,
            no_progress_steps=5,
        )

        self.assertIn(STEP_BUDGET_NOTICE_PREFIX, visible)
        self.assertIn("请开始收敛", visible)
        self.assertIn("利用已有核验信息", visible)
        self.assertIn("当前可调用工具仍以最新页面实际暴露的列表为准", visible)
        self.assertNotIn("绝对禁止新搜索", visible)
        self.assertNotIn(LOOP_RECOVERY_NOTICE_PREFIX, visible)

    def test_environment_exposes_finish_without_purchase_to_agent(self):
        class EnvironmentV21(FakeEnv):
            def reset(self, task_id):
                result = super().reset(task_id)
                result["environment_version"] = "shopsimulator-environment-v2.4"
                return result

        client = MockClient(
            [{"role": "assistant", "content": "stop"} for _ in range(3)]
        )
        env = EnvironmentV21()

        collect_for_task(
            {"task_id": 7},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=1,
            evaluation_extensions=True,
        )

        tool_names = [schema["function"]["name"] for schema in client.requests[0]["tools"]]
        self.assertEqual(
            tool_names,
            [schema["function"]["name"] for schema in EVALUATION_TOOL_SCHEMAS],
        )
        self.assertIn("finish_without_purchase", client.requests[0]["messages"][0]["content"])
        self.assertTrue(env.released)

    def test_normal_step_budget_does_not_allow_an_extra_terminal_call(self):
        class HardLimitEnv(FakeEnv):
            def step(self, action):
                self.actions.append(action)
                if action == "search[乳胶枕]":
                    return {
                        "instruction": "搜索结果\n1|100000000001|乳胶枕|100.00",
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[100000000001]":
                    return {
                        "instruction": 'detail\n\n可点击的按钮: ["Buy Now"]',
                        "reward": 0.0,
                        "done": False,
                        "progress": {"termination_reason": "max_steps"},
                    }
                if action == "click[Buy Now]":
                    return {
                        "instruction": "done",
                        "reward": 1.0,
                        "done": True,
                        "purchase": {"asin": "100000000001"},
                    }
                raise AssertionError(action)

        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "open"),
                assistant_tool("buy_now", {}, "grace_buy"),
            ]
        )
        env = HardLimitEnv()

        trajectory = collect_for_task(
            {"task_id": 18},
            client=client,
            env_factory=lambda **kwargs: env,
            max_steps=2,
            evaluation_extensions=True,
        )

        self.assertEqual(trajectory["status"], "max_steps")
        self.assertEqual(len(trajectory["steps"]), 2)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(env.actions, ["search[乳胶枕]", "click[100000000001]"])
        self.assertNotIn("terminal_grace", trajectory)

    def test_collect_for_task_blocks_invalid_click_then_keeps_clean_recovery(self):
        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("view_features", {}, "call_invalid"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )
        env = FakeEnv()

        traj = collect_for_task(
            {"task_id": 10},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=5,
        )

        self.assertEqual(traj["status"], "done")
        self.assertEqual(env.actions, ["search[乳胶枕]", "click[100000000001]", "click[Buy Now]"])
        self.assertEqual(len(traj["blocked_tool_calls"]), 1)
        self.assertEqual(
            traj["blocked_tool_calls"][0]["reason"], "unknown_tool"
        )
        self.assertNotIn(
            "view_features",
            [step["tool_name"] for step in traj["steps"]],
        )
        self.assertTrue(
            any(
                message.get("role") == "tool"
                and message.get("tool_call_id") == "call_invalid"
                and message.get("runtime_action_guard") is True
                for message in traj["messages"]
            )
        )

    def test_collect_for_task_rejects_removed_view_tool_after_option_selection(self):
        """即使旧页面显示 Description，已删除的 view 工具也不能执行。"""

        class OptionEnv(FakeEnv):
            def step(self, action):
                self.actions.append(action)
                if action == "search[乳胶枕]":
                    return {
                        "instruction": "搜索结果\n1|100000000001|乳胶枕|100.00",
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[100000000001]":
                    return {
                        "instruction": 'detail\n\n可点击的按钮: ["opt_0123456789abcdef", "Description", "Buy Now"]',
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[opt_0123456789abcdef]":
                    return {
                        "instruction": 'selected\n\n可点击的按钮: ["Description", "Buy Now"]',
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[Description]":
                    return {
                        "instruction": 'details\n\n可点击的按钮: ["Buy Now"]',
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[Buy Now]":
                    return {
                        "instruction": "done",
                        "reward": 1.0,
                        "done": True,
                        "over": True,
                        "purchase": {"asin": "A1"},
                        "reward_detail": {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1},
                    }
                raise AssertionError(action)

        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("select_option", {"value": "opt_0123456789abcdef"}, "call_option"),
                assistant_tool("view_description", {}, "call_wrong_navigation"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )
        env = OptionEnv()

        traj = collect_for_task({"task_id": 12}, client=client, env_factory=lambda **kwargs: env)

        self.assertEqual(traj["status"], "done")
        self.assertEqual(
            [step["env_action"] for step in traj["steps"]],
            [
                "search[乳胶枕]",
                "click[100000000001]",
                "click[opt_0123456789abcdef]",
                "click[Buy Now]",
            ],
        )
        self.assertEqual(len(traj["blocked_tool_calls"]), 1)
        self.assertEqual(traj["blocked_tool_calls"][0]["reason"], "unknown_tool")

    def test_collect_for_task_corrects_missing_tool_call_without_consuming_a_step(self):
        class BuyReadyEnv(FakeEnv):
            def reset(self, task_id):
                return {
                    "env_idx": 0,
                    "instruction": 'detail\n\n可点击的按钮: ["Buy Now"]',
                }

        client = MockClient(
            [
                {"role": "assistant", "content": "还需要分析一下"},
                assistant_tool("buy_now", {}, "call_buy_after_correction"),
            ]
        )
        env = BuyReadyEnv()

        trajectory = collect_for_task(
            {"task_id": 14},
            client=client,
            env_factory=lambda **kwargs: env,
        )

        self.assertEqual(trajectory["status"], "done")
        self.assertEqual(len(trajectory["steps"]), 1)
        self.assertEqual(env.actions, ["click[Buy Now]"])
        self.assertEqual(len(trajectory["missing_tool_call_corrections"]), 1)
        correction_messages = [
            message
            for message in client.requests[1]["messages"]
            if message.get("role") == "user"
            and message.get("content") == MISSING_TOOL_CALL_CORRECTION
        ]
        self.assertEqual(len(correction_messages), 1)
        self.assertNotIn("768", MISSING_TOOL_CALL_CORRECTION)
        self.assertNotIn("截断", MISSING_TOOL_CALL_CORRECTION)

    def test_collect_for_task_marks_assistant_final_after_two_corrections_fail(self):
        client = MockClient(
            [
                {"role": "assistant", "content": "分析一"},
                {"role": "assistant", "content": "分析二"},
                {"role": "assistant", "content": "分析三"},
            ]
        )
        env = FakeEnv()

        trajectory = collect_for_task(
            {"task_id": 15},
            client=client,
            env_factory=lambda **kwargs: env,
        )

        self.assertEqual(trajectory["status"], "assistant_final")
        self.assertTrue(trajectory["done"])
        self.assertEqual(len(trajectory["steps"]), 0)
        self.assertEqual(len(trajectory["missing_tool_call_corrections"]), 2)
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(
            [
                message["content"]
                for message in trajectory["messages"]
                if message.get("role") == "user"
                and message.get("content") == MISSING_TOOL_CALL_CORRECTION
            ],
            [MISSING_TOOL_CALL_CORRECTION, MISSING_TOOL_CALL_CORRECTION],
        )

    def test_collect_for_task_keeps_one_tool_schema_after_option_selection(self):
        """所有阶段使用同一工具 schema，环境 observation 是唯一动作边界。"""

        class OptionEnv(FakeEnv):
            def step(self, action):
                self.actions.append(action)
                if action == "search[乳胶枕]":
                    return {
                        "instruction": "搜索结果\n1|100000000001|乳胶枕|100.00",
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[100000000001]":
                    return {
                        "instruction": 'detail\n\n可点击的按钮: ["opt_0123456789abcdef", "Buy Now"]',
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[opt_0123456789abcdef]":
                    return {
                        "instruction": 'selected\n\n可点击的按钮: ["Buy Now"]',
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[Buy Now]":
                    return {
                        "instruction": "done",
                        "reward": 1.0,
                        "done": True,
                        "over": True,
                        "purchase": {"asin": "A1"},
                        "reward_detail": {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1},
                    }
                raise AssertionError(action)

        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("select_option", {"value": "opt_0123456789abcdef"}, "call_option"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )

        trajectory = collect_for_task(
            {"task_id": 13}, client=client, env_factory=OptionEnv, base_url="http://shop.test"
        )

        self.assertEqual(trajectory["status"], "done")
        exposed_after_selection = [
            schema["function"]["name"] for schema in client.requests[3]["tools"]
        ]
        for removed_name in (
            "view_features",
            "view_description",
            "view_reviews",
            "view_attributes",
        ):
            self.assertNotIn(removed_name, exposed_after_selection)
        self.assertIn("search_products", exposed_after_selection)

    def test_collect_for_task_allows_recovery_after_separated_guard_rejections(self):
        """合法动作应重置守卫计数，避免累计三次历史点击提前中止。"""
        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "999999999999"}, "call_old_asin"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("view_attributes", {}, "call_missing_attributes"),
                assistant_tool("view_features", {}, "call_features"),
                assistant_tool("buy_now", {}, "call_buy_on_subpage"),
                assistant_tool("prev_page", {}, "call_return"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )
        env = GuardRecoveryEnv()

        traj = collect_for_task(
            {"task_id": 11},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=8,
        )

        self.assertEqual(traj["status"], "done")
        self.assertEqual(len(traj["blocked_tool_calls"]), 3)
        self.assertEqual(
            env.actions,
            [
                "search[乳胶枕]",
                "click[100000000001]",
                "click[Buy Now]",
            ],
        )

    def test_collect_for_task_settles_three_consecutive_guard_rejections(self):
        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("view_description", {}, "call_bad_1"),
                assistant_tool("view_features", {}, "call_bad_2"),
                assistant_tool("view_attributes", {}, "call_bad_3"),
            ]
        )
        env = FakeEnv()

        trajectory = collect_for_task(
            {"task_id": 16},
            client=client,
            env_factory=lambda **kwargs: env,
        )

        self.assertEqual(trajectory["status"], "invalid_action_limit")
        self.assertTrue(trajectory["done"])
        self.assertEqual(len(trajectory["steps"]), 1)
        self.assertEqual(len(trajectory["blocked_tool_calls"]), 3)
        self.assertEqual(trajectory["final_reward"], -0.8)
        self.assertEqual(
            trajectory["terminal_result"]["reward_detail"]["reward_type"],
            "guard_rejection",
        )
        self.assertEqual(
            trajectory["terminal_result"]["termination_reason"],
            "invalid_action_limit",
        )

    def test_collect_for_task_keeps_exception_trajectory_and_releases_env(self):
        client = MockClient([assistant_tool("search_products", {"query": "乳胶枕"})])
        env = FailingEnv()

        traj = collect_for_task(
            {"task_id": 8},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=2,
        )

        self.assertEqual(traj["status"], "error")
        self.assertIn("env exploded", traj["error"]["message"])
        self.assertEqual(traj["steps"][0]["env_action"], "search[乳胶枕]")
        self.assertTrue(env.released)

    def test_rollout_interrupted_raises_keyboard_interrupt_for_finally_release(self):
        with self.assertRaises(KeyboardInterrupt):
            rollout_interrupted(None, None)

    def test_completed_task_attempts_defaults_missing_attempt_index_to_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            output.write_text(
                json.dumps({"task_id": 1, "trajectory_id": "old"})
                + "\n"
                + json.dumps({"task_id": 3, "trajectory_id": "old2"})
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(completed_task_attempts(output), {(1, 0), (3, 0)})

    def test_collect_tasks_skips_existing_output_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            output.write_text(json.dumps({"task_id": 1, "trajectory_id": "old"}) + "\n")
            client = MockClient([assistant_tool("buy_now", {}, "call_buy")])

            written = collect_tasks(
                [{"task_id": 1}, {"task_id": 2}],
                client=client,
                output_path=output,
                base_url="http://shop.test",
                max_steps=1,
                env_factory=FakeEnv,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["task_id"] for row in rows], [1, 2])
        self.assertEqual(len(written), 1)

    def test_collect_tasks_uses_the_task_specific_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            client = MockClient(
                [
                    assistant_tool("buy_now", {}, "call_1"),
                    assistant_tool("buy_now", {}, "call_2"),
                ]
            )

            collect_tasks(
                [{"task_id": 1}, {"task_id": 2}],
                client=client,
                output_path=output,
                base_url="http://shop.test",
                max_steps=1,
                env_factory=FakeEnv,
                system_prompt_factory=lambda task: f"Teacher prompt {task['task_id']}",
            )

        prompts = [request["messages"][0]["content"] for request in client.requests]
        prompt_changes = [
            prompt
            for index, prompt in enumerate(prompts)
            if index == 0 or prompt != prompts[index - 1]
        ]
        self.assertEqual(prompt_changes, ["Teacher prompt 1", "Teacher prompt 2"])

    def test_collect_tasks_resumes_missing_attempts_for_each_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            output.write_text(
                json.dumps({"task_id": 1, "attempt_index": 0, "trajectory_id": "old"}) + "\n",
                encoding="utf-8",
            )
            client = MockClient(
                [
                    assistant_tool("buy_now", {}, "call_1"),
                    assistant_tool("buy_now", {}, "call_2"),
                    assistant_tool("buy_now", {}, "call_3"),
                ]
            )

            written = collect_tasks(
                [{"task_id": 1}, {"task_id": 2}],
                client=client,
                output_path=output,
                base_url="http://shop.test",
                max_steps=1,
                env_factory=FakeEnv,
                attempts_per_task=2,
            )

        self.assertEqual(
            [(row["task_id"], row["attempt_index"]) for row in written],
            [(1, 1), (2, 0), (2, 1)],
        )

    def test_collect_tasks_can_append_a_second_probe_round(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            client = MockClient(
                [assistant_tool("buy_now", {}, f"call_{index}") for index in range(4)]
            )
            written = collect_tasks(
                [{"task_id": 1}],
                client=client,
                output_path=output,
                base_url="http://shop.test",
                max_steps=1,
                env_factory=FakeEnv,
                attempts_per_task=4,
                attempt_start=4,
            )
        self.assertEqual([row["attempt_index"] for row in written], [4, 5, 6, 7])

    def test_collect_tasks_can_use_attempt_specific_clients(self):
        seen = []

        def client_factory(task, attempt_index):
            seen.append((task["task_id"], attempt_index))
            return MockClient([assistant_tool("buy_now", {}, f"call_{attempt_index}")])

        with tempfile.TemporaryDirectory() as tmpdir:
            collect_tasks(
                [{"task_id": 3}],
                client=None,
                client_factory=client_factory,
                output_path=Path(tmpdir) / "raw.jsonl",
                base_url="http://shop.test",
                max_steps=1,
                env_factory=FakeEnv,
                attempts_per_task=2,
            )
        self.assertEqual(seen, [(3, 0), (3, 1)])

    def test_collect_tasks_supports_concurrent_workers_and_persists_each_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            client = MockClient(
                [assistant_tool("buy_now", {}, f"call_{index}") for index in range(4)]
            )

            written = collect_tasks(
                [{"task_id": task_id} for task_id in range(1, 5)],
                client=client,
                output_path=output,
                base_url="http://shop.test",
                max_steps=1,
                env_factory=FakeEnv,
                workers=2,
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual({row["task_id"] for row in written}, {1, 2, 3, 4})
        self.assertEqual({row["task_id"] for row in rows}, {1, 2, 3, 4})
        self.assertEqual(len(rows), 4)

    def test_collect_tasks_stops_after_a_release_failure(self):
        """环境租约未释放时，不能继续消耗后续 task。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            client = MockClient(
                [{"role": "assistant", "content": "stop"} for _ in range(3)]
            )

            with self.assertRaises(CollectionInfrastructureError):
                collect_tasks(
                    [{"task_id": 1}, {"task_id": 2}],
                    client=client,
                    output_path=output,
                    base_url="http://shop.test",
                    env_factory=ReleaseFailingEnv,
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["task_id"] for row in rows], [1])
        self.assertEqual(rows[0]["status"], "environment_release_failed")
        self.assertEqual(rows[0]["release_error"]["type"], "OSError")

    def test_collect_tasks_stops_after_environment_resource_is_unavailable(self):
        """服务报告无可用环境时，不能把其余 task 误记成失败轨迹。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            client = MockClient([])

            with self.assertRaises(CollectionInfrastructureError):
                collect_tasks(
                    [{"task_id": 1}, {"task_id": 2}],
                    client=client,
                    output_path=output,
                    base_url="http://shop.test",
                    env_factory=UnavailableEnv,
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["task_id"] for row in rows], [1])
        self.assertEqual(rows[0]["error"]["type"], "ShopEnvironmentError")

    def test_collect_tasks_stops_after_model_api_is_unreachable(self):
        """模型 API 重试后仍断线时，不能把后续 task 批量记成失败。"""

        class DisconnectedClient:
            def complete(self, messages, tools):
                raise URLError("Remote end closed connection without response")

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"

            with self.assertRaises(CollectionInfrastructureError):
                collect_tasks(
                    [{"task_id": 1}, {"task_id": 2}],
                    client=DisconnectedClient(),
                    output_path=output,
                    base_url="http://shop.test",
                    env_factory=FakeEnv,
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["task_id"] for row in rows], [1])
        self.assertEqual(rows[0]["error"]["type"], "URLError")

    def test_collect_for_task_serializes_multiple_tool_calls_before_execution(self):
        client = MockClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": "search_products",
                                "arguments": json.dumps(
                                    {"query": f"乳胶枕{index}"}, ensure_ascii=False
                                ),
                            },
                        }
                        for index in range(3)
                    ],
                },
                assistant_tool(
                    "search_products", {"query": "第二次观察后搜索"}, "call_after_observation"
                ),
            ]
        )
        env = NonTerminalEnv()

        traj = collect_for_task(
            {"task_id": 9},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=2,
        )

        self.assertEqual(traj["status"], "max_steps")
        self.assertEqual(len(traj["steps"]), 2)
        self.assertEqual(env.actions, ["search[乳胶枕0]", "search[第二次观察后搜索]"])
        self.assertEqual(len(traj["messages"][2]["tool_calls"]), 1)
        self.assertEqual(
            [call["id"] for call in traj["tool_call_truncations"][0]["dropped_tool_calls"]],
            ["call_1", "call_2"],
        )

    def test_load_tasks_reads_interaction_task_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = Path(tmpdir) / "tasks.jsonl"
            tasks.write_text(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": "hello"}],
                        "extra_info": {"interaction_kwargs": {"task_id": 42}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = load_tasks(tasks)

        self.assertEqual(rows[0]["task_id"], 42)
        self.assertEqual(rows[0]["prompt"][0]["content"], "hello")

    def test_openai_client_sends_standard_tool_payload(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update(
                {"url": url, "payload": payload, "headers": headers, "timeout": timeout}
            )
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="deepseek-chat",
            base_url="https://api.example.test/v1",
            api_key="secret",
            temperature=0.2,
            top_p=0.9,
            timeout=12,
            transport=transport,
        )

        message = client.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

        self.assertEqual(message["content"], "ok")
        self.assertEqual(captured["url"], "https://api.example.test/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "deepseek-chat")
        self.assertEqual(captured["payload"]["tools"], [{"type": "function"}])
        self.assertEqual(captured["payload"]["max_tokens"], 768)
        self.assertEqual(captured["payload"]["temperature"], 0.2)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(
            captured["headers"]["User-Agent"],
            "shopping-grpo/0.1 (OpenAI-compatible evaluation)",
        )

    def test_openai_client_allows_bounded_completion_override(self):
        """本地推理服务必须收到单次生成上限，避免无工具文本耗尽上下文。"""
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="Qwen/Qwen3.5-2B",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            max_tokens=256,
            transport=transport,
        )

        client.complete([{"role": "user", "content": "hi"}], tools=[])

        self.assertEqual(captured["payload"]["max_tokens"], 256)

    def test_openai_client_retries_missing_required_tool_call(self):
        payloads = []

        def transport(url, payload, headers, timeout):
            payloads.append(payload)
            if len(payloads) == 1:
                return {
                    "choices": [{"message": {"role": "assistant", "content": ""}}],
                    "usage": {"completion_tokens": 768},
                }
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "buy_now", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"completion_tokens": 32},
            }

        client = OpenAIChatClient(
            model="deepseek-v4-flash",
            base_url="https://api.example.test/v1",
            api_key="secret",
            tool_choice="required",
            missing_tool_call_retries=2,
            transport=transport,
        )

        message = client.complete([{"role": "user", "content": "buy"}], tools=[])

        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["tool_choice"], "required")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "buy_now")
        self.assertEqual(client.last_call_metrics["missing_tool_call_retries"], 1)

    def test_multikey_client_keeps_keys_inside_isolated_slots(self):
        seen_keys = []

        def transport(url, payload, headers, timeout):
            seen_keys.append(headers["Authorization"])
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "buy_now", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            }

        client = MultiKeyOpenAIChatClient(
            api_keys=["first", "second"],
            per_key_concurrency=1,
            client_kwargs={
                "model": "deepseek-v4-flash",
                "base_url": "https://api.example.test/v1",
                "tool_choice": "required",
                "transport": transport,
            },
        )

        client.complete([{"role": "user", "content": "buy"}], tools=[])
        client.complete([{"role": "user", "content": "buy"}], tools=[])

        self.assertEqual(seen_keys, ["Bearer first", "Bearer second"])

    def test_openai_client_compacts_old_complete_tool_groups_before_request(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "task"},
            assistant_tool("search_products", {"query": "old"}, "old"),
            {
                "role": "tool",
                "tool_call_id": "old",
                "name": "search_products",
                "content": "old page",
            },
            assistant_tool("search_products", {"query": "middle"}, "middle"),
            {
                "role": "tool",
                "tool_call_id": "middle",
                "name": "search_products",
                "content": "middle page",
            },
            assistant_tool("search_products", {"query": "latest"}, "latest"),
            {
                "role": "tool",
                "tool_call_id": "latest",
                "name": "search_products",
                "content": "latest page",
            },
        ]
        client = OpenAIChatClient(
            model="shopping",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            max_tokens=2,
            context_window=9,
            context_safety_margin=1,
            context_compaction_enable=True,
            token_counter=lambda candidate, tools: len(candidate),
            transport=transport,
        )

        client.complete(messages, tools=[])

        self.assertNotIn("old page", str(captured["payload"]["messages"]))
        self.assertIn("middle page", str(captured["payload"]["messages"]))
        self.assertIn("latest page", str(captured["payload"]["messages"]))
        self.assertEqual(client.last_context_event["removed_groups"], 1)
        self.assertEqual(messages[3]["content"], "old page")

    def test_openai_client_projects_tool_observation_with_serving_tokenizer(self):
        client = OpenAIChatClient(
            model="shopping",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            observation_token_budget=128,
            observation_generic_token_budget=128,
            observation_token_counter=len,
        )
        raw = (
            "Description "
            + "x" * 200
            + "\n\n搜索功能是否可用: False"
            + '\n\n可点击的按钮: ["back to search", "< prev"]'
        )

        visible, meta = client.project_observation("view_description", raw, {})

        self.assertLessEqual(len(visible), 128)
        self.assertTrue(meta["truncated"])
        self.assertTrue(meta["critical_footer_preserved"])

    def test_openai_client_thinking_mode_keeps_reasoning_for_tool_follow_up(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "先核对规格，再搜索。",
                            "tool_calls": [
                                {
                                    "id": "call_search",
                                    "type": "function",
                                    "function": {
                                        "name": "search_products",
                                        "arguments": '{"query":"乳胶枕"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        client = OpenAIChatClient(
            model="deepseek-v4-flash",
            base_url="https://api.example.test/v1",
            api_key="secret",
            thinking=True,
            reasoning_effort="high",
            transport=transport,
        )

        message = client.complete(
            [{"role": "user", "content": "买乳胶枕"}], tools=[{"type": "function"}]
        )

        self.assertEqual(captured["payload"]["thinking"], {"type": "enabled"})
        self.assertEqual(captured["payload"]["reasoning_effort"], "high")
        self.assertNotIn("temperature", captured["payload"])
        self.assertNotIn("top_p", captured["payload"])
        self.assertEqual(message["reasoning_content"], "先核对规格，再搜索。")

    def test_openai_client_uses_low_reasoning_and_single_tool_calls_on_opencode_go(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="deepseek-v4-flash",
            base_url="https://opencode.ai/zen/go/v1",
            api_key="secret",
            thinking=False,
            reasoning_effort="max",
            transport=transport,
        )

        client.complete([{"role": "user", "content": "继续"}], tools=[])

        self.assertEqual(captured["payload"]["reasoning"], {"effort": "low"})
        self.assertFalse(captured["payload"]["parallel_tool_calls"])
        self.assertNotIn("thinking", captured["payload"])
        self.assertNotIn("reasoning_effort", captured["payload"])
        self.assertEqual(captured["payload"]["temperature"], 0.0)
        self.assertEqual(captured["payload"]["top_p"], 1.0)

    def test_openai_client_explicitly_disables_deepseek_v4_thinking_off_go(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="deepseek-v4-flash",
            base_url="https://opencode.ai/zen/v1",
            api_key="secret",
            thinking=False,
            transport=transport,
        )

        client.complete([{"role": "user", "content": "继续"}], tools=[])

        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning", captured["payload"])
        self.assertNotIn("parallel_tool_calls", captured["payload"])

    def test_openai_client_does_not_send_thinking_to_local_non_deepseek_model(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="Qwen/Qwen3.5-2B",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            thinking=False,
            transport=transport,
        )

        client.complete([{"role": "user", "content": "继续"}], tools=[])

        self.assertNotIn("thinking", captured["payload"])
        self.assertNotIn("reasoning_effort", captured["payload"])

    def test_openai_client_retries_transient_disconnect_without_replaying_tools(self):
        attempts = []

        def transport(url, payload, headers, timeout):
            attempts.append(payload)
            if len(attempts) == 1:
                raise RemoteDisconnected("connection closed")
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="deepseek-v4-pro",
            base_url="https://api.example.test/v1",
            api_key="secret",
            transport=transport,
        )

        with patch("shopping_grpo.evaluation.rollout.time.sleep") as sleep:
            message = client.complete([{"role": "user", "content": "继续"}], tools=[])

        self.assertEqual(message["content"], "ok")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])
        sleep.assert_called_once()

    def test_openai_client_retries_incomplete_chunked_response(self):
        attempts = []

        def transport(url, payload, headers, timeout):
            attempts.append(payload)
            if len(attempts) == 1:
                raise IncompleteRead(b"")
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="deepseek-v4-flash",
            base_url="https://api.example.test/v1",
            api_key="secret",
            transport=transport,
        )

        with patch("shopping_grpo.evaluation.rollout.time.sleep") as sleep:
            message = client.complete([{"role": "user", "content": "继续"}], tools=[])

        self.assertEqual(message["content"], "ok")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
