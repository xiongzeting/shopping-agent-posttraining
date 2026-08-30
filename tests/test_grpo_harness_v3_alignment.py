from __future__ import annotations

import asyncio
import unittest

from shopping_grpo.evaluation.rollout import (
    CANDIDATE_PHASE_CHOOSE,
    CANDIDATE_PHASE_SELECT,
    EVALUATION_TOOL_SCHEMAS,
)
from shopping_grpo.training.grpo.adapter.runtime import (
    current_environment,
    current_runtime_state,
    make_runtime_state,
)
from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool


def schema(name):
    return next(
        item for item in EVALUATION_TOOL_SCHEMAS if item["function"]["name"] == name
    )


class FakeEnvironment:
    def __init__(self):
        self.actions = []
        self.reset_no_progress_calls = 0

    def step(self, action):
        self.actions.append(action)
        if action.startswith("reopen["):
            return {
                "instruction": (
                    "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\n"
                    "规格状态: 0 / 1 个规格轴已选择\n"
                    "搜索功能是否可用: False\n"
                    '可点击的按钮: ["opt_0123456789abcdef", "Buy Now"]'
                ),
                "reward": 0.0,
                "done": False,
                "progress": {"no_progress_steps": 0},
            }
        return {
            "instruction": (
                "[SHOPPING_OBSERVATION_V2]\npage_type: search_results\n"
                "搜索功能是否可用: True\n可点击的按钮: []"
            ),
            "reward": 0.0,
            "done": False,
            "progress": {
                "no_progress_steps": 6,
                "candidate_recovery_required": True,
            },
        }

    def reset_no_progress(self):
        self.reset_no_progress_calls += 1
        return {"no_progress_steps": 0, "consecutive_repeats": 0}


class GrpoHarnessV3AlignmentTest(unittest.TestCase):
    def test_runtime_uses_four_stable_candidates(self):
        state = make_runtime_state(task_id=1, max_steps=45)
        self.assertEqual(state["candidate_memory"]["max_entries"], 4)
        self.assertTrue(state["candidate_memory"]["stable_candidate_ids"])
        self.assertIsNone(state["candidate_forced_phase"])

    def test_candidate_recovery_and_phase_gate_match_evaluation(self):
        env = FakeEnvironment()
        state = make_runtime_state(task_id=1, max_steps=45)
        state["latest_observation"] = (
            "[SHOPPING_OBSERVATION_V2]\npage_type: search_home\n"
            "搜索功能是否可用: True\n可点击的按钮: []"
        )
        state["candidate_memory"]["entries"] = [
            {
                "candidate_id": "C1",
                "asin": "123456789012",
                "price": "99",
                "brand": "Example",
                "category": "Example",
                "selected_options": {},
                "title": "Example product",
                "evidence": ["public"],
            }
        ]
        env_token = current_environment.set(env)
        state_token = current_runtime_state.set(state)
        try:
            search = ShopSimulatorTool({}, schema("search_products"))
            response, _, _ = asyncio.run(
                search.execute("search", {"query": "example product"})
            )
            self.assertEqual(state["candidate_forced_phase"], CANDIDATE_PHASE_CHOOSE)
            self.assertIn("page_type: candidate_selection", response.text)
            self.assertEqual(env.reset_no_progress_calls, 1)

            opened = ShopSimulatorTool({}, schema("open_product"))
            response, _, _ = asyncio.run(
                opened.execute("open", {"asin": "123456789012"})
            )
            self.assertEqual(env.actions[-1], "reopen[123456789012]")
            self.assertEqual(state["candidate_forced_phase"], CANDIDATE_PHASE_SELECT)
            self.assertIn("候选规格阶段", response.text)

            blocked = ShopSimulatorTool({}, schema("search_products"))
            response, _, _ = asyncio.run(
                blocked.execute("blocked", {"query": "another product"})
            )
            self.assertIn("action guard rejected", response.text)
            self.assertEqual(len(env.actions), 2)
        finally:
            current_runtime_state.reset(state_token)
            current_environment.reset(env_token)


if __name__ == "__main__":
    unittest.main()
