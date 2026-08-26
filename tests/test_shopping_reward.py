"""Reward v3 diagnostics consumed by the GRPO adapter."""

import unittest

from shopping_grpo.training.grpo.adapter.runtime import (
    make_runtime_state,
    record_action_attempt,
    reward_breakdown,
    terminal_reward,
    validate_reward,
)


def reward_detail(
    reward_type="gold_purchase",
    terminal_utility=1.0,
    *,
    reward_valid=True,
    weighted_score=1.0,
    evidence_coverage=1.0,
    step_count=0,
    step_penalty=0.0,
):
    purchase_success = reward_type in {
        "gold_purchase",
        "valid_alternative_purchase",
    }
    return {
        "reward_version": "shopsimulator-reward-v4",
        "query_constraint_version": "shopping-query-constraints-v1",
        "constraint_results": [],
        "reward_type": reward_type,
        "reward_valid": reward_valid,
        "termination_reason": reward_type,
        "target_asin_match": reward_type == "gold_purchase",
        "hard_gates": {
            "category": {
                "status": "pass",
                "passed": True,
                "verifiable": True,
                "comparator": "category_v1",
                "source_field": "category",
            },
            "budget": {
                "status": "pass",
                "passed": True,
                "verifiable": True,
                "comparator": "budget_v1",
                "source_field": "selected_price",
            },
        },
        "weighted_score": weighted_score,
        "evidence_coverage": evidence_coverage,
        "dimension_scores": {
            "brand": weighted_score,
            "model": weighted_score,
            "core_functions": weighted_score,
            "key_options": weighted_score,
        },
        "terminal_utility": terminal_utility,
        "base_terminal_utility": terminal_utility - step_penalty,
        "step_count": step_count,
        "step_penalty": step_penalty,
        "step_penalty_version": "shopping-step-penalty-v1",
        "purchase_success": purchase_success,
        "sampling_invalid": not reward_valid,
    }


def terminal_state(detail=None):
    detail = validate_reward(detail or reward_detail())
    state = make_runtime_state(task_id=1, max_steps=45)
    state.update(
        {
            "done": True,
            "terminal_result": {"done": True, "over": True},
            "final_reward": detail["terminal_utility"],
            "reward_version": detail["reward_version"],
            "reward_type": detail["reward_type"],
            "reward_valid": detail["reward_valid"],
            "reward_unverifiable": not detail["reward_valid"],
            "reward_detail": detail,
        }
    )
    return state


class ShoppingRewardTest(unittest.TestCase):
    def test_tool_free_assistant_finish_is_valid_negative_reward(self):
        state = make_runtime_state(task_id=1, max_steps=45)
        state["reward_version"] = "shopsimulator-reward-v4"
        state["reward_type"] = "assistant_final"
        state["termination_reason"] = "assistant_final"
        state["terminate"] = True

        result = reward_breakdown(state)

        self.assertEqual(result["total"], -0.8)
        self.assertEqual(result["terminal_utility"], -0.8)
        self.assertEqual(result["penalty_unfinished"], -0.8)
        self.assertFalse(result["sampling_invalid"])
        self.assertFalse(result["infrastructure_invalid"])
        self.assertFalse(result["reward_unverifiable"])
        self.assertEqual(terminal_reward(state, mode="constraint_aware"), -0.8)

    def test_assistant_final_also_receives_cumulative_step_penalty(self):
        state = make_runtime_state(task_id=1, max_steps=45)
        state["steps"] = [{} for _ in range(21)]
        state["reward_version"] = "shopsimulator-reward-v4"
        state["reward_type"] = "assistant_final"
        state["termination_reason"] = "assistant_final"
        state["terminate"] = True

        result = reward_breakdown(state)

        self.assertAlmostEqual(result["penalty_overlong"], -0.07)
        self.assertAlmostEqual(result["terminal_utility"], -0.87)

    def test_consecutive_guard_rejections_are_valid_negative_training_signal(self):
        state = make_runtime_state(task_id=1, max_steps=45)
        state["steps"] = [{} for _ in range(21)]
        state["reward_version"] = "shopsimulator-reward-v4"
        state["reward_type"] = "guard_rejection"
        state["reward_valid"] = True
        state["reward_unverifiable"] = False
        state["termination_reason"] = "guard_rejection"
        state["terminate"] = True

        result = reward_breakdown(state)

        self.assertEqual(result["penalty_unfinished"], -0.8)
        self.assertAlmostEqual(result["penalty_overlong"], -0.07)
        self.assertAlmostEqual(result["terminal_utility"], -0.87)
        self.assertFalse(result["sampling_invalid"])
        self.assertFalse(result["infrastructure_invalid"])
        self.assertFalse(result["reward_unverifiable"])
        self.assertAlmostEqual(
            terminal_reward(state, mode="constraint_aware"), -0.87
        )

    def test_gold_purchase_uses_terminal_utility_without_extra_shaping(self):
        result = reward_breakdown(terminal_state())

        self.assertEqual(result["strict"], 1.0)
        self.assertEqual(result["purchase_success"], 1.0)
        self.assertEqual(result["total"], 1.0)
        self.assertFalse(result["sampling_invalid"])

    def test_step_penalized_terminal_utility_reaches_grpo_unchanged(self):
        detail = reward_detail(
            terminal_utility=0.92,
            step_count=21,
            step_penalty=-0.08,
        )
        state = terminal_state(detail)

        self.assertEqual(state["reward_detail"]["base_terminal_utility"], 1.0)
        self.assertEqual(state["reward_detail"]["step_count"], 21)
        self.assertAlmostEqual(state["reward_detail"]["step_penalty"], -0.08)
        self.assertAlmostEqual(reward_breakdown(state)["total"], 0.92)

    def test_valid_alternative_is_success_but_not_strict_gold(self):
        detail = reward_detail(
            "valid_alternative_purchase",
            0.55,
            weighted_score=1.0,
        )
        result = reward_breakdown(terminal_state(detail))

        self.assertEqual(result["strict"], 0.0)
        self.assertEqual(result["purchase_success"], 1.0)
        self.assertEqual(result["total"], 0.55)

    def test_partial_purchase_preserves_reward_v4_terminal_utility(self):
        detail = reward_detail(
            "partial_alternative_purchase",
            0.10,
            weighted_score=0.73,
            evidence_coverage=0.8,
        )
        result = reward_breakdown(terminal_state(detail))

        self.assertEqual(result["purchase_success"], 0.0)
        self.assertAlmostEqual(result["match_score"], 0.73)
        self.assertAlmostEqual(result["evidence_coverage"], 0.8)
        self.assertAlmostEqual(result["total"], 0.10)

    def test_unverifiable_reward_is_sampling_invalid_and_has_no_learning_signal(self):
        detail = reward_detail(
            "reward_unverifiable",
            0.0,
            reward_valid=False,
            weighted_score=0.0,
            evidence_coverage=0.0,
        )
        result = reward_breakdown(terminal_state(detail))

        self.assertTrue(result["sampling_invalid"])
        self.assertTrue(result["reward_unverifiable"])
        self.assertEqual(result["total"], 0.0)

    def test_reward_v4_contract_rejects_invalid_payloads(self):
        wrong_version = reward_detail()
        wrong_version["reward_version"] = "shopsimulator-reward-v2"
        with self.assertRaisesRegex(ValueError, "reward_version"):
            validate_reward(wrong_version)

        inconsistent = reward_detail("reward_unverifiable", 0.0)
        with self.assertRaisesRegex(ValueError, "reward_valid"):
            validate_reward(inconsistent)

        non_finite = reward_detail()
        non_finite["terminal_utility"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_reward(non_finite)

    def test_same_action_on_same_page_within_three_attempts_is_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=45)

        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")
        record_action_attempt(state, "open_product", {"asin": "123"}, "search page")
        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")

        self.assertEqual(state["action_attempt_count"], 3)
        self.assertEqual(state["repeat_action_count"], 1)
        self.assertAlmostEqual(reward_breakdown(state)["repeat_action_rate"], 1 / 3)

    def test_different_parameters_or_page_are_not_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=45)

        record_action_attempt(state, "search_products", {"query": "mug"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "cup"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "mug"}, "page 2")

        self.assertEqual(state["repeat_action_count"], 0)

    def test_think_is_not_an_environment_action_attempt(self):
        state = make_runtime_state(task_id=1, max_steps=45)

        record_action_attempt(state, "think", {"note": "plan"}, "page")

        self.assertEqual(state["action_attempt_count"], 0)
        self.assertEqual(state["recent_action_signatures"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
