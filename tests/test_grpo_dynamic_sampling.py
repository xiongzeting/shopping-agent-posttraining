from __future__ import annotations

import ast
import unittest
from pathlib import Path

from shopping_grpo.training.grpo.dynamic_sampling import select_reward_varying_groups


class DynamicSamplingSelectionTest(unittest.TestCase):
    def test_empty_result_keeps_rollout_replicas_awake(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "shopping_grpo"
            / "training"
            / "grpo"
            / "verl_dynamic_sampling.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        matching_finally_blocks = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try) or not node.finalbody:
                continue
            if any(
                isinstance(child, ast.Attribute)
                and child.attr == "sleep_replicas"
                for child in ast.walk(node)
            ):
                matching_finally_blocks.append(node.finalbody)
        self.assertEqual(len(matching_finally_blocks), 1)
        finalbody = matching_finally_blocks[0]
        self.assertEqual(len(finalbody), 1)
        self.assertIsInstance(finalbody[0], ast.If)
        self.assertIsInstance(finalbody[0].test, ast.Name)
        self.assertEqual(finalbody[0].test.id, "kept_batches")

    def test_drops_near_constant_group_within_tolerance(self):
        selected, stats = select_reward_varying_groups(
            ["a"] * 4,
            [0.10, 0.11, 0.09, 0.10],
            terminal_utilities=[0.10, 0.11, 0.09, 0.10],
            tolerance=0.021,
        )
        self.assertEqual(selected, [])
        self.assertEqual(stats["all_equal_group_count"], 1)
        self.assertEqual(stats["low_reward_variation_group_count"], 1)
        self.assertEqual(stats["groups"][0]["drop_reason"], "low_reward_variation")
        self.assertAlmostEqual(stats["groups"][0]["reward_range"], 0.02)

    def test_keeps_reward_varying_group_and_drops_invalid_group(self):
        selected, stats = select_reward_varying_groups(
            ["varying"] * 4 + ["invalid"] * 4,
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            terminal_utilities=[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            sampling_invalid=[False] * 4 + [False, True, False, False],
        )
        self.assertEqual(selected, [0, 1, 2, 3])
        self.assertEqual(stats["kept_group_count"], 1)
        self.assertEqual(stats["sampling_invalid_group_count"], 1)

    def test_all_unverifiable_trajectories_form_a_safe_empty_group(self):
        selected, stats = select_reward_varying_groups(
            ["forbidden"] * 4,
            [0.0] * 4,
            terminal_utilities=[0.0] * 4,
            purchase_success=[False] * 4,
            sampling_invalid=[True] * 4,
            sampling_invalid_reasons=[("reward_unverifiable",)] * 4,
            require_purchase_success=True,
        )
        self.assertEqual(selected, [])
        self.assertEqual(stats["groups"][0]["terminal_utilities"], ())
        self.assertEqual(stats["groups"][0]["drop_reason"], "forbidden_trajectory_in_group")
        self.assertFalse(stats["groups"][0]["retryable"])
        self.assertEqual(stats["all_zero_utility_group_count"], 0)
        self.assertEqual(stats["all_purchase_success_group_count"], 0)
        self.assertEqual(stats["no_purchase_success_group_count"], 1)

    def test_requires_a_gold_or_valid_purchase_when_enabled(self):
        selected, stats = select_reward_varying_groups(
            ["failures"] * 4 + ["frontier"] * 4,
            [-0.85, -0.65, -0.50, -0.30, -0.85, -0.65, 1.0, -0.30],
            terminal_utilities=[-0.85, -0.65, -0.50, -0.30, -0.85, -0.65, 1.0, -0.30],
            purchase_success=[False] * 4 + [False, False, True, False],
            require_purchase_success=True,
        )
        self.assertEqual(selected, [4, 5, 6, 7])
        self.assertEqual(stats["no_purchase_success_group_count"], 1)
        self.assertEqual(stats["groups"][0]["drop_reason"], "no_purchase_success")

    def test_assistant_final_is_kept_as_negative_training_signal(self):
        selected, stats = select_reward_varying_groups(
            ["assistant"] * 4,
            [-0.4, 1.0, -0.3, -0.65],
            terminal_utilities=[-0.4, 1.0, -0.3, -0.65],
            purchase_success=[False, True, False, False],
            sampling_invalid=[False] * 4,
            sampling_invalid_reasons=[("assistant_final",), (), (), ()],
            require_purchase_success=True,
        )
        self.assertEqual(selected, [0, 1, 2, 3])
        self.assertEqual(stats["groups"][0]["invalid_trajectory_count"], 0)
        self.assertFalse(stats["groups"][0]["discarded_invalid_trajectories"])
        self.assertTrue(stats["groups"][0]["assistant_final"])
        self.assertIsNone(stats["groups"][0]["drop_reason"])


if __name__ == "__main__":
    unittest.main()
