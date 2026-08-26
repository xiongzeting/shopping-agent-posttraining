import unittest

from shopping_grpo.training.grpo.probe_gates import (
    decide_grpo_admission,
    validate_probe_task_data,
)


def _features(_instruction, _product):
    return {
        "reward_feature_version": "shopping-reward-features-v2",
        "category": "category",
        "required_options_by_key": {},
        "unresolved_option_requirements": [],
        "query_constraint_contract": {"constraints": []},
    }


def _price(_product, _selection):
    return {"status": "pass", "price": 12.0, "method": "test"}


def _trajectory(task_id, attempt_index, reward, *, valid=True, done=True):
    return {
        "task_id": task_id,
        "attempt_index": attempt_index,
        "done": done,
        "terminal_result": {
            "reward_detail": {
                "reward_valid": valid,
                "sampling_invalid": not valid,
                "terminal_utility": reward,
                "purchase_success": reward > 0.5,
                "reward_type": "gold_purchase" if reward > 0.5 else "early_abstain",
            }
        },
    }


class GrpoProbeGateTests(unittest.TestCase):
    def test_data_gate_accepts_complete_scorable_task(self):
        product = {
            "tag": "train",
            "asin": "a1",
            "title": "title",
            "shop_name": "shop",
            "category": "category",
            "pricing": [12.0],
            "customization_options": {},
            "instructions": [{"asin": "a1", "instruction": "buy it"}],
        }
        result = validate_probe_task_data(
            1,
            product,
            compile_reward_features=_features,
            resolve_variant_price=_price,
        )
        self.assertTrue(result["accepted"])

    def test_data_gate_can_explicitly_validate_eval_source(self):
        product = {
            "tag": "eval",
            "asin": "a1",
            "title": "title",
            "shop_name": "shop",
            "category": "category",
            "pricing": [12.0],
            "customization_options": {},
            "instructions": [{"asin": "a1", "instruction": "buy it"}],
        }
        default_result = validate_probe_task_data(
            1,
            product,
            compile_reward_features=_features,
            resolve_variant_price=_price,
        )
        eval_result = validate_probe_task_data(
            1,
            product,
            compile_reward_features=_features,
            resolve_variant_price=_price,
            allowed_source_tags=("eval",),
        )
        self.assertFalse(default_result["accepted"])
        self.assertIn("source_tag_not_allowed:eval", default_result["reasons"])
        self.assertTrue(eval_result["accepted"])

    def test_data_gate_rejects_inconsistent_price_range_and_normalized_value_collision(self):
        product = {
            "tag": "train",
            "asin": "a1",
            "title": "title",
            "shop_name": "shop",
            "category": "category",
            "pricing": [10.0, 20.0],
            "customization_options": {
                "size": [
                    {"value": "A B", "price": 10.0},
                    {"value": "AB", "price": 30.0},
                ]
            },
            "instructions": [{"asin": "a1", "instruction": "buy it"}],
        }
        result = validate_probe_task_data(
            1,
            product,
            compile_reward_features=_features,
            resolve_variant_price=_price,
        )
        self.assertFalse(result["accepted"])
        self.assertIn("normalized_option_value_collision", result["reasons"])
        self.assertIn("pricing_option_range_mismatch", result["reasons"])

    def test_data_gate_rejects_unresolved_reward_features(self):
        def unresolved(instruction, product):
            result = _features(instruction, product)
            result["unresolved_option_requirements"] = ["red"]
            return result

        product = {
            "tag": "train",
            "asin": "a1",
            "title": "title",
            "shop_name": "shop",
            "category": "category",
            "instructions": [{"asin": "a1", "instruction": "buy red"}],
        }
        result = validate_probe_task_data(
            1,
            product,
            compile_reward_features=unresolved,
            resolve_variant_price=_price,
        )
        self.assertFalse(result["accepted"])
        self.assertIn("required_options_unresolved", result["reasons"])

    def test_data_gate_passes_source_axis_and_plain_value_to_price_resolver(self):
        seen = []

        def structured_features(instruction, product):
            result = _features(instruction, product)
            result["required_options_by_key"] = {
                "color": {
                    "value": "red",
                    "source_axis": "颜色分类",
                    "source": "instruction.instruction_options",
                }
            }
            return result

        def capture_price(product, selection):
            seen.append(selection)
            if selection == {"颜色分类": "red"}:
                return {"status": "pass", "price": 12.0, "method": "selected"}
            return {"status": "unverifiable", "price": None, "method": "missing"}

        product = {
            "tag": "train",
            "asin": "a1",
            "title": "title",
            "shop_name": "shop",
            "category": "category",
            "customization_options": {"颜色分类": [{"value": "red", "price": 12.0}]},
            "instructions": [{"asin": "a1", "instruction": "buy red"}],
        }
        result = validate_probe_task_data(
            1,
            product,
            compile_reward_features=structured_features,
            resolve_variant_price=capture_price,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(seen[0], {"颜色分类": "red"})

    def test_post_gate_accepts_valid_varying_group(self):
        rows = [_trajectory(7, index, reward) for index, reward in enumerate((0.0, 0.2, 0.1, 0.8))]
        result = decide_grpo_admission(rows)
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["probe_role"], "frontier")

    def test_post_gate_requests_retry_for_near_constant_rewards(self):
        rows = [_trajectory(7, index, reward) for index, reward in enumerate((0.10, 0.11, 0.09, 0.10))]
        result = decide_grpo_admission(rows)
        self.assertEqual(result["decision"], "reprobe")
        self.assertTrue(result["eligible_for_more_sampling"])
        self.assertEqual(result["rounds"][0]["signal_class"], "invalid_or_low_variation")

    def test_post_gate_allows_third_attempt_after_two_failed_attempts(self):
        rows = [
            _trajectory(7, index, 0.0, valid=False)
            for index in range(8)
        ]
        result = decide_grpo_admission(rows)
        self.assertEqual(result["decision"], "reprobe")
        self.assertEqual(result["attempts_observed"], 2)

    def test_post_gate_rejects_and_quarantines_after_three_invalid_attempts(self):
        rows = [
            _trajectory(7, index, 0.0, valid=False)
            for index in range(12)
        ]
        result = decide_grpo_admission(rows)
        self.assertEqual(result["decision"], "reject")
        self.assertEqual(result["reason"], "three_attempts_without_valid_reward_variation")
        self.assertFalse(result["eligible_for_more_sampling"])
        self.assertTrue(result["quarantine"])
        self.assertEqual(result["probe_role"], "quarantine")

    def test_post_gate_accepts_valid_third_attempt(self):
        rows = [_trajectory(7, index, 0.1) for index in range(8)]
        rows.extend(
            _trajectory(7, index + 8, reward)
            for index, reward in enumerate((0.0, 0.2, 0.1, 0.8))
        )
        result = decide_grpo_admission(rows)
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["accepted_round"], 2)
        self.assertEqual(result["rounds"][2]["signal_class"], "mixed_outcome_frontier")
        self.assertEqual(result["probe_role"], "frontier")

    def test_post_gate_rejects_duplicate_attempt_indices(self):
        rows = [_trajectory(7, index, reward) for index, reward in enumerate((0.0, 0.2, 0.1, 0.8))]
        rows[-1]["attempt_index"] = 2
        result = decide_grpo_admission(rows)
        self.assertEqual(result["decision"], "reprobe")
        self.assertIn("attempt_indices_incomplete_or_duplicate", result["rounds"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
