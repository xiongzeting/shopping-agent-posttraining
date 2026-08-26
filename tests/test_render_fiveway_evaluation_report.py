import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_fiveway_evaluation_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("render_fiveway", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FivewayReportTest(unittest.TestCase):
    def test_aggregate_and_task_stage_sets_are_distinct(self):
        module = load_module()
        self.assertEqual(
            module.LABELS,
            ("base", "sft", "grpo50", "grpo100", "grpo230", "qwen38_27b"),
        )
        self.assertEqual(module.TASK_LABELS, ("base", "sft", "grpo230"))

    def test_per_task_contains_only_requested_transitions(self):
        module = load_module()
        module.REWARD_OVERLAY.clear()
        module.REWARD_OVERLAY.update(
            {
                "base": {
                    1: {
                        "strict_gold": False,
                        "purchase_success": False,
                        "reward_valid": True,
                        "final_reward": -0.8,
                        "reward_type": "guard_rejection",
                    }
                },
                "sft": {
                    1: {
                        "strict_gold": True,
                        "purchase_success": True,
                        "reward_valid": True,
                        "final_reward": 1.0,
                        "reward_type": "gold_purchase",
                    }
                },
                "grpo230": {
                    1: {
                        "strict_gold": True,
                        "purchase_success": True,
                        "reward_valid": True,
                        "final_reward": 1.0,
                        "reward_type": "gold_purchase",
                    }
                },
            }
        )
        rows = module.build_per_task(
            {"base": {}, "sft": {}, "grpo50": {}, "grpo100": {}, "grpo230": {}},
            [1],
        )
        row = rows[0]
        self.assertEqual(row["sft_to_grpo230_strict_transition"], "success_to_success")
        self.assertFalse(any(key.startswith("base_to_sft_") for key in row))
        self.assertFalse(any(key.startswith("grpo50_") for key in row))
        self.assertFalse(any(key.startswith("grpo100_") for key in row))

    def test_tool_usage_marks_removed_information_tools_as_unused(self):
        module = load_module()

        def evaluation(**counts):
            return {
                "deterministic": {
                    "actions_and_efficiency": {"tool_counts": counts}
                }
            }

        evaluations = {
            label: [evaluation(search_products=1)] for label in module.LABELS
        }
        evaluations["base"] = [
            evaluation(search_products=2, view_description=3, think=4)
        ]
        usage = module.build_tool_usage(evaluations)

        self.assertEqual(usage["models"]["base"]["schema_tool_count"], 12)
        self.assertEqual(usage["models"]["base"]["counts"]["view_description"], 3)
        self.assertEqual(usage["models"]["base"]["counts"]["view_attributes"], 0)
        self.assertEqual(usage["models"]["base"]["nonstandard_tool_counts"], {"think": 4})
        self.assertEqual(usage["models"]["grpo230"]["schema_tool_count"], 8)
        self.assertIsNone(usage["models"]["grpo230"]["counts"]["view_description"])
        self.assertIsNone(usage["models"]["qwen38_27b"]["counts"]["view_attributes"])


if __name__ == "__main__":
    unittest.main()
