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
    def test_aggregate_and_task_stage_sets_cover_eight_runs(self):
        module = load_module()
        self.assertEqual(
            module.LABELS,
            (
                "base",
                "sft_normal1000",
                "sft",
                "grpo100",
                "grpo230",
                "qwen38_27b",
                "grpo230_v3",
                "qwen38_27b_v3",
            ),
        )
        self.assertEqual(module.TASK_LABELS, module.LABELS)
        self.assertEqual(module.DISPLAY["sft_normal1000"], "普通 SFT v1")
        self.assertEqual(module.DISPLAY["sft"], "纠错 SFT v1")
        self.assertEqual(
            module.TRANSITIONS,
            (
                ("sft", "grpo100"),
                ("grpo100", "grpo230"),
                ("grpo230", "grpo230_v3"),
            ),
        )

    def test_per_task_contains_only_requested_transitions(self):
        module = load_module()
        module.REWARD_OVERLAY.clear()
        failure = {
            "strict_gold": False,
            "purchase_success": False,
            "reward_valid": True,
            "final_reward": -0.8,
            "reward_type": "guard_rejection",
        }
        success = {
            "strict_gold": True,
            "purchase_success": True,
            "reward_valid": True,
            "final_reward": 1.0,
            "reward_type": "gold_purchase",
        }
        module.REWARD_OVERLAY.update(
            {
                label: {1: dict(success if label != "base" else failure)}
                for label in module.TASK_LABELS
            }
        )
        rows = module.build_per_task(
            {label: {} for label in module.TASK_LABELS},
            [1],
        )
        row = rows[0]
        self.assertEqual(row["sft_to_grpo100_strict_transition"], "success_to_success")
        self.assertEqual(row["grpo100_to_grpo230_purchase_transition"], "success_to_success")
        self.assertEqual(row["grpo230_to_grpo230_v3_purchase_transition"], "success_to_success")
        self.assertFalse(any(key.startswith("base_to_sft_") for key in row))
        self.assertFalse(any(key.startswith("qwen38_27b_to_") for key in row))

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
        self.assertEqual(usage["models"]["sft_normal1000"]["schema_tool_count"], 12)
        self.assertEqual(usage["models"]["base"]["counts"]["view_description"], 3)
        self.assertEqual(usage["models"]["base"]["counts"]["view_attributes"], 0)
        self.assertEqual(usage["models"]["base"]["nonstandard_tool_counts"], {"think": 4})
        self.assertEqual(usage["models"]["grpo230"]["schema_tool_count"], 8)
        self.assertIsNone(usage["models"]["grpo230"]["counts"]["view_description"])
        self.assertIsNone(usage["models"]["qwen38_27b"]["counts"]["view_attributes"])

    def test_full_trajectory_renderer_keeps_every_model_round_and_response(self):
        module = load_module()
        trajectory = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "request"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_products",
                                "arguments": '{"query":"测试商品"}',
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "search_products",
                    "content": "完整 Observation <商品A>",
                },
                {
                    "role": "assistant",
                    "content": "需要改正动作",
                    "tool_calls": [],
                },
                {"role": "user", "content": "Harness 完整纠正提示"},
            ]
        }

        rendered, round_count = module.render_trajectory_rounds(trajectory)

        self.assertEqual(round_count, 2)
        self.assertEqual(rendered.count('class="trajectory-round"'), 2)
        self.assertIn("完整 Observation &lt;商品A&gt;", rendered)
        self.assertIn("Harness 完整纠正提示", rendered)
        self.assertIn("search_products", rendered)
        self.assertIn("测试商品", rendered)

    def test_trajectory_tool_schema_contains_the_complete_eight_tool_contract(self):
        module = load_module()
        rendered = module.render_tool_schema()
        for name in (
            "search_products",
            "open_product",
            "select_option",
            "next_page",
            "prev_page",
            "back_to_search",
            "buy_now",
            "finish_without_purchase",
        ):
            self.assertIn(f"<code>{name}</code>", rendered)

    def test_candidate_recovery_summary_reports_real_r4_conversion(self):
        module = load_module()
        if not module.GRPO230_V3_TRAJECTORIES.exists():
            self.skipTest("raw trajectories are intentionally excluded from the public repository")
        summary = module.build_candidate_recovery_summary()

        self.assertEqual(summary["triggered"], 34)
        self.assertEqual(summary["entered"], 34)
        self.assertEqual(summary["gold"], 12)
        self.assertEqual(summary["valid"], 4)
        self.assertEqual(summary["successful"], 16)
        self.assertAlmostEqual(summary["success_rate"], 16 / 34)
        self.assertEqual(summary["wrong"], 15)
        self.assertEqual(summary["partial"], 1)
        self.assertEqual(summary["guard_rejection"], 2)
        self.assertEqual(summary["remaining_loop_tasks"], [788])

    def test_success_stratification_uses_frozen_recall_and_difficulty_buckets(self):
        module = load_module()
        success = {"purchase_success": True}
        failure = {"purchase_success": False}
        module.REWARD_OVERLAY.clear()
        module.REWARD_OVERLAY.update(
            {
                label: {
                    1: dict(success),
                    2: dict(failure),
                    3: dict(success if label == "grpo230_v3" else failure),
                }
                for label in module.LABELS
            }
        )
        slices = {
            1: {"retrieval_bucket": "rank1", "difficulty_bucket": "under_10"},
            2: {"retrieval_bucket": "rank1", "difficulty_bucket": "10_15"},
            3: {"retrieval_bucket": "missing", "difficulty_bucket": "18_plus"},
        }

        result = module.build_success_stratification(slices)

        recall = result["retrieval_bucket"]["buckets"]
        self.assertEqual(list(recall), ["rank1", "missing"])
        self.assertEqual(recall["rank1"]["tasks"], 2)
        self.assertEqual(recall["rank1"]["models"]["base"]["successes"], 1)
        self.assertAlmostEqual(
            recall["rank1"]["models"]["base"]["success_rate"], 0.5
        )
        self.assertEqual(
            recall["missing"]["models"]["grpo230_v3"]["successes"], 1
        )
        difficulty = result["difficulty_bucket"]["buckets"]
        self.assertEqual(list(difficulty), ["under_10", "10_15", "18_plus"])

    def test_teacher_corrective_showcase_covers_four_strict_success_strategies(self):
        module = load_module()
        if not module.SFT_FINAL1000.exists():
            self.skipTest("private SFT data is intentionally excluded from the public repository")
        cases = module.build_teacher_corrective_cases()

        self.assertEqual(
            [(case["strategy"], case["task_id"]) for case in cases],
            [
                ("loop_recovery", 11555),
                ("near_miss_rejection", 9512),
                ("option_grounding", 22436),
                ("terminal_tool_commit", 15956),
            ],
        )
        for case in cases:
            self.assertEqual(case["actions"][-1], "buy_now")
            self.assertIn("selected_options", case["final_detail"])
        self.assertEqual(cases[0]["steps"], 12)
        self.assertIn("全铜6分排水阀+2米管+扳手+卡扣", cases[0]["final_detail"]["selected_options"])
        self.assertIn("空白贴", cases[1]["correction"])
        self.assertEqual(cases[2]["action_chain"].count("select_option"), 3)
        self.assertIn("买一送一", cases[3]["final_detail"]["selected_options"])


if __name__ == "__main__":
    unittest.main()
