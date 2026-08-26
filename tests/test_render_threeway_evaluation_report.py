import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_threeway_evaluation_report.py"
SPEC = spec_from_file_location("render_threeway_evaluation_report", SCRIPT)
report = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(report)


def evaluation(
    task_id,
    *,
    strict=False,
    reward=0.0,
    judge_status="valid",
    total_tokens=1000,
    duration=10.0,
    context_hard_limit=False,
):
    scores = {
        name: {"score": 2 if strict else 1} for name in report.DIMENSIONS
    }
    return {
        "task_id": task_id,
        "reward_and_terminal": {
            "metrics": {
                "strict_gold_success": strict,
                "purchase_success": strict,
                "reward_valid": True,
                "final_reward": reward,
                "reward_type": "gold_purchase" if strict else "wrong_purchase",
                "done": True,
            }
        },
        "trajectory_quality": {
            "judge_status": judge_status,
            "not_judged_reason": None,
            "dimension_scores": scores if judge_status == "valid" else {},
            "errors": {"primary": None, "secondary": []},
        },
        "requirement_rubric": {
            "rubrics": [{"rubric_id": "r1", "hardness": "hard"}],
            "assessments": [{"rubric_id": "r1", "status": "satisfied"}],
        },
        "deterministic": {
            "actions_and_efficiency": {
                "executed_tool_steps": 3,
                "action_attempts": 4,
            },
            "legality": {"guard_rejection_count": 1},
            "repetition": {
                "duplicate_canonical_action_count": 1,
                "duplicate_search_query_count": 0,
            },
            "context": {
                "any_observation_truncated": False,
                "max_context_usage_ratio": 0.5,
                "total_tokens": total_tokens,
            },
            "timing": {"trajectory_duration_seconds": duration},
            "validity": {
                "infrastructure_invalid": False,
                "context_hard_limit": context_hard_limit,
            },
        },
    }


class ThreewayReportTest(unittest.TestCase):
    def test_bilingual_display_labels_keep_original_machine_terms(self):
        self.assertEqual(report.DISPLAY["base"], "Base")
        self.assertEqual(
            report.rubric_status_display("not_applicable"),
            "not_applicable（不适用）",
        )
        self.assertEqual(report.stratum_display("suite"), "suite")
        self.assertEqual(
            report.error_display("illegal_action"),
            "illegal_action（非法动作）",
        )
        self.assertEqual(
            report.reward_type_display("gold_purchase"),
            "gold_purchase（目标商品购买）",
        )
        self.assertEqual(report.RUBRIC_HARDNESS_DISPLAY["hard"], "Hard（硬约束）")
        self.assertIn(
            "违反 <strong>3</strong>",
            report.rubric_status_stack_html({"violated": 3}),
        )
        self.assertEqual(
            report.stratum_display("candidate_comparison"),
            "candidate_comparison（候选比较）",
        )
        self.assertEqual(report.error_display("future_error"), "future_error")

    def test_enriches_token_latency_quantiles_and_hard_limit_count(self):
        summaries = {label: {"deterministic": {}} for label in report.LABELS}
        evaluations = {
            label: [
                evaluation(1, total_tokens=100, duration=1.0),
                evaluation(
                    2,
                    total_tokens=300,
                    duration=5.0,
                    context_hard_limit=(label == "grpo"),
                ),
            ]
            for label in report.LABELS
        }

        report.enrich_deterministic_summaries(summaries, evaluations)

        self.assertEqual(
            summaries["base"]["deterministic"]["provider_total_tokens_p50"], 200
        )
        self.assertEqual(
            summaries["sft"]["deterministic"]["trajectory_duration_seconds_p95"], 4.8
        )
        self.assertEqual(
            summaries["base"]["deterministic"]["context_hard_limit_tasks"], 0
        )
        self.assertEqual(
            summaries["base"]["deterministic"]["context_usage_ratio_p50"], 0.5
        )
        self.assertEqual(
            summaries["grpo"]["deterministic"]["context_hard_limit_tasks"], 1
        )

    def test_rubric_categories_split_price_color_and_size(self):
        self.assertEqual(
            report.rubric_category(
                {"constraint_type": "price_preference", "field_path": "purchase.price"}
            ),
            "price",
        )
        self.assertEqual(
            report.rubric_category(
                {"constraint_type": "option", "field_path": "purchase.options.color"}
            ),
            "color_option",
        )
        self.assertEqual(
            report.rubric_category(
                {"constraint_type": "option", "field_path": "purchase.options.size"}
            ),
            "size_option",
        )

    def test_build_per_task_counts_missing_base_as_failure(self):
        indexes = {
            "base": {1: evaluation(1)},
            "sft": {1: evaluation(1, strict=True), 2: evaluation(2)},
            "grpo": {1: evaluation(1, strict=True), 2: evaluation(2, strict=True)},
        }
        summaries = {
            "base": {"expected_tasks": 2, "missing_task_ids": [2]},
            "sft": {"expected_tasks": 2, "missing_task_ids": []},
            "grpo": {"expected_tasks": 2, "missing_task_ids": []},
        }

        rows = report.build_per_task(indexes, summaries)

        self.assertEqual([row["task_id"] for row in rows], [1, 2])
        missing = rows[1]
        self.assertFalse(missing["base_completed"])
        self.assertEqual(missing["base_not_judged_reason"], "missing_trajectory")
        self.assertEqual(
            missing["base_to_sft_strict_transition"], "failure_to_failure"
        )
        self.assertEqual(
            missing["sft_to_grpo_strict_transition"], "failure_to_success"
        )

    def test_transition_summary_uses_fixed_four_cells(self):
        pair = {
            "reward_and_terminal": {
                "strict_success_transitions": {
                    "failure_to_success": 3,
                    "success_to_success": 4,
                    "success_to_failure": 2,
                    "failure_to_failure": 1,
                }
            }
        }

        text = report.strict_transition_line(pair)

        self.assertEqual(
            text, "失败→成功 3；成功→成功 4；成功→失败 2；失败→失败 1"
        )


if __name__ == "__main__":
    unittest.main()
