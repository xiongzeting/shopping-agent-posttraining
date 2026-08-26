"""Final-240 and Reward v4 adaptations for the four-panel evaluator."""

from copy import deepcopy
import unittest

from shopping_grpo.evaluation.comparison import compare_evaluation_runs
from shopping_grpo.evaluation.contracts import (
    JUDGE_DIMENSIONS,
    JUDGE_SCHEMA_VERSION,
    RUBRIC_SCHEMA_VERSION,
)
from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.results import (
    assemble_task_evaluation,
    summarize_evaluations,
)
from shopping_grpo.evaluation.trajectory import NORMALIZED_TRAJECTORY_VERSION


def _normalized(status="fail"):
    constraint_results = [
        {
            "constraint_id": "budget-1",
            "constraint_type": "budget_upper",
            "role": "hard_gate",
            "expected": 100.0,
            "source": "query.explicit_budget",
            "query_evidence": "100元以内",
            "status": status,
            "comparator": "numeric_upper_bound_v1",
            "actual": 120.0,
            "source_field": "variant_price",
            "evidence": {},
        }
    ]
    return {
        "schema_version": NORMALIZED_TRAJECTORY_VERSION,
        "trajectory_id": "trajectory-1",
        "task_id": 1,
        "status": "done",
        "done": True,
        "final_reward": -0.85,
        "events": [],
        "terminal": {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v4",
                "query_constraint_version": "shopping-query-constraints-v1",
                "constraint_results": constraint_results,
                "constraint_summary": {
                    "total": 1,
                    "status_counts": {
                        "pass": int(status == "pass"),
                        "fail": int(status == "fail"),
                        "unverifiable": int(status == "unverifiable"),
                    },
                },
                "reward_type": "wrong_purchase",
                "reward_valid": True,
                "purchase_success": False,
                "termination_reason": "wrong_purchase",
                "terminal_utility": -0.85,
                "weighted_score": 0.0,
            },
        },
        "errors": {},
        "context": {},
    }


def _rubric_bundle():
    return {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "task_id": 1,
        "query": "100元以内的杯子",
        "rubric_version": "test-rubric-v1",
        "generation": {
            "extractor_version": "test",
            "curator_model": "test",
            "curator_prompt_version": "test",
            "task_data_hash": "hash",
            "query_hash": "hash",
        },
        "rubrics": [],
    }


def _judge_result():
    return {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "task_id": 1,
        "trajectory_id": "trajectory-1",
        "judge_status": "valid",
        "rubric_assessments": [],
        "dimension_scores": {
            name: {"score": 2, "reason": "ok", "evidence_event_ids": []}
            for name in JUDGE_DIMENSIONS
        },
        "errors": {
            "primary": None,
            "secondary": [],
            "evidence_event_ids": [],
        },
        "overall_diagnosis": "ok",
    }


def _evaluation(status="fail"):
    normalized = _normalized(status)
    metrics = compute_deterministic_metrics(normalized)
    return assemble_task_evaluation(
        actor={"label": "test"},
        normalized_trajectory=normalized,
        deterministic_metrics=metrics,
        rubric_bundle=_rubric_bundle(),
        judge_result=_judge_result(),
    )


class EvaluationAdaptationTest(unittest.TestCase):
    def test_reward_constraints_are_routed_to_requirement_panel(self):
        record = _evaluation("fail")

        self.assertEqual(
            record["requirement_rubric"]["reward_constraint_version"],
            "shopping-query-constraints-v1",
        )
        self.assertEqual(
            record["requirement_rubric"]["reward_constraint_results"][0][
                "status"
            ],
            "fail",
        )
        self.assertNotIn("requirement_constraints", record["deterministic"])

    def test_four_panel_summary_counts_reward_constraint_statuses(self):
        summary = summarize_evaluations(
            expected_task_ids=[1],
            evaluations=[_evaluation("unverifiable")],
            task_slices={
                1: {
                    "suite": "challenge",
                    "domain": "家居家装",
                    "challenge_slice": "price_semantics",
                }
            },
        )

        requirement = summary["requirement_rubric"]
        self.assertEqual(
            requirement["reward_constraint_status_counts"],
            {"unverifiable": 1},
        )
        self.assertEqual(
            summary["stratified"]["challenge_slice"]["price_semantics"][
                "expected_tasks"
            ],
            1,
        )

    def test_paired_comparison_uses_reward_constraint_fail_delta(self):
        failed = _evaluation("fail")
        passed = deepcopy(_evaluation("pass"))
        comparison = compare_evaluation_runs(
            expected_task_ids=[1],
            runs={"baseline": [failed], "sft": [passed]},
            task_slices={
                1: {
                    "suite": "core",
                    "domain": "家居家装",
                    "challenge_slice": None,
                }
            },
        )

        delta = comparison["pairwise"]["baseline_to_sft"][
            "requirement_rubric"
        ]["reward_constraint_fail_delta"]
        self.assertEqual(delta["improved_tasks"], 1)
        self.assertEqual(
            comparison["stratified"]["suite"]["core"]["expected_tasks"],
            1,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
