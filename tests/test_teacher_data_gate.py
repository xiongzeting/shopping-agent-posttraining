import json
import unittest

from shopping_grpo.collection.data_gate import (
    audit_data_gate,
    retrieval_bucket,
    trajectory_coverage,
)

POLICY = {
    "target_rows": 10,
    "retrieval": {
        "rank1_max_share": 0.4,
        "rank2_5_min_share": 0.2,
        "rank6_20_min_share": 0.1,
        "rank21_150_min_share": 0.1,
        "missing_min_share": 0.2,
    },
    "coverage_min_share": {
        "search_reformulation": 0.2,
        "candidate_comparison": 0.2,
        "multiple_options": 0.2,
        "guard_recovery": 0.1,
        "medium_or_long": 0.2,
        "long": 0.1,
    },
    "caps": {
        "exact_action_sequence_max_share": 0.3,
        "eight_step_max_share": 0.3,
    },
}


def _step(tool_name, **parameters):
    return {"tool_name": tool_name, "parameters": parameters}


def _trajectory(task_id):
    steps = [
        _step("search_products", query=f"query-{task_id}"),
        _step("open_product", asin=f"asin-{task_id}"),
        _step("buy_now"),
    ]
    if task_id in {0, 1}:
        steps.insert(1, _step("search_products", query=f"reformulated-{task_id}"))
        steps.insert(3, _step("open_product", asin=f"alternative-{task_id}"))
        steps.insert(4, _step("select_option", value="large"))
        steps.insert(5, _step("select_option", value="blue"))
    if task_id == 0:
        while len(steps) <= 20:
            steps.insert(-1, _step("view_description"))
    elif task_id == 1:
        while len(steps) <= 10:
            steps.insert(-1, _step("view_features"))
    else:
        patterns = {
            0: ["view_description"],
            1: ["view_features", "prev_page"],
            2: ["view_reviews", "back_to_search", "search_products"],
            3: ["view_attributes", "next_page", "prev_page", "open_product"],
        }
        for name in patterns[task_id % 4]:
            parameters = {"query": f"extra-{task_id}"} if name == "search_products" else {}
            steps.insert(-1, _step(name, **parameters))
    trajectory = {"task_id": task_id, "steps": steps}
    if task_id == 0:
        trajectory["blocked_tool_calls"] = [{"reason": "not_clickable"}]
    return trajectory


def _passing_rows_and_ranks():
    rows = [_trajectory(task_id) for task_id in range(10)]
    ranks = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 5,
        6: 6,
        7: 21,
        8: None,
        9: None,
    }
    return rows, ranks


class TeacherDataGateTests(unittest.TestCase):
    def test_problem_focused_coverage_detects_terminal_and_repeat_quality(self):
        row = {
            "steps": [
                _step("search_products", query="same"),
                _step("open_product", asin="candidate"),
                _step("view_features"),
                _step("search_products", query="same"),
                _step("buy_now"),
            ]
        }

        coverage = trajectory_coverage(row)

        self.assertTrue(coverage["evidence_verification"])
        self.assertTrue(coverage["explicit_terminal_buy"])
        self.assertFalse(coverage["clean_critical_actions"])

    def test_retrieval_bucket_treats_beyond_top150_as_missing(self):
        self.assertEqual(retrieval_bucket(150), "rank21_150")
        self.assertEqual(retrieval_bucket(151), "missing")
        with self.assertRaises(ValueError):
            retrieval_bucket(0)

    def test_exact_retrieval_and_coverage_contract_passes(self):
        rows, ranks = _passing_rows_and_ranks()

        report = audit_data_gate(rows, retrieval_ranks=ranks, policy=POLICY)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["deficits"], {})
        self.assertEqual(report["retrieval_rank_counts"]["rank1"], 4)
        self.assertEqual(report["coverage_counts"]["candidate_comparison"], 2)

    def test_rank1_survivor_bias_fails(self):
        rows, ranks = _passing_rows_and_ranks()
        ranks[4] = 1

        report = audit_data_gate(rows, retrieval_ranks=ranks, policy=POLICY)

        self.assertEqual(report["status"], "failed")
        self.assertIn("rank1_max", report["deficits"])

    def test_candidate_comparison_deficit_fails(self):
        rows, ranks = _passing_rows_and_ranks()
        rows[1]["steps"] = [
            step
            for step in rows[1]["steps"]
            if step.get("parameters", {}).get("asin") != "alternative-1"
        ]

        report = audit_data_gate(rows, retrieval_ranks=ranks, policy=POLICY)

        self.assertIn("coverage.candidate_comparison", report["deficits"])

    def test_exact_action_sequence_cap_fails(self):
        rows, ranks = _passing_rows_and_ranks()
        identical = [
            _step("search_products", query="same"),
            _step("open_product", asin="x"),
            _step("buy_now"),
        ]
        for row in rows[:4]:
            row["steps"] = json.loads(json.dumps(identical))

        report = audit_data_gate(rows, retrieval_ranks=ranks, policy=POLICY)

        self.assertIn("exact_action_sequence_max", report["deficits"])

    def test_missing_rank_metadata_fails_without_hiding_other_coverage(self):
        rows, ranks = _passing_rows_and_ranks()
        ranks.pop(0)

        report = audit_data_gate(rows, retrieval_ranks=ranks, policy=POLICY)

        self.assertIn("retrieval_rank_metadata", report["deficits"])
        self.assertEqual(report["coverage_counts"]["guard_recovery"], 1)
        self.assertEqual(report["length_histogram"][len(rows[0]["steps"])], 1)

    def test_wrong_target_row_count_fails(self):
        rows, ranks = _passing_rows_and_ranks()

        report = audit_data_gate(rows[:-1], retrieval_ranks=ranks, policy=POLICY)

        self.assertIn("target_rows", report["deficits"])

    def test_coverage_reads_legacy_tool_call_arguments(self):
        trajectory = {
            "steps": [
                {
                    "tool_name": "search_products",
                    "tool_call": {
                        "function": {
                            "arguments": json.dumps({"query": "first"})
                        }
                    },
                },
                {
                    "tool_name": "search_products",
                    "tool_call": {
                        "function": {
                            "arguments": json.dumps({"query": "second"})
                        }
                    },
                },
            ]
        }

        self.assertTrue(trajectory_coverage(trajectory)["search_reformulation"])


if __name__ == "__main__":
    unittest.main()
