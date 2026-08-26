import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.select_diverse_teacher_data import (
    _materialize_references,
    _quality_reasons,
    _references_from_paths,
    _write_references_atomic,
    select_references,
    select_rows,
)


def trajectory(task_id, sequence):
    return {
        "task_id": task_id,
        "trajectory_id": f"trajectory-{task_id}",
        "steps": [{"tool_name": name} for name in sequence],
    }


class SelectDiverseTeacherDataTests(unittest.TestCase):
    def test_quality_filter_requires_terminal_buy_and_rejects_repeated_critical_action(self):
        assistant_final = trajectory(
            1, ("search_products", "open_product", "view_features")
        )
        repeated_search = trajectory(
            2,
            (
                "search_products",
                "open_product",
                "search_products",
                "buy_now",
            ),
        )
        repeated_search["steps"][0]["parameters"] = {"query": "same"}
        repeated_search["steps"][2]["parameters"] = {"query": "same"}

        self.assertIn(
            "quality_missing_explicit_terminal_buy", _quality_reasons(assistant_final)
        )
        self.assertIn(
            "quality_repeated_critical_action", _quality_reasons(repeated_search)
        )

        recovered_search = trajectory(
            3,
            (
                "search_products",
                "open_product",
                "view_features",
                "prev_page",
                "back_to_search",
                "search_products",
                "open_product",
                "buy_now",
            ),
        )
        recovered_search["steps"][0]["parameters"] = {"query": "same"}
        recovered_search["steps"][5]["parameters"] = {"query": "same"}
        self.assertNotIn(
            "quality_repeated_critical_action", _quality_reasons(recovered_search)
        )

    def test_sequence_and_eight_step_caps_are_enforced(self):
        dominant = (
            "search_products",
            "open_product",
            "select_option",
            "view_features",
            "prev_page",
            "view_description",
            "prev_page",
            "buy_now",
        )
        varied = [
            ("search_products", "open_product", "buy_now"),
            ("search_products", "open_product", "view_reviews", "prev_page", "buy_now"),
            ("search_products", "back_to_search", "search_products", "open_product", "buy_now"),
            ("search_products", "open_product", "back_to_search", "open_product", "buy_now"),
        ]
        rows = [trajectory(index, dominant) for index in range(10)]
        rows.extend(
            trajectory(100 + index, sequence)
            for index, sequence in enumerate(varied)
        )

        with patch(
            "scripts.select_diverse_teacher_data.acceptance_reasons",
            return_value=(True, []),
        ):
            selected, report = select_rows(
                {"lt10": rows},
                quotas={"lt10": 10, "10to15": 0, "15to18": 0, "ge18": 0},
                target_total=10,
                sequence_cap_ratio=0.2,
                eight_step_cap_ratio=0.3,
                seed="test",
            )

        self.assertEqual(len(selected), 6)
        self.assertEqual(report["result"]["eight_step_rows"], 2)
        self.assertLessEqual(
            max(item["count"] for item in report["result"]["top_sequences"]), 2
        )
        self.assertEqual(report["deficits"]["lt10"], 4)

    def test_path_selection_keeps_only_lightweight_references_until_materialization(self):
        rows = [
            trajectory(1, ("search_products", "open_product", "buy_now")),
            trajectory(
                2,
                (
                    "search_products",
                    "open_product",
                    "view_reviews",
                    "prev_page",
                    "buy_now",
                ),
            ),
        ]
        rows[0]["large_observation"] = "x" * 10000
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with patch(
                "scripts.select_diverse_teacher_data.acceptance_reasons",
                return_value=(True, []),
            ):
                references = _references_from_paths({"lt10": [path]}, seed="test")
                selected, report = select_references(
                    references,
                    quotas={"lt10": 2, "10to15": 0, "15to18": 0, "ge18": 0},
                    target_total=2,
                    sequence_cap_ratio=1.0,
                    eight_step_cap_ratio=1.0,
                )
                materialized = _materialize_references(selected)
                output = Path(tmpdir) / "selected.jsonl"
                _write_references_atomic(output, selected)
                written = [
                    json.loads(line)
                    for line in output.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]

        self.assertEqual(report["result"]["rows"], 2)
        self.assertTrue(all("path" in reference for reference in selected))
        self.assertTrue(all("large_observation" not in reference for reference in selected))
        self.assertEqual({row["task_id"] for row in materialized}, {1, 2})
        materialized_by_task = {row["task_id"]: row for row in materialized}
        self.assertEqual(len(materialized_by_task[1]["large_observation"]), 10000)
        self.assertEqual({row["task_id"] for row in written}, {1, 2})


if __name__ == "__main__":
    unittest.main()
