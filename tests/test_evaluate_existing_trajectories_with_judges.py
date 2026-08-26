import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_existing_trajectories_with_judges.py"
SPEC = spec_from_file_location("existing_trajectory_judge", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExistingTrajectoryJudgeTest(unittest.TestCase):
    def test_task_ids_resolve_through_product_asin(self):
        products = [
            {"asin": "a", "category": "cat", "title": "A"},
            {"asin": "b", "category": "cat", "title": "B"},
        ]
        goals = [
            {
                "asin": "b",
                "instruction_text": "buy B",
                "attributes": ["feature"],
                "goal_options": [],
            },
            {
                "asin": "a",
                "instruction_text": "buy A",
                "attributes": ["feature"],
                "goal_options": [],
            },
        ]

        facts = MODULE.build_task_facts_for_task_ids(
            task_ids=[0, 1],
            all_products=products,
            goals=goals,
        )

        self.assertEqual([row["task_id"] for row in facts], [0, 1])
        self.assertEqual([row["query"] for row in facts], ["buy A", "buy B"])

    def test_first_goal_for_an_asin_matches_environment_contract(self):
        products = [{"asin": "a", "category": "cat", "title": "A"}]
        goals = [
            {
                "asin": "a",
                "instruction_text": "first query",
                "attributes": ["feature"],
                "goal_options": [],
            },
            {
                "asin": "a",
                "instruction_text": "later query",
                "attributes": ["feature"],
                "goal_options": [],
            },
        ]

        facts = MODULE.build_task_facts_for_task_ids(
            task_ids=[0],
            all_products=products,
            goals=goals,
        )

        self.assertEqual(facts[0]["query"], "first query")


if __name__ == "__main__":
    unittest.main()
