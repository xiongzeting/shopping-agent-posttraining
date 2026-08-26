import unittest
from collections import Counter

from scripts.build_difficulty_stratified_teacher_tasks import (
    REPAIR_STRATEGY_CYCLE,
    _band,
    _interleave_proportional,
    _proportional_quotas,
)


class DifficultyStratifiedTeacherTasksTests(unittest.TestCase):
    def test_band_boundaries(self):
        self.assertEqual(_band(9.999), "lt10")
        self.assertEqual(_band(10), "10to15")
        self.assertEqual(_band(15), "15to18")
        self.assertEqual(_band(18), "ge18")

    def test_source_proportions_allocate_500_rows(self):
        counts = Counter(lt10=5321, **{"10to15": 13077, "15to18": 2811, "ge18": 753})
        self.assertEqual(
            _proportional_quotas(counts, 500),
            {"lt10": 121, "10to15": 298, "15to18": 64, "ge18": 17},
        )

    def test_repair_profile_targets_the_observed_failure_mix(self):
        self.assertEqual(
            Counter(REPAIR_STRATEGY_CYCLE),
            {
                "loop_recovery": 8,
                "near_miss_rejection": 7,
                "terminal_tool_commit": 3,
                "option_grounding": 2,
            },
        )

    def test_interleave_keeps_all_bands_distributed(self):
        rows = _interleave_proportional(
            {
                "lt10": [{"task_id": 1}, {"task_id": 2}],
                "10to15": [{"task_id": 3}, {"task_id": 4}, {"task_id": 5}, {"task_id": 6}],
                "15to18": [{"task_id": 7}],
                "ge18": [{"task_id": 8}],
            }
        )

        self.assertEqual({row["task_id"] for row in rows}, set(range(1, 9)))
        self.assertEqual({row["task_id"] for row in rows[:4]}, {1, 3, 7, 8})


if __name__ == "__main__":
    unittest.main()
