import unittest

from scripts.finalize_grpo_training_probe import finalize


def _trajectory(task_id, attempt_index, reward, valid=True):
    return {
        "task_id": task_id,
        "attempt_index": attempt_index,
        "done": True,
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


class FinalizeGrpoTrainingProbeTests(unittest.TestCase):
    def test_partitions_all_candidates(self):
        candidates = [{"task_id": 1}, {"task_id": 2}, {"task_id": 3}]
        trajectories = []
        trajectories.extend(_trajectory(1, i, r) for i, r in enumerate((0.0, 0.2, 0.1, 0.8)))
        trajectories.extend(_trajectory(2, i, 0.1) for i in range(4))
        trajectories.extend(_trajectory(3, i, 0.0, valid=False) for i in range(12))
        outputs = finalize(
            candidates,
            trajectories,
            rollout_n=4,
            max_rounds=3,
            reward_tolerance=0.025,
        )
        self.assertEqual([row["task_id"] for row in outputs["accepted"]], [1])
        self.assertEqual([row["task_id"] for row in outputs["reprobe"]], [2])
        self.assertEqual([row["task_id"] for row in outputs["rejected"]], [3])


if __name__ == "__main__":
    unittest.main()
