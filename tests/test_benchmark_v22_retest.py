import json
import unittest
from pathlib import Path

from scripts.merge_benchmark_v22_trajectories import _compatible_run_contract


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class BenchmarkV22RetestTest(unittest.TestCase):
    def test_retest_subset_matches_replacement_manifest(self):
        manifest = json.loads(
            (ROOT / "data/evaluation/replacement-manifest-v2.2.json").read_text(
                encoding="utf-8"
            )
        )
        retest_ids = {
            int(row["task_id"])
            for row in _jsonl(ROOT / "data/evaluation/retest-v2.2/tasks.jsonl")
        }
        self.assertEqual(retest_ids, set(manifest["added_task_ids"]))
        self.assertEqual(len(retest_ids), 23)
        self.assertEqual(manifest["retained_count"], 217)

    def test_merge_contract_ignores_worker_count_but_not_model_or_protocol(self):
        manifest = {
            "actor": {"model": "checkpoint", "tokenizer": "tokenizer"},
            "protocol": {"workers": 8, "temperature": 0.0, "context_window": 35000},
            "environment": {"environment": "v2.4", "reward": "v3.2"},
            "code": {"system_prompt_sha256": "prompt", "tool_schema_sha256": "tools"},
        }
        other = json.loads(json.dumps(manifest))
        other["protocol"]["workers"] = 1
        self.assertEqual(
            _compatible_run_contract(manifest),
            _compatible_run_contract(other),
        )
        other["actor"]["model"] = "different-checkpoint"
        self.assertNotEqual(
            _compatible_run_contract(manifest),
            _compatible_run_contract(other),
        )


if __name__ == "__main__":
    unittest.main()
