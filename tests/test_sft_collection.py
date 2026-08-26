import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from shopping_grpo.collection.sft import (
    acceptance_reasons,
    build_collection_artifacts,
    build_sft_row,
    result_consistency_warnings,
    sanitizable_duplicate_call_ids,
)
from shopping_grpo.environment.actions import RUNTIME_GUARD_FIELD


def _assistant_tool(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _tool_message(call_id, name, content):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


def _accepted_trajectory(task_id=1):
    asin = "100000000001"
    option_id = "opt_0123456789abcdef"
    messages = [
        {"role": "system", "content": "use tools"},
        {"role": "user", "content": "帮我买乳胶枕"},
        _assistant_tool("search_products", {"query": "乳胶枕"}, "call-1"),
        _tool_message(
            "call-1",
            "search_products",
            f"1|{asin}|99.0|brand|category|attr|乳胶枕\n"
            f'可点击的按钮: ["{asin}"]',
        ),
        _assistant_tool("open_product", {"asin": asin}, "call-2"),
        _tool_message(
            "call-2",
            "open_product",
            f'详情\n可点击的按钮: ["{option_id}", "Buy Now"]',
        ),
        _assistant_tool("select_option", {"value": option_id}, "call-3"),
        _tool_message(
            "call-3",
            "select_option",
            '已选择\n可点击的按钮: ["Buy Now"]',
        ),
        _assistant_tool("buy_now", {}, "call-4"),
        _tool_message("call-4", "buy_now", "Reward: 1.0; Gold purchase"),
    ]
    messages[2]["reasoning_content"] = "Teacher private reasoning"
    return {
        "trajectory_id": f"trajectory-{task_id}",
        "task_id": task_id,
        "attempt_index": 0,
        "status": "done",
        "done": True,
        "error": None,
        "release_error": None,
        "initial_result": {
            "environment_version": "shopsimulator-environment-v2.4"
        },
        "messages": messages,
        "steps": [
            {
                "tool_name": "search_products",
                "tool_call": messages[2]["tool_calls"][0],
                "env_action": "search[乳胶枕]",
                "done": False,
            },
            {
                "tool_name": "open_product",
                "tool_call": messages[4]["tool_calls"][0],
                "env_action": f"click[{asin}]",
                "done": False,
            },
            {
                "tool_name": "select_option",
                "tool_call": messages[6]["tool_calls"][0],
                "env_action": f"click[{option_id}]",
                "done": False,
            },
            {
                "tool_name": "buy_now",
                "tool_call": messages[8]["tool_calls"][0],
                "env_action": "click[Buy Now]",
                "done": True,
            },
        ],
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v4",
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "purchase_success": True,
                "termination_reason": "gold_purchase",
            },
        },
    }


def _guard_recovery_trajectory(task_id=1):
    trajectory = _accepted_trajectory(task_id)
    blocked = _assistant_tool(
        "open_product",
        {"asin": "999999999999"},
        "blocked-1",
    )
    rejection = _tool_message(
        "blocked-1",
        "open_product",
        "Action Guard rejected a non-clickable ASIN",
    )
    rejection[RUNTIME_GUARD_FIELD] = True
    trajectory["messages"][2:2] = [blocked, rejection]
    trajectory["blocked_tool_calls"] = [
        {"tool_call": blocked["tool_calls"][0], "reason": "not_clickable"}
    ]
    return trajectory


def _write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class SftCollectionTests(unittest.TestCase):
    def test_accepts_only_strict_reward_v4_gold_purchase(self):
        accepted, reasons = acceptance_reasons(_accepted_trajectory())
        self.assertTrue(accepted)
        self.assertEqual(reasons, [])
        self.assertEqual(build_sft_row(_accepted_trajectory())["messages"][-1]["content"], "购买已完成。")

        invalid = _accepted_trajectory()
        invalid["terminal_result"]["reward_detail"]["reward_valid"] = False
        accepted, reasons = acceptance_reasons(invalid)
        self.assertFalse(accepted)
        self.assertIn("reward_v4_invalid", reasons)

    def test_redundant_terminal_fields_and_release_error_are_warnings(self):
        trajectory = _accepted_trajectory()
        trajectory["status"] = "assistant_final"
        trajectory["done"] = False
        trajectory["release_error"] = {"type": "OSError", "message": "release failed"}
        reward = trajectory["terminal_result"]["reward_detail"]
        reward["purchase_success"] = False
        reward["termination_reason"] = "stale_copy"

        accepted, reasons = acceptance_reasons(trajectory)
        warnings = result_consistency_warnings(trajectory)

        self.assertTrue(accepted)
        self.assertEqual(reasons, [])
        self.assertEqual(
            warnings,
            [
                "status_not_done",
                "trajectory_not_done",
                "purchase_not_successful",
                "termination_not_gold_purchase",
                "release_error",
            ],
        )

    def test_process_gate_keeps_and_sanitizes_guard_recovery(self):
        trajectory = _guard_recovery_trajectory()

        accepted, reasons = acceptance_reasons(trajectory)
        row = build_sft_row(trajectory)
        payload = json.dumps(row, ensure_ascii=False)

        self.assertTrue(accepted)
        self.assertEqual(reasons, [])
        self.assertNotIn("blocked-1", payload)
        self.assertNotIn("Action Guard rejected", payload)

    def test_process_gate_allows_non_consecutive_search_reformulation(self):
        trajectory = _accepted_trajectory()
        repeated_search = deepcopy(trajectory["steps"][0])
        repeated_search["tool_call"]["id"] = "call-reformulated"
        trajectory["steps"].insert(2, repeated_search)

        accepted, reasons = acceptance_reasons(trajectory)

        self.assertTrue(accepted)
        self.assertEqual(reasons, [])

    def test_process_gate_still_rejects_consecutive_duplicate_action(self):
        trajectory = _accepted_trajectory()
        trajectory["steps"].extend(
            [deepcopy(trajectory["steps"][0]), deepcopy(trajectory["steps"][0])]
        )

        accepted, reasons = acceptance_reasons(trajectory)

        self.assertFalse(accepted)
        self.assertIn("process_consecutive_duplicate_action", reasons)

    def test_process_gate_sanitizes_duplicate_noop_with_identical_state(self):
        trajectory = _accepted_trajectory()
        state = {
            "page_type": "search_results",
            "query": "乳胶枕",
            "actions": ["100000000001"],
        }
        trajectory["steps"][0]["result"] = {"observation_state": state, "over": False}
        duplicate_step = deepcopy(trajectory["steps"][0])
        duplicate_step["tool_call"]["id"] = "call-duplicate"
        duplicate_step["result"] = {"observation_state": deepcopy(state), "over": False}
        trajectory["steps"].insert(1, duplicate_step)
        duplicate_assistant = deepcopy(trajectory["messages"][2])
        duplicate_assistant["tool_calls"][0]["id"] = "call-duplicate"
        duplicate_tool = deepcopy(trajectory["messages"][3])
        duplicate_tool["tool_call_id"] = "call-duplicate"
        trajectory["messages"][4:4] = [duplicate_assistant, duplicate_tool]

        accepted, reasons = acceptance_reasons(trajectory)
        row = build_sft_row(trajectory)
        payload = json.dumps(row, ensure_ascii=False)

        self.assertTrue(accepted)
        self.assertEqual(reasons, [])
        self.assertEqual(sanitizable_duplicate_call_ids(trajectory), {"call-duplicate"})
        self.assertNotIn("call-duplicate", payload)
        self.assertEqual(payload.count("call-1"), 2)

    def test_process_gate_rejects_duplicate_action_when_state_changes(self):
        trajectory = _accepted_trajectory()
        trajectory["steps"][0]["result"] = {
            "observation_state": {"page_type": "product_detail"},
            "over": False,
        }
        duplicate_step = deepcopy(trajectory["steps"][0])
        duplicate_step["tool_call"]["id"] = "call-state-changing-duplicate"
        duplicate_step["result"] = {
            "observation_state": {"page_type": "search_results"},
            "over": False,
        }
        trajectory["steps"].insert(1, duplicate_step)

        accepted, reasons = acceptance_reasons(trajectory)

        self.assertFalse(accepted)
        self.assertIn("process_consecutive_duplicate_action", reasons)
        self.assertEqual(sanitizable_duplicate_call_ids(trajectory), set())

    def test_collection_summary_separates_the_two_gates(self):
        result_failure = _accepted_trajectory(2)
        result_failure["terminal_result"]["reward_detail"]["reward_valid"] = False
        recovery = _guard_recovery_trajectory(3)
        process_failure = _accepted_trajectory(4)
        process_failure["tool_call_truncations"] = [{"step_index": 0}]
        consistency_warning = _accepted_trajectory(5)
        consistency_warning["release_error"] = {
            "type": "OSError",
            "message": "release failed",
        }
        duplicate_recovery = _accepted_trajectory(6)
        duplicate_state = {"page_type": "search_results", "query": "乳胶枕"}
        duplicate_recovery["steps"][0]["result"] = {
            "observation_state": duplicate_state,
            "over": False,
        }
        duplicate_step = deepcopy(duplicate_recovery["steps"][0])
        duplicate_step["tool_call"]["id"] = "call-duplicate-summary"
        duplicate_step["result"] = {
            "observation_state": deepcopy(duplicate_state),
            "over": False,
        }
        duplicate_recovery["steps"].insert(1, duplicate_step)
        duplicate_assistant = deepcopy(duplicate_recovery["messages"][2])
        duplicate_assistant["tool_calls"][0]["id"] = "call-duplicate-summary"
        duplicate_tool = deepcopy(duplicate_recovery["messages"][3])
        duplicate_tool["tool_call_id"] = "call-duplicate-summary"
        duplicate_recovery["messages"][4:4] = [duplicate_assistant, duplicate_tool]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "raw.jsonl"
            _write_jsonl(
                raw,
                [
                    _accepted_trajectory(1),
                    result_failure,
                    recovery,
                    process_failure,
                    consistency_warning,
                    duplicate_recovery,
                ],
            )
            summary = build_collection_artifacts(
                raw_path=raw,
                output_dir=root / "derived",
                validation_ratio=0.0,
            )

        self.assertEqual(summary["accepted"], 4)
        self.assertEqual(summary["result_gate_rejected"], 1)
        self.assertEqual(summary["process_gate_rejected"], 1)
        self.assertEqual(summary["recoverable_guard_trajectories"], 1)
        self.assertEqual(summary["recoverable_guard_calls_sanitized"], 1)
        self.assertEqual(summary["consistency_warning_trajectories"], 1)
        self.assertEqual(summary["consistency_warnings"], {"release_error": 1})
        self.assertEqual(summary["recoverable_duplicate_trajectories"], 1)
        self.assertEqual(summary["recoverable_duplicate_calls_sanitized"], 1)
        self.assertEqual(
            summary["teacher_selection"],
            "shopping-teacher-recoverable-process-v4",
        )

    def test_sft_row_removes_reasoning_and_terminal_reward(self):
        row = build_sft_row(_accepted_trajectory())
        payload = json.dumps(row, ensure_ascii=False)

        self.assertNotIn("Teacher private reasoning", payload)
        self.assertNotIn("reasoning_content", payload)
        self.assertNotIn("Reward: 1.0", payload)
        self.assertEqual(row["messages"][-1]["content"], "购买已完成。")
        self.assertTrue(row["tools"])

    def test_build_artifacts_excludes_held_out_tasks_and_splits_by_task(self):
        rejected = deepcopy(_accepted_trajectory(4))
        rejected["status"] = "error"
        rejected["error"] = {"type": "RuntimeError", "message": "boom"}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "raw.jsonl"
            output = root / "derived"
            _write_jsonl(
                raw,
                [
                    _accepted_trajectory(1),
                    _accepted_trajectory(2),
                    _accepted_trajectory(3),
                    rejected,
                ],
            )

            summary = build_collection_artifacts(
                raw_path=raw,
                output_dir=output,
                held_out_task_ids={3},
                validation_ratio=0.5,
                seed=42,
                collection_config={"model": "teacher-test"},
            )

            train = [
                json.loads(line)
                for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            validation = [
                json.loads(line)
                for line in (output / "validation.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rejected_rows = [
                json.loads(line)
                for line in (output / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            metadata_exists = (output / "metadata.json").exists()
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))

        train_ids = {row["task_id"] for row in train}
        validation_ids = {row["task_id"] for row in validation}
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(train_ids | validation_ids, {1, 2})
        self.assertEqual(summary["accepted"], 2)
        self.assertEqual(summary["held_out_excluded"], 1)
        self.assertEqual(summary["rejected"], 1)
        self.assertIn("held_out_task", rejected_rows[0]["reject_reasons"])
        self.assertTrue(metadata_exists)
        self.assertEqual(metadata["collection_config"]["model"], "teacher-test")


if __name__ == "__main__":
    unittest.main()
