"""End-to-end checks for the unified Benchmark v2.1 evaluation path."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_shop_benchmark
from shopping_grpo.evaluation.comparison import compare_evaluation_runs
from shopping_grpo.evaluation.pipeline import evaluate_trajectories
from shopping_grpo.evaluation.rollout import OpenAIChatClient


def _trajectory(task_id=1, *, success=True, trajectory_id=None):
    reward_type = "gold_purchase" if success else "wrong_purchase"
    return {
        "trajectory_id": trajectory_id or f"trajectory-{task_id}",
        "task_id": task_id,
        "attempt_index": 0,
        "created_at": "2026-08-06T00:00:00+00:00",
        "status": "done",
        "done": True,
        "final_reward": 1.0 if success else -0.85,
        "messages": [],
        "initial_result": {
            "instruction": f"购买测试商品 {task_id}",
            "environment_version": "shopsimulator-environment-v2.4",
            "reward_version": "shopsimulator-reward-v4",
        },
        "steps": [
            {
                "step_index": 0,
                "tool_call": {
                    "id": "call-1",
                    "function": {
                        "name": "search_products",
                        "arguments": '{"query":"测试商品"}',
                    },
                },
                "tool_name": "search_products",
                "parameters": {"query": "测试商品"},
                "env_action": "search[测试商品]",
                "observation": "100000000001|测试商品",
                "projection": {
                    "truncated": True,
                    "raw_tokens": 200,
                    "visible_tokens": 100,
                    "critical_footer_preserved": True,
                },
                "reward": 0.0,
                "done": False,
                "result": {},
                "tool_latency_seconds": 0.25,
            }
        ],
        "blocked_tool_calls": [],
        "tool_call_truncations": [],
        "context_compactions": [],
        "context_turn_tokens": [{"step_index": 0, "input_tokens": 800}],
        "context_budget": {
            "context_window": 1536,
            "reserved_completion_tokens": 512,
            "safety_margin_tokens": 0,
            "max_input_tokens": 1000,
        },
        "model_calls": [
            {
                "step_index": 0,
                "latency_seconds": 0.5,
                "attempts": 1,
                "usage": {
                    "prompt_tokens": 800,
                    "completion_tokens": 20,
                    "total_tokens": 820,
                    "prompt_tokens_details": {"cached_tokens": 100},
                },
            }
        ],
        "timing": {"trajectory_duration_seconds": 1.0},
        "terminal_result": {
            "done": True,
            "over": True,
            "reward": 1.0 if success else -0.85,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v4",
                "query_constraint_version": "shopping-query-constraints-v1",
                "constraint_results": [],
                "constraint_summary": {
                    "total": 0,
                    "status_counts": {"pass": 0, "fail": 0, "unverifiable": 0},
                },
                "reward_type": reward_type,
                "reward_valid": True,
                "purchase_success": success,
                "termination_reason": reward_type,
                "terminal_utility": 1.0 if success else -0.85,
                "weighted_score": 1.0 if success else 0.0,
            },
        },
        "error": None,
        "release_error": None,
    }


class BenchmarkV2PipelineTest(unittest.TestCase):
    def test_lightweight_token_counter_uses_tokenizer_json_without_transformers(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tokenizer.json"
            tokenizer = Tokenizer(
                models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")
            )
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(path))

            chat_counter, text_counter = evaluate_shop_benchmark._lightweight_token_counters(
                path
            )

            self.assertGreater(text_counter("hello world"), 0)
            self.assertGreater(
                chat_counter([{"role": "user", "content": "hello"}], []),
                0,
            )

    def test_actor_client_records_seed_usage_and_latency(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update(payload)
            return {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }

        client = OpenAIChatClient(
            model="test",
            base_url="http://localhost/v1",
            api_key="secret",
            seed=123,
            transport=transport,
        )
        client.complete([{"role": "user", "content": "test"}], [])

        self.assertEqual(captured["seed"], 123)
        self.assertEqual(client.last_call_metrics["usage"]["total_tokens"], 12)
        self.assertGreaterEqual(client.last_call_metrics["latency_seconds"], 0.0)

    def test_pipeline_keeps_missing_tasks_in_denominator_and_reports_telemetry(self):
        result = evaluate_trajectories(
            expected_task_ids=[1, 2],
            trajectories=[_trajectory(1)],
            actor={"label": "test"},
        )

        summary = result["summary"]
        self.assertEqual(summary["missing_task_ids"], [2])
        self.assertEqual(summary["reward_and_terminal"]["gold_purchase_rate"], 0.5)
        self.assertEqual(summary["trajectory_quality"]["judge_status_counts"], {"not_judged": 1})
        deterministic = result["deterministic_metrics"][0]
        self.assertEqual(deterministic["context"]["completion_tokens"], 20)
        self.assertEqual(deterministic["context"]["max_context_usage_ratio"], 0.8)
        self.assertEqual(deterministic["timing"]["model_latency_seconds"], 0.5)
        self.assertEqual(
            summary["deterministic"]["executed_steps_successful_tasks"]["median"],
            1,
        )

    def test_comparison_treats_missing_tasks_as_strict_failures(self):
        baseline = evaluate_trajectories(
            expected_task_ids=[1],
            trajectories=[_trajectory(1, success=True)],
            actor={"label": "baseline"},
        )["evaluations"]
        sft = evaluate_trajectories(
            expected_task_ids=[2],
            trajectories=[_trajectory(2, success=True)],
            actor={"label": "sft"},
        )["evaluations"]
        comparison = compare_evaluation_runs(
            expected_task_ids=[1, 2],
            runs={"baseline": baseline, "sft": sft},
        )

        pair = comparison["pairwise"]["baseline_to_sft"]
        self.assertEqual(
            pair["reward_and_terminal"]["strict_success_transitions"],
            {"failure_to_success": 1, "success_to_failure": 1},
        )
        self.assertEqual(pair["both_completed_tasks"], 0)
        self.assertTrue(comparison["models"]["baseline"]["missing_tasks_count_as_failures"])

    def test_formal_cli_writes_complete_run_directory_without_secrets_or_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark = root / "tasks.jsonl"
            benchmark.write_text('{"task_id":1}\n{"task_id":2}\n', encoding="utf-8")
            environment = root / "environment.json"
            environment.write_text("{}\n", encoding="utf-8")
            run_dir = root / "run"
            metadata = {
                "tasks": 2,
                "task_sha256": "task-hash",
                "slice_sha256": "slice-hash",
                "source_pool": "test",
                "environment": "shopsimulator-environment-v2.4",
                "reward": "shopsimulator-reward-v4",
                "termination": "shopping-termination-v3.1",
                "observation": "shopping-observation-v2",
                "tool_schema": "shopping-tools-v2",
            }
            slices = {
                1: {"suite": "core", "domain": "测试", "challenge_slice": None},
                2: {"suite": "challenge", "domain": "测试", "challenge_slice": "long_horizon"},
            }
            environment_manifest = {
                "environment_version": "shopsimulator-environment-v2.4",
                "reward": {"version": "shopsimulator-reward-v4"},
            }

            def fake_collect(tasks, client, output_path, **kwargs):
                Path(output_path).write_text(
                    json.dumps(_trajectory(1), ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                return []

            argv = [
                "evaluate_shop_benchmark.py",
                "--benchmark",
                str(benchmark),
                "--benchmark-metadata",
                str(root / "metadata.json"),
                "--benchmark-slices",
                str(root / "slices.jsonl"),
                "--environment-manifest",
                str(environment),
                "--run-dir",
                str(run_dir),
                "--model",
                "test-model",
                "--llm-base-url",
                "http://127.0.0.1:8000/v1",
                "--api-key",
                "TOP-SECRET",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    evaluate_shop_benchmark,
                    "validate_canonical_benchmark_files",
                    return_value=(metadata, slices),
                ),
                patch.object(evaluate_shop_benchmark, "guard_blind_final"),
                patch.object(
                    evaluate_shop_benchmark,
                    "validate_manifest",
                    return_value=environment_manifest,
                ),
                patch.object(evaluate_shop_benchmark, "collect_tasks", side_effect=fake_collect),
            ):
                evaluate_shop_benchmark.main()

            expected_files = {
                "run_manifest.json",
                "trajectories.jsonl",
                "evaluations.jsonl",
                "summary.json",
            }
            self.assertEqual({path.name for path in run_dir.iterdir()}, expected_files)
            manifest_text = (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            self.assertNotIn("TOP-SECRET", manifest_text)
            self.assertNotIn(str(root), manifest_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
