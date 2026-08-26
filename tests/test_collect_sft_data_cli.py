import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.collect_sft_data import (
    TEACHER_PROMPT_VERSION,
    _lightweight_token_counters,
    _validate_args,
    batch_paths,
    collect_until_target,
    collection_system_prompt,
    parse_args,
    teacher_prompt_manifest,
)


class CollectSftDataCliTests(unittest.TestCase):
    def test_teacher_prompt_uses_sufficient_evidence_not_forced_length(self):
        prompt = collection_system_prompt()

        self.assertIn("最短的充分证据轨迹", prompt)
        self.assertIn("轨迹长度必须由任务难度自然产生", prompt)
        self.assertIn("第一个看似匹配的候选不自动等于最佳候选", prompt)
        self.assertNotIn("11～20", prompt)
        self.assertNotIn("至少 21", prompt)

    def test_natural_strategy_prompts_change_method_without_forcing_steps(self):
        comparison = collection_system_prompt("candidate_comparison")
        evidence = collection_system_prompt("evidence_verification")

        self.assertIn("至少两个不同商品", comparison)
        self.assertIn("标题和搜索摘要不能作为", evidence)
        self.assertNotIn("至少 21", comparison)
        self.assertNotIn("11～20", evidence)

    def test_repair_prompt_targets_convergence_near_miss_and_terminal_tools(self):
        prompt = collection_system_prompt("near_miss_rejection")

        self.assertIn("检索后无法收敛", prompt)
        self.assertIn("部分替代", prompt)
        self.assertIn("下一次 assistant 行为必须直接调用 `buy_now`", prompt)
        self.assertIn("近似候选拒绝", prompt)

    def test_unknown_teacher_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown Teacher strategy"):
            collection_system_prompt("unsupported")

    def test_prompt_manifest_hashes_every_exact_variant(self):
        manifest = teacher_prompt_manifest()

        self.assertEqual(manifest["schema_version"], TEACHER_PROMPT_VERSION)
        self.assertEqual(len(manifest["base_sha256"]), 64)
        self.assertEqual(
            set(manifest["strategy_sha256"]),
            {
                "focused_verification",
                "search_reformulation",
                "candidate_comparison",
                "evidence_verification",
                "price_semantics",
                "multi_option",
                "loop_recovery",
                "near_miss_rejection",
                "terminal_tool_commit",
                "option_grounding",
            },
        )
        self.assertTrue(all(len(value) == 64 for value in manifest["strategy_sha256"].values()))

    def test_defaults_match_the_current_teacher_collection_contract(self):
        with patch.object(sys, "argv", ["collect_sft_data.py", "--tasks", "tasks.jsonl"]):
            args = parse_args()

        self.assertEqual(args.model, "deepseek-v4-flash")
        self.assertEqual(args.base_url, "http://127.0.0.1:5700")
        self.assertEqual(args.max_steps, 45)
        self.assertEqual(args.output_dir, Path("outputs/sft-collection"))
        self.assertFalse(hasattr(args, "length_policy"))
        self.assertFalse(hasattr(args, "target_length_bucket"))

    def test_teacher_profile_environment_is_used_without_overwriting_cli(self):
        environment = {
            "SHOPPING_TEACHER_MODEL": "deepseek-v4-flash",
            "SHOPPING_TEACHER_BASE_URL": "https://teacher.test/v1",
            "SHOPPING_TEACHER_API_KEY": "teacher-secret",
        }
        with (
            patch.dict("os.environ", environment, clear=False),
            patch.object(sys, "argv", ["collect_sft_data.py", "--tasks", "tasks.jsonl"]),
        ):
            args = parse_args()

        self.assertEqual(args.model, "deepseek-v4-flash")
        self.assertEqual(args.llm_base_url, "https://teacher.test/v1")
        self.assertEqual(args.api_key, "teacher-secret")

    def test_batch_paths_keep_raw_and_derived_files_together(self):
        paths = batch_paths(Path("outputs/example"))

        self.assertEqual(paths["raw"], Path("outputs/example/raw.jsonl"))
        self.assertEqual(paths["accepted"], Path("outputs/example/accepted.jsonl"))
        self.assertEqual(paths["train"], Path("outputs/example/train.jsonl"))
        self.assertEqual(paths["validation"], Path("outputs/example/validation.jsonl"))

    def test_lightweight_counter_loads_tokenizer_json(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            tokenizer = Tokenizer(
                models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")
            )
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(path))
            chat_counter, text_counter = _lightweight_token_counters(str(path))

            self.assertGreater(text_counter("hello world"), 0)
            self.assertGreater(
                chat_counter([{"role": "user", "content": "hello"}], []),
                0,
            )

    def test_build_only_needs_neither_tasks_nor_model_credentials(self):
        with patch.object(
            sys,
            "argv",
            ["collect_sft_data.py", "--build-only"],
        ):
            args = parse_args()

        _validate_args(args)
        self.assertIsNone(args.tasks)

    def test_collection_stops_at_the_accepted_target(self):
        rows = [
            {"trajectory_id": "one", "task_id": 1},
            {"trajectory_id": "two", "task_id": 2},
        ]
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("scripts.collect_sft_data.collect_for_task", side_effect=rows) as collect,
            patch("scripts.collect_sft_data.acceptance_reasons", return_value=(True, [])),
        ):
            written, accepted = collect_until_target(
                tasks=[{"task_id": 1}, {"task_id": 2}, {"task_id": 3}],
                target_accepted=2,
                client=object(),
                output_path=Path(tmpdir) / "raw.jsonl",
                base_url="http://shop.test",
                max_steps=45,
                attempts_per_task=1,
                workers=1,
            )

        self.assertEqual([row["trajectory_id"] for row in written], ["one", "two"])
        self.assertEqual(accepted, 2)
        self.assertEqual(collect.call_count, 2)

    def test_collection_tries_unique_tasks_before_retrying(self):
        calls = []

        def collect(task, *, attempt_index, **kwargs):
            calls.append((task["task_id"], attempt_index))
            return {
                "trajectory_id": f"trajectory-{task['task_id']}-{attempt_index}",
                "task_id": task["task_id"],
            }

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("scripts.collect_sft_data.collect_for_task", side_effect=collect),
            patch("scripts.collect_sft_data.acceptance_reasons", return_value=(True, [])),
        ):
            written, accepted = collect_until_target(
                tasks=[{"task_id": 1}, {"task_id": 2}, {"task_id": 3}],
                target_accepted=2,
                client=object(),
                output_path=Path(tmpdir) / "raw.jsonl",
                base_url="http://shop.test",
                max_steps=45,
                attempts_per_task=2,
                workers=1,
            )

        self.assertEqual(calls, [(1, 0), (2, 0)])
        self.assertEqual([row["task_id"] for row in written], [1, 2])
        self.assertEqual(accepted, 2)

    def test_accepted_target_ignores_held_out_rows_already_in_raw(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = Path(tmpdir) / "raw.jsonl"
            raw.write_text(json.dumps({"task_id": 1}) + "\n", encoding="utf-8")
            with (
                patch(
                    "scripts.collect_sft_data.collect_for_task",
                    return_value={"trajectory_id": "two", "task_id": 2},
                ) as collect,
                patch(
                    "scripts.collect_sft_data.acceptance_reasons",
                    return_value=(True, []),
                ),
            ):
                written, accepted = collect_until_target(
                    tasks=[{"task_id": 2}],
                    target_accepted=1,
                    client=object(),
                    output_path=raw,
                    base_url="http://shop.test",
                    max_steps=45,
                    attempts_per_task=1,
                    workers=1,
                    excluded_task_ids={1},
                )

        self.assertEqual([row["task_id"] for row in written], [2])
        self.assertEqual(accepted, 1)
        self.assertEqual(collect.call_count, 1)

    def test_parallel_collection_uses_an_independent_client_per_trajectory(self):
        clients = []

        def client_factory():
            client = object()
            clients.append(client)
            return client

        def collect(task, *, client, **kwargs):
            return {
                "trajectory_id": f"trajectory-{task['task_id']}",
                "task_id": task["task_id"],
                "client_id": id(client),
            }

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("scripts.collect_sft_data.collect_for_task", side_effect=collect),
            patch("scripts.collect_sft_data.acceptance_reasons", return_value=(True, [])),
        ):
            written, _ = collect_until_target(
                tasks=[{"task_id": 1}, {"task_id": 2}],
                target_accepted=2,
                client=None,
                client_factory=client_factory,
                output_path=Path(tmpdir) / "raw.jsonl",
                base_url="http://shop.test",
                max_steps=45,
                attempts_per_task=1,
                workers=2,
            )

        self.assertEqual(len(clients), 2)
        self.assertEqual(len({row["client_id"] for row in written}), 2)


if __name__ == "__main__":
    unittest.main()
