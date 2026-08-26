"""Public CPU and parameterized GRPO entry-point tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train_grpo import build_command, parse_args, reset_environment_leases
from shopping_grpo.cli import main as cli_main
from shopping_grpo.smoke import run_cpu_smoke


class PublicEntrypointTest(unittest.TestCase):
    def test_grpo_launcher_resets_stale_environment_leases(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(
                    {
                        "result": {
                            "message": "All environments have been initialized",
                            "environment_slots": 20,
                            "free_environment_slots": 20,
                        }
                    }
                ).encode("utf-8")

        with patch("scripts.train_grpo.urllib.request.urlopen", return_value=Response()) as request:
            result = reset_environment_leases("http://shop.test/")

        self.assertIn("initialized", result["message"])
        sent_request = request.call_args.args[0]
        self.assertEqual(sent_request.full_url, "http://shop.test/api/shop_agent")
        self.assertEqual(json.loads(sent_request.data.decode("utf-8")), {"action": "release_all"})

    def test_cpu_smoke_covers_public_contracts(self):
        result = run_cpu_smoke()

        self.assertEqual(
            result["checks"],
            [
                "action_schema",
                "trajectory_normalization",
                "reward_sample",
                "sft_label_mask",
                "dynamic_sampling_grouping",
            ],
        )

    def test_offline_example_cli_runs_without_models_or_environment(self):
        root = Path(__file__).resolve().parents[1]
        with patch.object(
            sys,
            "argv",
            [
                "shopping-grpo",
                "evaluate",
                str(root / "examples/trajectories.jsonl"),
            ],
        ), patch("builtins.print") as output:
            cli_main()

        summary = json.loads(output.call_args.args[0])
        self.assertEqual(summary["trajectory_count"], 3)
        self.assertEqual(summary["strict_gold_success_count"], 1)

    def test_public_grpo_launcher_accepts_sharded_weights_and_console(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                "{}",
                encoding="utf-8",
            )
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--logger",
                    "console",
                    "--dry-run",
                ],
            ):
                args = parse_args()
            command, environment = build_command(args)

        self.assertIn("verl.trainer.main_ppo", command)
        self.assertEqual(environment["GRPO_MODEL_PATH"], str(model))
        self.assertEqual(environment["GRPO_TRAIN_FILE"], str(train))
        self.assertEqual(environment["GRPO_VAL_FILE"], str(validation))
        self.assertEqual(environment["VLLM_ENABLE_V1_MULTIPROCESSING"], "0")
        self.assertEqual(
            environment["SHOPPING_GRPO_VLLM_GENERATION_TIMEOUT_SECONDS"],
            "10000",
        )
        self.assertIn("trainer.logger=[console]", command)


if __name__ == "__main__":
    unittest.main()
