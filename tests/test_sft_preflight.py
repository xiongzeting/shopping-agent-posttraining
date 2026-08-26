"""验证 canonical SFT 训练前的数据与本地模型检查。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.check_sft_runtime import (
    ROOT,
    _nvidia_smi_snapshot,
    _runtime_package_names,
    build_report,
    parse_args,
    sha256_file,
    validate_datasets,
    validate_model,
)
from shopping_grpo.collection.data_gate import DATA_GATE_VERSION, DEFAULT_POLICY


def _passing_data_gate_report():
    return {
        "schema_version": DATA_GATE_VERSION,
        "status": "passed",
        "policy": deepcopy(DEFAULT_POLICY),
        "rows": 1000,
        "unique_task_ids": 1000,
        "deficits": {},
        "audit": {
            "collection_schema_version": "shopping-sft-collection-v3",
            "teacher_selection": "shopping-teacher-recoverable-process-v4",
            "search_contract": "shopsimulator-multifield-bm25-v2.1",
            "audited_rows": 1000,
            "input_sha256": "1" * 64,
            "products_sha256": "2" * 64,
            "search_index_sha256": "3" * 64,
        },
    }


class SftPreflightTest(unittest.TestCase):
    def test_nvidia_smi_snapshot_tolerates_platform_permission_denial(self):
        with patch("scripts.check_sft_runtime.subprocess.run", side_effect=PermissionError):
            self.assertEqual(_nvidia_smi_snapshot(), {})

    def test_flash_attention_recipe_requires_flash_attn_runtime_package(self):
        args = SimpleNamespace(
            recipe_variant="flash-attn2+liger",
            attention_implementation="flash_attention_2",
        )

        package_names = _runtime_package_names(args)

        self.assertIn("flash-attn", package_names)
        self.assertIn("liger-kernel", package_names)

    def test_runtime_only_does_not_require_data_or_model(self):
        with patch.object(
            sys,
            "argv",
            ["check_sft_runtime.py", "--runtime-only", "--skip-gpu-check"],
        ):
            args = parse_args()

        with (
            patch(
                "scripts.check_sft_runtime.validate_runtime",
                return_value={"runtime": "passed"},
            ) as runtime_check,
            patch("scripts.check_sft_runtime.validate_datasets") as data_check,
            patch("scripts.check_sft_runtime.validate_model") as model_check,
            patch("scripts.check_sft_runtime._git_snapshot", return_value={}),
        ):
            report = build_report(args)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["checks"], {"runtime": {"runtime": "passed"}})
        runtime_check.assert_called_once()
        data_check.assert_not_called()
        model_check.assert_not_called()

    def _dataset_args(self, metadata):
        return SimpleNamespace(
            all_data=ROOT / "data/sft/all.jsonl",
            train=ROOT / "data/sft/train.jsonl",
            validation=ROOT / "data/sft/validation.jsonl",
            metadata=metadata,
            evaluation_tasks=ROOT / "data/evaluation/tasks.jsonl",
            evaluation_metadata=ROOT / "data/evaluation/metadata.json",
        )

    def test_checked_in_canonical_dataset_passes_the_current_contract(self):
        args = self._dataset_args(ROOT / "data/sft/metadata.json")
        errors = []

        result = validate_datasets(args, errors)

        self.assertEqual(result["all"]["rows"], 1000)
        self.assertEqual(result["train"]["rows"], 900)
        self.assertEqual(result["validation"]["rows"], 100)
        self.assertEqual(result["evaluation"]["rows"], 240)
        self.assertEqual(result["train_validation_overlap"], 0)
        self.assertEqual(result["evaluation_overlap"], 0)
        self.assertEqual(errors, [])
        self.assertEqual(result["data_gate"]["status"], "passed")

        metadata = json.loads((ROOT / "data/sft/metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "current")
        self.assertEqual(
            metadata["teacher_selection"], "shopping-teacher-recoverable-process-v4"
        )

    def test_promoted_dataset_requires_a_hashed_passing_data_gate_report(self):
        original = json.loads(
            (ROOT / "data/sft/metadata.json").read_text(encoding="utf-8")
        )
        report = _passing_data_gate_report()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "data_gate.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            metadata = {
                **original,
                "schema_version": "shopping-sft-dataset-v3",
                "status": "current",
                "contract": "environment-v2.4/reward-v3.2/sft-v3",
                "teacher_selection": "shopping-teacher-recoverable-process-v4",
                "data_gate": {
                    "schema_version": DATA_GATE_VERSION,
                    "status": "passed",
                    "path": "data_gate.json",
                    "sha256": sha256_file(report_path),
                },
            }
            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            args = self._dataset_args(metadata_path)
            errors = []

            result = validate_datasets(args, errors)

        self.assertEqual(errors, [])
        self.assertEqual(result["data_gate"]["status"], "passed")
        self.assertEqual(result["data_gate"]["rows"], 1000)

    def test_promoted_dataset_rejects_a_tampered_data_gate_report(self):
        original = json.loads(
            (ROOT / "data/sft/metadata.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "data_gate.json"
            report_path.write_text(
                json.dumps(_passing_data_gate_report(), indent=2) + "\n",
                encoding="utf-8",
            )
            metadata = {
                **original,
                "schema_version": "shopping-sft-dataset-v3",
                "status": "current",
                "teacher_selection": "shopping-teacher-recoverable-process-v4",
                "data_gate": {
                    "schema_version": DATA_GATE_VERSION,
                    "status": "passed",
                    "path": "data_gate.json",
                    "sha256": "0" * 64,
                },
            }
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            errors = []

            validate_datasets(self._dataset_args(metadata_path), errors)

        self.assertTrue(any("SHA-256" in error for error in errors))

    def test_checked_in_datasets_still_match_hashes_and_have_no_final_overlap(self):
        args = SimpleNamespace(
            all_data=ROOT / "data/sft/all.jsonl",
            train=ROOT / "data/sft/train.jsonl",
            validation=ROOT / "data/sft/validation.jsonl",
            metadata=ROOT / "data/sft/metadata.json",
            evaluation_tasks=ROOT / "data/evaluation/tasks.jsonl",
            evaluation_metadata=ROOT / "data/evaluation/metadata.json",
        )
        errors = []

        result = validate_datasets(args, errors)

        self.assertEqual(result["all"]["rows"], 1000)
        self.assertEqual(result["train"]["rows"], 900)
        self.assertEqual(result["validation"]["rows"], 100)
        self.assertEqual(result["evaluation"]["rows"], 240)
        self.assertEqual(result["train_validation_overlap"], 0)
        self.assertEqual(result["evaluation_overlap"], 0)
        self.assertEqual(
            result["train"]["tool_schema_sha256"],
            result["validation"]["tool_schema_sha256"],
        )

        metadata = json.loads((ROOT / "data/sft/metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["schema_version"], "shopping-sft-dataset-v3")
        self.assertEqual(metadata["environment"], "shopsimulator-environment-v2.4")
        self.assertEqual(metadata["reward"], "shopsimulator-reward-v3.2")
        self.assertEqual(metadata["termination"], "shopping-termination-v3.1")
        self.assertEqual(
            metadata["teacher_selection"], "shopping-teacher-recoverable-process-v4"
        )
        self.assertEqual(metadata["final_240_family_overlap"], 0)
        self.assertEqual(metadata["final_240_semantic_overlap"], 0)
        self.assertEqual(metadata["token_audit"]["status"], "passed")
        self.assertEqual(metadata["token_audit"]["expected_optimizer_steps"], 339)
        self.assertEqual(metadata["token_audit"]["max_length"], 30000)

        token_audit = json.loads(
            (ROOT / "data/sft/token_audit.json").read_text(encoding="utf-8")
        )
        self.assertEqual(token_audit["status"], "passed")
        self.assertEqual(token_audit["result"]["train"]["kept"], 900)
        self.assertEqual(token_audit["result"]["validation"]["kept"], 100)
        self.assertEqual(token_audit["result"]["expected_optimizer_steps"], 339)
        self.assertEqual(token_audit["result"]["max_length"], 30000)

    def test_local_model_check_rejects_missing_weight_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}),
                encoding="utf-8",
            )
            (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model / "processor_config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer": "model-00001-of-00002.safetensors"}}),
                encoding="utf-8",
            )
            args = SimpleNamespace(model=str(model), hash_model_weights=False)
            errors = []
            warnings = []

            with patch.dict("sys.modules", {"transformers": None}):
                result = validate_model(args, errors, warnings)

        self.assertTrue(result["exists"])
        self.assertTrue(any("权重分片缺失" in error for error in errors))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
