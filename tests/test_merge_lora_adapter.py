"""验证 LoRA 合并入口的纯配置与 artifact 校验，不需要加载模型。"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_lora_adapter import (
    _validate_adapter,
    build_merge_manifest,
    choose_model_class,
)


class _Config:
    def __init__(self, model_type):
        self.model_type = model_type


class MergeLoraAdapterTest(unittest.TestCase):
    @staticmethod
    def _sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _write_json(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def test_qwen35_uses_multimodal_model_class(self):
        self.assertEqual(choose_model_class(_Config("qwen3_5"), "causal", "multimodal"), "multimodal")
        self.assertEqual(choose_model_class(_Config("qwen3"), "causal", "multimodal"), "causal")

    def test_merge_manifest_is_auditable(self):
        manifest = build_merge_manifest(
            base_model="Qwen/Qwen3.5-2B",
            adapter_path="checkpoints/sft",
            output_path="checkpoints/sft_merged",
            model_type="qwen3_5",
        )
        self.assertEqual(manifest["operation"], "peft_merge_and_unload")
        self.assertEqual(manifest["schema_version"], "shopping-sft-merge-manifest-v1")
        self.assertEqual(manifest["source"]["adapter"], "checkpoints/sft")
        self.assertEqual(manifest["output"], "checkpoints/sft_merged")

    def test_adapter_validation_detects_changed_processor_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            adapter = root / "adapter"
            base.mkdir()
            adapter.mkdir()
            adapter_config = {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj"],
                "base_model_name_or_path": str(base),
            }
            self._write_json(adapter / "adapter_config.json", adapter_config)
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            (adapter / "chat_template.jinja").write_text("original", encoding="utf-8")
            contract = {
                "status": "passed",
                "checks": {
                    "datasets": {
                        split: {"sha256": split}
                        for split in ("train", "validation", "evaluation")
                    },
                    "model": {
                        "metadata_sha256": {"config.json": "hash"},
                        "weight_files": [{"name": "model.safetensors", "sha256": "hash"}],
                        "lora_target_parameter_counts": {"q_proj": 1},
                    },
                },
            }
            self._write_json(adapter / "training_contract.json", contract)
            artifact_names = (
                "adapter_config.json",
                "adapter_model.safetensors",
                "chat_template.jinja",
                "training_contract.json",
            )
            artifacts = [
                {
                    "name": name,
                    "bytes": (adapter / name).stat().st_size,
                    "sha256": self._sha256(adapter / name),
                }
                for name in artifact_names
            ]
            summary = {
                "schema_version": "shopping-sft-train-summary-v1",
                "status": "completed",
                "optimizer_steps": 177,
                "validation_examples": 1,
                "last_eval": {"eval_loss": 1.0},
                "recipe": {"name": "canonical", "canonical": True},
                "arguments": {
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj"],
                    "max_steps": -1,
                },
                "data": {
                    "train": {"sha256": "train"},
                    "validation": {"sha256": "validation"},
                },
                "training_contract": {
                    "sha256": self._sha256(adapter / "training_contract.json")
                },
                "adapter_artifacts": artifacts,
            }
            self._write_json(adapter / "train_summary.json", summary)
            preflight = {
                "status": "passed",
                "checks": {
                    "datasets": contract["checks"]["datasets"],
                    "model": {"path": str(base), **contract["checks"]["model"]},
                },
            }
            preflight_path = root / "preflight.json"
            self._write_json(preflight_path, preflight)

            validation = _validate_adapter(
                adapter,
                base.resolve(),
                (root / "merged").resolve(),
                preflight_path,
            )
            self.assertTrue(validation["canonical"])

            (adapter / "chat_template.jinja").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate_adapter(
                    adapter,
                    base.resolve(),
                    (root / "merged").resolve(),
                    preflight_path,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
