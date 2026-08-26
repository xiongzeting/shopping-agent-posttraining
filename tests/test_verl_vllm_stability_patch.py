"""CPU-only checks for the pinned veRL single-GPU vLLM stability patch."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import apply_verl_vllm_stability_patch as stability_patch
from scripts.check_grpo_runtime import (
    VLLM_STABILITY_PATCH_MARKER,
    validate_vllm_stability_patch,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class VerlVllmStabilityPatchTest(unittest.TestCase):
    def test_apply_is_idempotent_and_restore_recovers_original(self):
        original = (
            "# fixture\n"
            + stability_patch.EXECUTOR_ORIGINAL
            + "            # executor tail\n"
            + stability_patch.GENERATION_ORIGINAL
            + "        # generation tail\n"
        ).encode("utf-8")
        patched = stability_patch.patch_source(original.decode("utf-8")).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "vllm_async_server.py"
            target.write_bytes(original)
            with (
                patch.object(
                    stability_patch,
                    "EXPECTED_ORIGINAL_SHA256",
                    _digest(original),
                ),
                patch.object(
                    stability_patch,
                    "EXPECTED_PATCHED_SHA256",
                    _digest(patched),
                ),
                patch.object(
                    stability_patch,
                    "PATCH_FILE",
                    Path(__file__),
                ),
                patch.object(stability_patch.py_compile, "compile"),
            ):
                stability_patch.apply_patch(target)
                first_result = target.read_bytes()
                stability_patch.apply_patch(target)
                self.assertEqual(target.read_bytes(), first_result)
                self.assertIn(stability_patch.PATCH_MARKER.encode(), first_result)
                self.assertIn(b'distributed_executor_backend = "uni"', first_result)
                self.assertIn(b"asyncio.wait_for", first_result)
                self.assertIn(b'"shopping_generation_timeout": True', first_result)
                self.assertNotIn(b"raise TimeoutError", first_result)

                stability_patch.restore_patch(target)
                self.assertEqual(target.read_bytes(), original)

    def test_patch_source_rejects_incomplete_or_unknown_source(self):
        with self.assertRaisesRegex(RuntimeError, "executor anchor"):
            stability_patch.patch_source("# unknown\n")

        incomplete = stability_patch.EXECUTOR_ORIGINAL + "# no generation anchor\n"
        with self.assertRaisesRegex(RuntimeError, "generation anchor"):
            stability_patch.patch_source(incomplete)

    def test_production_hashes_match_locked_verl_wheel_source(self):
        root = Path(__file__).resolve().parents[1]
        extracted = (
            root
            / ".codex-tmp"
            / "verl-0.8.0-source"
            / "original"
            / "verl"
            / "workers"
            / "rollout"
            / "vllm_rollout"
            / "vllm_async_server.py"
        )
        if not extracted.is_file():
            self.skipTest("locked veRL wheel source is not present in the local verification cache")
        original = extracted.read_bytes()
        self.assertEqual(_digest(original), stability_patch.EXPECTED_ORIGINAL_SHA256)
        transformed = stability_patch.patch_source(original.decode("utf-8")).encode("utf-8")
        self.assertEqual(_digest(transformed), stability_patch.EXPECTED_PATCHED_SHA256)

    def test_preflight_rejects_missing_marker_and_accepts_locked_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            verl_source = Path(temp_dir) / "verl" / "__init__.py"
            server_source = (
                verl_source.parent / "workers" / "rollout" / "vllm_rollout" / "vllm_async_server.py"
            )
            server_source.parent.mkdir(parents=True)
            verl_source.write_text("", encoding="utf-8")
            server_source.write_text("# unpatched\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "stability patch is missing"):
                validate_vllm_stability_patch(verl_source, {"verl": "0.8.0"})

            server_source.write_text(
                f"# {VLLM_STABILITY_PATCH_MARKER}\n",
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                    "SHOPPING_GRPO_VLLM_GENERATION_TIMEOUT_SECONDS": "180",
                },
            ):
                validate_vllm_stability_patch(verl_source, {"verl": "0.8.0"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
