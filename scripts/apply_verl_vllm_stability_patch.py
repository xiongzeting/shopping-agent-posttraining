#!/usr/bin/env python3
"""Apply or restore the pinned veRL 0.8 single-GPU vLLM stability patch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import py_compile
import shutil
from pathlib import Path

EXPECTED_VERL_VERSION = "0.8.0"
EXPECTED_ORIGINAL_SHA256 = "c7aafaa923edb7ab19c6a3d147643013be687df76d79ef38e855958d8382c68c"
EXPECTED_PATCHED_SHA256 = "50145626788a3385df6f9decf73098dd64858468c82f2fd7a5b809293ce78525"
PATCH_MARKER = "SHOPPING_GRPO_VLLM_SINGLE_GPU_STABILITY_PATCH_V2"
BACKUP_SUFFIX = ".shopping-grpo-vllm-stability.orig"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = PROJECT_ROOT / "patches/verl-0.8.0-vllm-single-gpu-stability.patch"

EXECUTOR_ORIGINAL = """        compilation_config = json.dumps(compilation_config)
        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "distributed_executor_backend": "mp",
"""
EXECUTOR_PATCHED = """        compilation_config = json.dumps(compilation_config)
        single_gpu_executor = (
            self.config.tensor_model_parallel_size == 1
            and self.config.data_parallel_size == 1
            and self.nnodes == 1
            and self.gpus_per_node == 1
        )
        distributed_executor_backend = "uni" if single_gpu_executor else "mp"
        logger.info(
            "vLLM distributed executor backend: %s (single_gpu=%s)",
            distributed_executor_backend,
            single_gpu_executor,
        )  # SHOPPING_GRPO_VLLM_SINGLE_GPU_STABILITY_PATCH_V2
        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "distributed_executor_backend": distributed_executor_backend,
"""
GENERATION_ORIGINAL = """        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None
"""
GENERATION_PATCHED = """        # Never allow an EngineCore stall to leave every AgentLoop request
        # waiting forever. Abort the individual request, then return an empty,
        # explicitly invalid trajectory so dynamic sampling can retry it.
        async def collect_final_response() -> Optional[RequestOutput]:
            final_response: Optional[RequestOutput] = None
            async for output in generator:
                final_response = output
            return final_response

        timeout_seconds = float(os.getenv("SHOPPING_GRPO_VLLM_GENERATION_TIMEOUT_SECONDS", "180"))
        if timeout_seconds <= 0:
            raise ValueError("SHOPPING_GRPO_VLLM_GENERATION_TIMEOUT_SECONDS must be positive")
        try:
            final_res = await asyncio.wait_for(collect_final_response(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            try:
                abort_result = await asyncio.wait_for(
                    self.abort_request(request_id, reset_prefix_cache=False),
                    timeout=10,
                )
            except asyncio.TimeoutError:
                abort_result = {"aborted": False, "error": "abort timed out after 10 seconds"}
            except Exception as abort_error:
                abort_result = {"aborted": False, "error": repr(abort_error)}
            message = (
                f"SHOPPING_GRPO_VLLM_GENERATION_TIMEOUT request_id={request_id} "
                f"timeout_seconds={timeout_seconds:g} abort_result={abort_result}"
            )
            logger.error(message)
            return TokenOutput(
                token_ids=[],
                log_probs=None,
                routed_experts=None,
                stop_reason="aborted",
                extra_fields={
                    "global_steps": self.global_steps,
                    "shopping_generation_timeout": True,
                    "shopping_generation_timeout_message": message,
                },
            )
        assert final_res is not None
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_installed_vllm_async_server() -> Path:
    installed_distribution = importlib.metadata.distribution("verl")
    installed_version = installed_distribution.version
    if installed_version != EXPECTED_VERL_VERSION:
        raise RuntimeError(f"expected verl=={EXPECTED_VERL_VERSION}, got verl=={installed_version}")
    verl_source = Path(installed_distribution.locate_file("verl/__init__.py")).resolve()
    expected_environment = (PROJECT_ROOT / ".venv").resolve()
    if not verl_source.is_relative_to(expected_environment):
        raise RuntimeError(f"verl.__file__ is not from the project environment: {verl_source}")
    target = verl_source.parent / "workers" / "rollout" / "vllm_rollout" / "vllm_async_server.py"
    if not target.is_file():
        raise RuntimeError(f"installed vllm_async_server.py does not exist: {target}")
    return target.resolve()


def validate_runtime_and_target(target_override: Path | None) -> Path:
    if target_override is None:
        return resolve_installed_vllm_async_server()
    target = target_override.resolve()
    if not target.is_file():
        raise RuntimeError(f"target vllm_async_server.py does not exist: {target}")
    return target


def patch_source(source: str) -> str:
    if PATCH_MARKER in source:
        return source
    if source.count(EXECUTOR_ORIGINAL) != 1:
        raise RuntimeError("cannot find the unique veRL vLLM executor anchor")
    if source.count(GENERATION_ORIGINAL) != 1:
        raise RuntimeError("cannot find the unique veRL vLLM generation anchor")
    return source.replace(EXECUTOR_ORIGINAL, EXECUTOR_PATCHED).replace(
        GENERATION_ORIGINAL,
        GENERATION_PATCHED,
    )


def verify_patched(target: Path) -> None:
    target_hash = sha256(target)
    if target_hash != EXPECTED_PATCHED_SHA256:
        raise RuntimeError(
            "patched vllm_async_server.py hash mismatch: "
            f"expected {EXPECTED_PATCHED_SHA256}, got {target_hash}"
        )
    source = target.read_text(encoding="utf-8")
    required_fragments = (
        PATCH_MARKER,
        'distributed_executor_backend = "uni" if single_gpu_executor else "mp"',
        'os.getenv("SHOPPING_GRPO_VLLM_GENERATION_TIMEOUT_SECONDS", "180")',
        "await asyncio.wait_for(collect_final_response(), timeout=timeout_seconds)",
        "self.abort_request(request_id, reset_prefix_cache=False)",
        '"shopping_generation_timeout": True',
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise RuntimeError(f"patched vllm_async_server.py is incomplete: {missing}")
    py_compile.compile(str(target), doraise=True)


def apply_patch(target: Path) -> None:
    target_hash = sha256(target)
    if target_hash == EXPECTED_PATCHED_SHA256:
        verify_patched(target)
        print(f"veRL vLLM stability patch already applied: {target}")
        return
    if target_hash != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(
            "refusing to patch unknown vllm_async_server.py: "
            f"expected original SHA256 {EXPECTED_ORIGINAL_SHA256}, got {target_hash}"
        )
    if not PATCH_FILE.is_file():
        raise RuntimeError(f"patch file is missing: {PATCH_FILE}")

    backup = Path(str(target) + BACKUP_SUFFIX)
    if backup.exists() and sha256(backup) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"refusing to overwrite invalid backup: {backup}")
    if not backup.exists():
        shutil.copy2(target, backup)

    temporary = target.with_name(target.name + ".shopping-grpo-patch.tmp")
    try:
        original = target.read_bytes().decode("utf-8")
        temporary.write_bytes(patch_source(original).encode("utf-8"))
        temporary.replace(target)
        verify_patched(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(backup, target)
        raise

    print(f"applied veRL vLLM single-GPU stability patch: {target}")
    print(f"backup: {backup}")
    print(f"patched_sha256: {sha256(target)}")


def restore_patch(target: Path) -> None:
    backup = Path(str(target) + BACKUP_SUFFIX)
    if sha256(target) == EXPECTED_ORIGINAL_SHA256:
        print(f"veRL vllm_async_server.py is already original: {target}")
        return
    if not backup.is_file() or sha256(backup) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"cannot restore from a verified backup: {backup}")
    temporary = target.with_name(target.name + ".shopping-grpo-restore.tmp")
    shutil.copy2(backup, temporary)
    temporary.replace(target)
    if sha256(target) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"restore verification failed: {target}")
    py_compile.compile(str(target), doraise=True)
    print(f"restored original veRL vllm_async_server.py: {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--target", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum((args.restore, args.check)) > 1:
        raise SystemExit("--restore and --check are mutually exclusive")
    try:
        target = validate_runtime_and_target(args.target)
        if args.restore:
            restore_patch(target)
        elif args.check:
            verify_patched(target)
            print(f"verified veRL vLLM stability patch: {target}")
        else:
            apply_patch(target)
    except (OSError, RuntimeError, py_compile.PyCompileError) as exc:
        raise SystemExit(f"veRL vLLM stability patch error: {exc}") from exc


if __name__ == "__main__":
    main()
