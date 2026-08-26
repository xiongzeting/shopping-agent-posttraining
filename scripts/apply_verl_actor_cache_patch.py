#!/usr/bin/env python3
"""Apply or restore the pinned veRL 0.8 actor-cache wake-up patch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED_VERL_VERSION = "0.8.0"
EXPECTED_ORIGINAL_SHA256 = "721f866442475a08768854b58176a4dcd99de4564809e3d067dced7cba1122db"
EXPECTED_PATCHED_SHA256 = "7806d9635a7108d64fffb3d9f13aabda7fd611450105875073a62da033d4b06e"
PATCH_MARKER = "SHOPPING_GRPO_ACTOR_CACHE_BEFORE_VLLM_WAKEUP_V1"
BACKUP_SUFFIX = ".shopping-grpo-actor-cache.orig"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = PROJECT_ROOT / "patches/verl-0.8.0-actor-cache-before-vllm-wakeup.patch"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_installed_engine_workers() -> Path:
    installed_distribution = importlib.metadata.distribution("verl")
    installed_version = installed_distribution.version
    if installed_version != EXPECTED_VERL_VERSION:
        raise RuntimeError(
            f"expected verl=={EXPECTED_VERL_VERSION}, got verl=={installed_version}"
        )
    verl_source = Path(installed_distribution.locate_file("verl/__init__.py")).resolve()
    expected_environment = (PROJECT_ROOT / ".venv").resolve()
    if not verl_source.is_relative_to(expected_environment):
        raise RuntimeError(f"verl.__file__ is not from the project environment: {verl_source}")
    target = verl_source.parent / "workers" / "engine_workers.py"
    if not target.is_file():
        raise RuntimeError(f"installed engine_workers.py does not exist: {target}")
    return target.resolve()


def validate_runtime_and_target(target_override: Path | None) -> Path:
    installed_target = resolve_installed_engine_workers()
    if target_override is None:
        return installed_target
    target = target_override.resolve()
    if not target.is_file():
        raise RuntimeError(f"target engine_workers.py does not exist: {target}")
    return target


def verify_patched(target: Path) -> None:
    target_hash = sha256(target)
    if target_hash != EXPECTED_PATCHED_SHA256:
        raise RuntimeError(
            "patched engine_workers.py hash mismatch: "
            f"expected {EXPECTED_PATCHED_SHA256}, got {target_hash}"
        )
    if PATCH_MARKER not in target.read_text(encoding="utf-8"):
        raise RuntimeError(f"patched engine_workers.py is missing marker {PATCH_MARKER}")
    py_compile.compile(str(target), doraise=True)


def apply_patch(target: Path) -> None:
    target_hash = sha256(target)
    if target_hash == EXPECTED_PATCHED_SHA256:
        verify_patched(target)
        print(f"veRL actor-cache patch already applied: {target}")
        return
    if target_hash != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(
            "refusing to patch unknown engine_workers.py: "
            f"expected original SHA256 {EXPECTED_ORIGINAL_SHA256}, got {target_hash}"
        )
    if not PATCH_FILE.is_file():
        raise RuntimeError(f"patch file is missing: {PATCH_FILE}")
    patch_program = shutil.which("patch")
    if patch_program is None:
        raise RuntimeError("required system 'patch' executable is unavailable")

    backup = Path(str(target) + BACKUP_SUFFIX)
    if backup.exists() and sha256(backup) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"refusing to overwrite invalid backup: {backup}")
    if not backup.exists():
        shutil.copy2(target, backup)

    try:
        subprocess.run(
            [patch_program, "--batch", "--forward", "--silent", str(target), str(PATCH_FILE)],
            check=True,
            cwd=PROJECT_ROOT,
        )
        verify_patched(target)
    except Exception:
        shutil.copy2(backup, target)
        raise

    print(f"applied veRL actor-cache patch: {target}")
    print(f"backup: {backup}")
    print(f"patched_sha256: {sha256(target)}")


def restore_patch(target: Path) -> None:
    backup = Path(str(target) + BACKUP_SUFFIX)
    if sha256(target) == EXPECTED_ORIGINAL_SHA256:
        print(f"veRL engine_workers.py is already original: {target}")
        return
    if not backup.is_file() or sha256(backup) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"cannot restore from a verified backup: {backup}")
    restore_temp = target.with_name(target.name + ".shopping-grpo-restore.tmp")
    shutil.copy2(backup, restore_temp)
    restore_temp.replace(target)
    py_compile.compile(str(target), doraise=True)
    print(f"restored original veRL engine_workers.py: {target}")


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
            print(f"verified veRL actor-cache patch: {target}")
        else:
            apply_patch(target)
    except (OSError, RuntimeError, subprocess.CalledProcessError, py_compile.PyCompileError) as exc:
        raise SystemExit(f"veRL actor-cache patch error: {exc}") from exc


if __name__ == "__main__":
    main()
