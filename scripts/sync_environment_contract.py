#!/usr/bin/env python3
"""Check or refresh the frozen Environment v2.4 repository manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.environment.manifest import (
    RUNTIME_CONTRACT_FILES,
    sha256_file,
    validate_manifest,
    validate_runtime_files,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/environment.json"
RUNTIME_CONFIG = ROOT / "environments/ShopSimulator/shop_env/configs/environment.json"


def expected_contract() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runtime_config = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    manifest["reward"] = runtime_config["reward"]
    manifest["runtime_files_sha256"] = {
        name: sha256_file(ROOT / relative_path)
        for name, relative_path in RUNTIME_CONTRACT_FILES.items()
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check or refresh data/environment.json from frozen runtime files."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Atomically update the manifest; otherwise perform a read-only check.",
    )
    args = parser.parse_args()

    current = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = expected_contract()
    if current != expected:
        if not args.write:
            changed = []
            if current.get("reward") != expected["reward"]:
                changed.append("reward")
            current_hashes = current.get("runtime_files_sha256") or {}
            for name, digest in expected["runtime_files_sha256"].items():
                if current_hashes.get(name) != digest:
                    changed.append(f"runtime_files_sha256.{name}")
            for name in set(current_hashes) - set(expected["runtime_files_sha256"]):
                changed.append(f"runtime_files_sha256.{name}")
            raise SystemExit(
                "environment contract is stale; run with --write. mismatches="
                + json.dumps(sorted(changed), ensure_ascii=False)
            )
        temporary = MANIFEST.with_suffix(MANIFEST.suffix + ".tmp")
        temporary.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(MANIFEST)

    validated = validate_manifest(expected)
    validate_runtime_files(validated, ROOT)
    print(
        json.dumps(
            {
                "status": "synchronized",
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "runtime_file_count": len(RUNTIME_CONTRACT_FILES),
                "reward": expected["reward"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
