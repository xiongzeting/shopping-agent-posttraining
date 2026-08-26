#!/usr/bin/env python3
"""Freeze a passed no-GPU tokenizer preflight into the SFT data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed":
        raise SystemExit("preflight must pass before token audit finalization")
    tokenization = (preflight.get("checks") or {}).get("tokenization")
    if not isinstance(tokenization, dict) or tokenization.get("max_length") != 30000:
        raise SystemExit("preflight is missing the exact 30k tokenization audit")

    dataset = args.dataset_dir.resolve()
    metadata_path = dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = (preflight.get("checks") or {}).get("model") or {}
    runtime = (preflight.get("checks") or {}).get("runtime") or {}
    files = {
        name: {
            "rows": int(metadata[name]["rows"]),
            "sha256": sha256(dataset / f"{name}.jsonl"),
        }
        for name in ("all", "train", "validation")
    }
    for name, record in files.items():
        if record["sha256"] != metadata[name]["sha256"]:
            raise SystemExit(f"dataset changed after preflight: {name}")

    weights = model.get("weight_files") or []
    audit = {
        "schema_version": "shopping-sft-token-audit-v3",
        "status": "passed",
        "audited_at": preflight.get("created_at"),
        "scope": "model-and-tokenizer-only-no-gpu-streaming",
        "model": {
            "name": "Qwen3.5-2B",
            "snapshot": model.get("expected_snapshot"),
            "weight_sha256": weights[0].get("sha256") if weights else None,
            "chat_template_sha256": (model.get("metadata_sha256") or {}).get(
                "chat_template.jinja"
            ),
            "tokenizer_sha256": (model.get("metadata_sha256") or {}).get(
                "tokenizer.json"
            ),
            "tokenizer_config_sha256": (model.get("metadata_sha256") or {}).get(
                "tokenizer_config.json"
            ),
            "transformers_revision": (runtime.get("transformers_revision")),
        },
        "runtime_packages": runtime.get("packages"),
        "files": {
            "all_rows": files["all"]["rows"],
            "all_sha256": files["all"]["sha256"],
            "train_rows": files["train"]["rows"],
            "train_sha256": files["train"]["sha256"],
            "validation_rows": files["validation"]["rows"],
            "validation_sha256": files["validation"]["sha256"],
        },
        "result": tokenization,
        "remaining_training_host_gate": {
            "required": True,
            "checks": [
                "single BF16-capable GPU with at least 94 GiB total and 92 GiB free",
                "BF16 Flash Attention 2 CUDA forward/backward smoke test",
                "longest kept example no-gradient-checkpointing memory smoke test",
            ],
        },
    }
    audit_path = dataset / "token_audit.json"
    write_json_atomic(audit_path, audit)
    metadata["canonical_eligibility"] = "ready_for_gpu_runtime_preflight"
    metadata["token_audit"] = {
        "path": "token_audit.json",
        "status": "passed",
        "sha256": sha256(audit_path),
        "max_length": 30000,
        "train_kept": tokenization["train"]["kept"],
        "validation_kept": tokenization["validation"]["kept"],
        "expected_optimizer_steps": tokenization["expected_optimizer_steps"],
    }
    write_json_atomic(metadata_path, metadata)
    print(json.dumps(metadata["token_audit"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
