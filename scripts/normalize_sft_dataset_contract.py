#!/usr/bin/env python3
"""Normalize a generated SFT dataset to LF and refresh its immutable hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def write_json_atomic(path: Path, value: dict) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    write_bytes_atomic(path, payload)


def main() -> int:
    args = parse_args()
    dataset = args.dataset_dir.resolve()
    for name in ("all.jsonl", "train.jsonl", "validation.jsonl", "data_gate.json"):
        path = dataset / name
        write_bytes_atomic(path, path.read_bytes().replace(b"\r\n", b"\n"))

    metadata_path = dataset / "metadata.json"
    audit_path = dataset / "token_audit.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for name in ("all", "train", "validation"):
        digest = sha256(dataset / f"{name}.jsonl")
        metadata[name]["sha256"] = digest
        audit["files"][f"{name}_sha256"] = digest
    metadata["data_gate"]["sha256"] = sha256(dataset / "data_gate.json")
    write_json_atomic(audit_path, audit)
    metadata["token_audit"]["sha256"] = sha256(audit_path)
    write_json_atomic(metadata_path, metadata)
    print(json.dumps(metadata["token_audit"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
