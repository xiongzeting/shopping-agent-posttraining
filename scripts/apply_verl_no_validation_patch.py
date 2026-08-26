#!/usr/bin/env python3
"""Make veRL 0.8 skip validation dataset construction when val_files is null."""

from __future__ import annotations

import argparse
import hashlib
import py_compile
import shutil
from pathlib import Path


MARKER = "SHOPPING_GRPO_OPTIONAL_VALIDATION_PATCH_V1"
BACKUP_SUFFIX = ".shopping-grpo-optional-validation.orig"

MAIN_ORIGINAL = """        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
"""
MAIN_PATCHED = """        val_dataset = None
        if config.data.get("val_files"):
            val_dataset = create_rl_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor,
                is_train=False,
                max_samples=config.data.get("val_max_samples", -1),
            )  # SHOPPING_GRPO_OPTIONAL_VALIDATION_PATCH_V1
"""

RAY_ORIGINAL = """        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
"""
RAY_PATCHED = """        if val_dataset is None and self.config.data.get("val_files"):
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset  # SHOPPING_GRPO_OPTIONAL_VALIDATION_PATCH_V1
"""

RAY_LOADER_ORIGINAL = """        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )
"""
RAY_LOADER_PATCHED = """        self.val_dataloader = None
        if self.val_dataset is not None:
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)

            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        if self.val_dataloader is not None:
            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        val_loader_size = len(self.val_dataloader) if self.val_dataloader is not None else 0
        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{val_loader_size}"
        )
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(source: str, original: str, patched: str, label: str) -> str:
    if patched in source:
        return source
    if source.count(original) != 1:
        raise RuntimeError(f"cannot find unique {label} anchor")
    return source.replace(original, patched)


def patch_file(path: Path, replacements: list[tuple[str, str, str]]) -> None:
    source = path.read_text(encoding="utf-8")
    patched = source
    for original, replacement, label in replacements:
        patched = replace_once(patched, original, replacement, label)
    backup = Path(str(path) + BACKUP_SUFFIX)
    if patched != source:
        if not backup.exists():
            shutil.copy2(path, backup)
        temporary = path.with_name(path.name + ".shopping-grpo-patch.tmp")
        temporary.write_text(patched, encoding="utf-8")
        temporary.replace(path)
    py_compile.compile(str(path), doraise=True)
    if MARKER not in path.read_text(encoding="utf-8"):
        raise RuntimeError(f"patch marker missing from {path}")
    print(f"verified optional-validation patch: {path} sha256={sha256(path)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, default=Path(".venv"))
    args = parser.parse_args()
    site = args.venv.resolve() / "lib" / "python3.12" / "site-packages" / "verl" / "trainer"
    main_ppo = site / "main_ppo.py"
    ray_trainer = site / "ppo" / "ray_trainer.py"
    if not main_ppo.is_file() or not ray_trainer.is_file():
        raise SystemExit(f"veRL trainer files not found below {site}")
    patch_file(main_ppo, [(MAIN_ORIGINAL, MAIN_PATCHED, "main_ppo validation dataset")])
    patch_file(
        ray_trainer,
        [
            (RAY_ORIGINAL, RAY_PATCHED, "ray_trainer validation dataset"),
            (RAY_LOADER_ORIGINAL, RAY_LOADER_PATCHED, "ray_trainer validation dataloader"),
        ],
    )


if __name__ == "__main__":
    main()
