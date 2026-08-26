"""Frozen ShopSimulator Environment v2.4 contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


MANIFEST_VERSION = "shopping-environment-manifest-v1"
EMBEDDED_SOURCE_FILE = "EMBEDDED_SOURCE.json"
REQUIRED_KEYS = {
    "manifest_version",
    "environment_version",
    "shopsimulator_commit",
    "product_data_sha256",
    "reward_feature_version",
    "runtime_files_sha256",
    "search",
    "reward",
    "observation_version",
    "tool_version",
    "max_steps",
    "seed",
}

# Every file that can change the observable Environment v2.4 / Reward v4
# behavior must be frozen here.  Keeping this list next to manifest validation
# prevents launchers from validating only the top-level reward module while
# silently mixing old comparators, price semantics or alias tables.
RUNTIME_CONTRACT_FILES = {
    "actions.py": "src/shopping_grpo/environment/actions.py",
    "candidate_memory.py": "src/shopping_grpo/environment/candidate_memory.py",
    "brand_aliases.json": "environments/ShopSimulator/shop_env/configs/brand_aliases.json",
    "comparators.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/comparators.py",
    "config.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/config.py",
    "constraints.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/constraints.py",
    "engine.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/engine.py",
    "goal.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/goal.py",
    "observation.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/observation.py",
    "observation_projection.py": "src/shopping_grpo/environment/projection.py",
    "pack_api.py": "environments/ShopSimulator/shop_env/shop_env/pack_api.py",
    "price_semantics.py": "src/shopping_grpo/price_semantics.py",
    "public_observation.py": "src/shopping_grpo/environment/observation.py",
    "reward.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py",
    "reward_features.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/reward_features.py",
    "search.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/search.py",
    "semantic_aliases.json": "environments/ShopSimulator/shop_env/configs/semantic_aliases.json",
    "shop_agent.py": "environments/ShopSimulator/shop_env/shop_env/shop_agent.py",
    "slot_lease_pool.py": "environments/ShopSimulator/shop_env/shop_env/slot_lease_pool.py",
    "task_annotation_repairs.json": "environments/ShopSimulator/shop_env/configs/task_annotation_repairs.json",
    "termination.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/termination.py",
    "tools.json": "configs/tools.json",
    "tools.py": "src/shopping_grpo/environment/tools.py",
    "variant_price.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/variant_price.py",
    "web_agent_text_env.py": "environments/ShopSimulator/shop_env/web_agent_site/envs/web_agent_text_env.py",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shopsimulator_source_commit(repository):
    repository = Path(repository)
    embedded_source = repository / EMBEDDED_SOURCE_FILE
    if not embedded_source.is_file():
        raise ValueError(f"missing embedded source metadata: {embedded_source}")
    try:
        metadata = json.loads(embedded_source.read_text(encoding="utf-8"))
        commit = metadata["source_commit"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid embedded ShopSimulator source metadata: {exc}") from exc
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("embedded ShopSimulator commit is not a lowercase Git SHA")
    return commit


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("environment manifest must be an object")
    missing = REQUIRED_KEYS - set(manifest)
    if missing:
        raise ValueError(
            "environment manifest is missing: " + ", ".join(sorted(missing))
        )
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ValueError("unsupported environment manifest version")
    if manifest["observation_version"] != "shopping-observation-v2":
        raise ValueError("manifest does not select Observation v2")
    if manifest["tool_version"] != "shopping-tools-v2":
        raise ValueError("manifest does not select Tool v2")
    environment_version = manifest.get(
        "environment_version",
        "shopsimulator-environment-v2.4",
    )
    if environment_version != "shopsimulator-environment-v2.4":
        raise ValueError("manifest has an unsupported environment_version")
    if manifest["reward"].get("version") != "shopsimulator-reward-v4":
        raise ValueError(
            "shopsimulator-environment-v2.4 requires shopsimulator-reward-v4"
        )
    if manifest.get("reward_feature_version") != "shopping-reward-features-v2":
        raise ValueError(
            "shopsimulator-environment-v2.4 requires shopping-reward-features-v2"
        )
    runtime_hashes = manifest.get("runtime_files_sha256")
    if not isinstance(runtime_hashes, dict):
        raise ValueError("manifest runtime_files_sha256 must be an object")
    missing_runtime_files = set(RUNTIME_CONTRACT_FILES) - set(runtime_hashes)
    extra_runtime_files = set(runtime_hashes) - set(RUNTIME_CONTRACT_FILES)
    if missing_runtime_files or extra_runtime_files:
        raise ValueError(
            "manifest runtime_files_sha256 does not match the runtime dependency closure: "
            + json.dumps(
                {
                    "missing": sorted(missing_runtime_files),
                    "extra": sorted(extra_runtime_files),
                },
                sort_keys=True,
            )
        )
    for name, digest in runtime_hashes.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"manifest runtime digest is invalid: {name}")
    if manifest["search"].get("version") != "shopsimulator-multifield-bm25-v2.1":
        raise ValueError("manifest does not select multi-field BM25 v2.1")
    if int(manifest["search"].get("page_size", 0)) != 20:
        raise ValueError("Environment v2 page_size must equal 20")
    if int(manifest["max_steps"]) <= 0:
        raise ValueError("max_steps must be positive")
    for name in (
        "shopsimulator_commit",
        "product_data_sha256",
    ):
        value = manifest[name]
        expected_length = 40 if name.endswith("_commit") else 64
        if (
            not isinstance(value, str)
            or len(value) != expected_length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"manifest {name} is not a lowercase hexadecimal digest")
    return manifest


def validate_runtime_files(manifest, repository_root):
    """Verify the complete checked-in runtime dependency closure."""

    validate_manifest(manifest)
    repository_root = Path(repository_root)
    mismatches = {}
    for name, relative_path in RUNTIME_CONTRACT_FILES.items():
        path = repository_root / relative_path
        if not path.is_file():
            mismatches[name] = {"expected": manifest["runtime_files_sha256"][name], "actual": None}
            continue
        actual = sha256_file(path)
        expected = manifest["runtime_files_sha256"][name]
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ValueError(
            "Environment v2.4 runtime file hash mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return manifest
