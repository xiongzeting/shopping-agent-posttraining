#!/usr/bin/env python3
"""在加载模型前拒绝污染或版本不匹配的 GRPO 环境。"""

from __future__ import annotations

import json
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

EXPECTED_VERSIONS = {
    "verl": "0.8.0",
    "vllm": "0.25.1",
    "torch": "2.11.0",
    "transformers": "5.15.0.dev0",
    "ray": "2.56.1",
    "tensordict": "0.10.0",
    "numpy": "2.2.6",
    "swanlab": "0.9.1",
}
EXPECTED_TRANSFORMERS_REVISION = "7ea2320c76117e6742364808a666ef6f2fb40a67"
PATCH_MARKER = "SHOPPING_GRPO_DYNAMIC_SAMPLING_PATCH_V6"
ACTOR_CACHE_PATCH_MARKER = "SHOPPING_GRPO_ACTOR_CACHE_BEFORE_VLLM_WAKEUP_V1"
VLLM_STABILITY_PATCH_MARKER = "SHOPPING_GRPO_VLLM_SINGLE_GPU_STABILITY_PATCH_V2"
VLLM_GENERATION_TIMEOUT_SECONDS = 10000
MAX_SAFE_RESPONSE_LENGTH = 25904
MAX_SAFE_SEQUENCE_LENGTH = 30000
ACTOR_DYNAMIC_TOKEN_BUDGET = 30000
LOG_PROB_DYNAMIC_TOKEN_BUDGET = 36000
ROLLOUT_BATCHED_TOKEN_BUDGET = 60000


def validate_training_data_contract(path: Path) -> None:
    """Reject legacy GRPO parquet files that can contain zero-success groups."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to audit GRPO training parquet") from exc

    row_count = 0
    invalid_task_ids = []
    try:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["extra_info"], batch_size=1024):
            for row in batch.to_pylist():
                row_count += 1
                info = row.get("extra_info") or {}
                successes = info.get("accepted_probe_purchase_successes")
                if not isinstance(successes, int) or successes < 1:
                    invalid_task_ids.append(info.get("task_id"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot audit GRPO training parquet {path}: {exc}") from exc

    if row_count == 0:
        raise SystemExit(f"GRPO training parquet is empty: {path}")
    if invalid_task_ids:
        preview = invalid_task_ids[:20]
        raise SystemExit(
            "GRPO training parquet is legacy or contains groups without a Gold/Valid "
            f"purchase; rebuild it before training. affected task IDs: {preview}"
        )
    print(
        "GRPO training data gate passed: "
        + json.dumps(
            {
                "path": str(path.resolve()),
                "rows": row_count,
                "all_groups_have_purchase_success": True,
            },
            sort_keys=True,
        )
    )


def validate_runtime_files(manifest, root):
    if manifest.get("lease_contract") != "explicit-client-release-v1":
        raise SystemExit(
            "Environment v2.4 manifest must select explicit-client-release-v1"
        )
    from shopping_grpo.environment.manifest import validate_runtime_files as validate

    try:
        validate(manifest, root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def validate_environment_contract():
    required_version = os.environ.get(
        "SHOPPING_ENVIRONMENT_VERSION",
        "shopsimulator-environment-v2.4",
    )
    if required_version != "shopsimulator-environment-v2.4":
        raise SystemExit(
            "this repository supports only shopsimulator-environment-v2.4"
        )
    manifest_path = os.environ.get("SHOPPING_ENV_MANIFEST")
    if not manifest_path or not Path(manifest_path).is_file():
        raise SystemExit(
            f"{required_version} requires SHOPPING_ENV_MANIFEST pointing to a frozen manifest"
        )
    try:
        from shopping_grpo.environment.manifest import validate_manifest

        manifest = validate_manifest(
            json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        )
    except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {required_version} manifest: {exc}") from exc
    actual_environment_version = manifest.get(
        "environment_version",
        "shopsimulator-environment-v2.4",
    )
    if actual_environment_version != required_version:
        raise SystemExit(
            "environment manifest version mismatch: "
            f"expected {required_version}, got {actual_environment_version}"
        )
    tools_path = Path(
        os.environ.get(
            "SHOPPING_TOOL_CONFIG",
            Path(__file__).resolve().parents[1]
            / "configs/tools.json",
        )
    )
    tools = json.loads(tools_path.read_text(encoding="utf-8")).get("tools", [])
    tool_names = {
        item.get("tool_schema", {}).get("function", {}).get("name")
        for item in tools
    }
    if "finish_without_purchase" not in tool_names:
        raise SystemExit("Environment v2 tool config is missing finish_without_purchase")
    if int(manifest["max_steps"]) != 45:
        raise SystemExit("Environment v2 GRPO contract requires max_steps=45")
    validate_runtime_files(
        manifest,
        Path(__file__).resolve().parents[1],
    )
    print(
        f"{required_version} manifest preflight passed: "
        + json.dumps(
            {
                "manifest": str(Path(manifest_path).resolve()),
                "shopsimulator_commit": manifest["shopsimulator_commit"],
                "observation_version": manifest["observation_version"],
                "reward_version": manifest["reward"]["version"],
                "search_version": manifest["search"]["version"],
                "lease_contract": manifest.get("lease_contract"),
                "runtime_file_count": len(manifest.get("runtime_files_sha256") or {}),
            },
            sort_keys=True,
        )
    )


def compose_runtime_config(overrides):
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
    except ImportError as exc:
        raise SystemExit(f"cannot parse GRPO config before preflight: {exc}") from exc

    GlobalHydra.instance().clear()
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    config_name = os.environ.get("GRPO_CONFIG_NAME", "grpo")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=list(overrides))


def validate_transformers_revision():
    """The Qwen3.5 runtime uses one pinned upstream Transformers revision."""
    dist = distribution("transformers")
    direct_url = Path(dist.locate_file("transformers-5.15.0.dev0.dist-info/direct_url.json"))
    if not direct_url.is_file():
        raise SystemExit(
            "cannot verify pinned Transformers revision: direct_url.json is missing"
        )
    try:
        metadata = json.loads(direct_url.read_text(encoding="utf-8"))
        revision = metadata["vcs_info"]["commit_id"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid Transformers direct_url.json: {exc}") from exc
    if revision != EXPECTED_TRANSFORMERS_REVISION:
        raise SystemExit(
            "incompatible Transformers revision: expected "
            f"{EXPECTED_TRANSFORMERS_REVISION}, got {revision}"
        )
    print(f"pinned Transformers revision preflight passed: {revision}")


def validate_dynamic_sampling(config, verl_source: Path, installed):
    dynamic_config = config.get("shopping_dynamic_sampling", {})
    if not bool(dynamic_config.get("enable", False)):
        return

    if installed.get("verl") != "0.8.0":
        raise SystemExit(
            f"shopping dynamic sampling requires verl==0.8.0, got {installed.get('verl')}"
        )
    ray_trainer = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
    if not ray_trainer.is_file():
        raise SystemExit(f"cannot locate installed RayPPOTrainer source: {ray_trainer}")
    if PATCH_MARKER not in ray_trainer.read_text(encoding="utf-8"):
        raise SystemExit(
            "shopping dynamic sampling is enabled but the pinned veRL patch marker is missing; "
            "run scripts/apply_verl_dynamic_sampling_patch.py first"
        )

    try:
        from shopping_grpo.training.grpo.dynamic_sampling import (
            extract_shopping_group_signals,
            select_reward_varying_groups,
        )
    except ImportError as exc:
        raise SystemExit(f"shopping dynamic sampling helper is unavailable: {exc}") from exc
    utility, success, invalid, reasons = extract_shopping_group_signals(
        [
            {
                "infrastructure_invalid": False,
                "reward": {
                    "terminal_utility": reward,
                    "purchase_success": reward > 0,
                    "sampling_invalid": False,
                },
            }
            for reward in (0.0, 1.0, 0.0, 0.0)
        ]
    )
    indices, _ = select_reward_varying_groups(
        ["preflight"] * 4,
        [0.0, 1.0, 0.0, 0.0],
        terminal_utilities=utility,
        purchase_success=success,
        sampling_invalid=invalid,
        sampling_invalid_reasons=reasons,
    )
    if indices != [0, 1, 2, 3]:
        raise SystemExit("shopping dynamic sampling helper failed its import-time sanity check")
    rejected_indices, rejected_stats = select_reward_varying_groups(
        ["zero-success"] * 4,
        [-0.85, -0.65, -0.50, -0.30],
        terminal_utilities=[-0.85, -0.65, -0.50, -0.30],
        purchase_success=[False] * 4,
        sampling_invalid=[False] * 4,
        sampling_invalid_reasons=[()] * 4,
        require_purchase_success=True,
    )
    if rejected_indices or rejected_stats["groups"][0]["drop_reason"] != "no_purchase_success":
        raise SystemExit(
            "shopping dynamic sampling helper accepted a zero-success reward group"
        )

    if dynamic_config.get("metric") != "seq_reward":
        raise SystemExit("shopping_dynamic_sampling.metric must be seq_reward")
    if int(dynamic_config.get("max_num_gen_batches", 0)) != 2:
        raise SystemExit("shopping_dynamic_sampling.max_num_gen_batches must equal 2")
    if not bool(dynamic_config.get("require_purchase_success", False)):
        raise SystemExit(
            "shopping_dynamic_sampling.require_purchase_success must be true"
        )
    reward_tolerance = float(dynamic_config.get("reward_tolerance", -1))
    if not math.isfinite(reward_tolerance) or reward_tolerance != 0.025:
        raise SystemExit("shopping_dynamic_sampling.reward_tolerance must equal 0.025")
    if not bool(config.algorithm.rollout_correction.get("bypass_mode", False)):
        raise SystemExit("shopping dynamic sampling requires rollout_correction.bypass_mode=true")
    if not bool(config.actor_rollout_ref.rollout.get("calculate_log_probs", False)):
        raise SystemExit("shopping dynamic sampling requires rollout.calculate_log_probs=true")

    print(
        "shopping dynamic sampling preflight passed: "
        + json.dumps(
            {
                "enable": True,
                "metric": str(dynamic_config.metric),
                "max_num_gen_batches": int(dynamic_config.max_num_gen_batches),
                "exhausted_batch_action": "skip_actor_update_and_advance_step",
                "reward_tolerance": reward_tolerance,
                "require_purchase_success": True,
                "ray_trainer": str(ray_trainer),
                "marker": PATCH_MARKER,
            },
            sort_keys=True,
        )
    )


def validate_actor_cache_cleanup_patch(verl_source: Path, installed):
    if installed.get("verl") != "0.8.0":
        raise SystemExit(
            "actor cache cleanup patch requires verl==0.8.0, "
            f"got {installed.get('verl')}"
        )
    engine_workers = verl_source.parent / "workers" / "engine_workers.py"
    if not engine_workers.is_file():
        raise SystemExit(f"cannot locate installed veRL engine worker source: {engine_workers}")
    if ACTOR_CACHE_PATCH_MARKER not in engine_workers.read_text(encoding="utf-8"):
        raise SystemExit(
            "the actor CUDA cache cleanup patch is missing; "
            "run scripts/apply_verl_actor_cache_patch.py first"
        )
    print(
        "actor CUDA cache cleanup preflight passed: "
        + json.dumps(
            {
                "engine_workers": str(engine_workers),
                "marker": ACTOR_CACHE_PATCH_MARKER,
                "timing": "immediately_before_vllm_weight_wakeup",
            },
            sort_keys=True,
        )
    )


def validate_vllm_stability_patch(verl_source: Path, installed):
    if installed.get("verl") != "0.8.0":
        raise SystemExit(
            f"single-GPU vLLM stability patch requires verl==0.8.0, got {installed.get('verl')}"
        )
    vllm_server = (
        verl_source.parent / "workers" / "rollout" / "vllm_rollout" / "vllm_async_server.py"
    )
    if not vllm_server.is_file():
        raise SystemExit(f"cannot locate installed veRL vLLM server source: {vllm_server}")
    source = vllm_server.read_text(encoding="utf-8")
    if VLLM_STABILITY_PATCH_MARKER not in source:
        raise SystemExit(
            "the single-GPU vLLM stability patch is missing; "
            "run scripts/apply_verl_vllm_stability_patch.py first"
        )
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise SystemExit("VLLM_ENABLE_V1_MULTIPROCESSING must equal 0")
    raw_timeout = os.environ.get("SHOPPING_GRPO_VLLM_GENERATION_TIMEOUT_SECONDS")
    try:
        timeout_seconds = float(raw_timeout or "")
    except ValueError as exc:
        raise SystemExit("SHOPPING_GRPO_VLLM_GENERATION_TIMEOUT_SECONDS must be numeric") from exc
    if not math.isfinite(timeout_seconds) or timeout_seconds != VLLM_GENERATION_TIMEOUT_SECONDS:
        raise SystemExit(
            "SHOPPING_GRPO_VLLM_GENERATION_TIMEOUT_SECONDS must equal "
            f"{VLLM_GENERATION_TIMEOUT_SECONDS}"
        )
    print(
        "single-GPU vLLM stability preflight passed: "
        + json.dumps(
            {
                "abort_timeout_seconds": 10,
                "executor": "uni on one GPU; mp otherwise",
                "generation_timeout_seconds": timeout_seconds,
                "marker": VLLM_STABILITY_PATCH_MARKER,
                "v1_multiprocessing": False,
                "vllm_server": str(vllm_server),
            },
            sort_keys=True,
        )
    )


def validate_swanlab_tracking(config):
    """Validate SwanLab only when the user explicitly enables it."""
    logger_backends = list(config.trainer.get("logger", []))
    if "swanlab" not in logger_backends:
        return
    forbidden = {"wandb", "tracking", "vemlp_wandb"} & set(logger_backends)
    if forbidden:
        raise SystemExit(
            "Reward v4 GRPO forbids W&B logger backends: "
            + ", ".join(sorted(forbidden))
        )
    if os.environ.get("SWANLAB_MODE") != "online":
        raise SystemExit("Reward v4 GRPO requires SWANLAB_MODE=online")
    if not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit(
            "Reward v4 GRPO requires SWANLAB_API_KEY in the launching environment"
        )
    log_dir = os.environ.get("SWANLAB_LOG_DIR")
    if not log_dir:
        raise SystemExit("Reward v4 GRPO requires SWANLAB_LOG_DIR")
    resolved_log_dir = Path(log_dir).resolve()
    if str(config.trainer.get("project_name")) != "shopping-grpo":
        raise SystemExit("Reward v4 GRPO SwanLab project must be shopping-grpo")
    print(
        "SwanLab online preflight passed: "
        + json.dumps(
            {
                "api_key": "present",
                "logger": logger_backends,
                "log_dir": str(resolved_log_dir),
                "mode": "online",
                "project": str(config.trainer.project_name),
                "run_name": str(config.trainer.experiment_name),
            },
            sort_keys=True,
        )
    )


def validate_training_memory_budget(config):
    prompt_length = int(config.data.max_prompt_length)
    response_length = int(config.data.max_response_length)
    total_length = prompt_length + response_length
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    reference = config.actor_rollout_ref.ref

    if str(actor.loss_agg_mode) != "seq-mean-token-mean":
        raise SystemExit(
            "actor.loss_agg_mode must equal seq-mean-token-mean so trajectories are equally weighted"
        )

    if int(config.data.train_batch_size) != 2:
        raise SystemExit("data.train_batch_size must equal 2")
    if int(actor.ppo_mini_batch_size) != 2:
        raise SystemExit("actor.ppo_mini_batch_size must equal 2")
    if int(rollout.n) != 4:
        raise SystemExit("rollout.n must equal 4")
    if int(rollout.max_num_seqs) != 8:
        raise SystemExit("rollout.max_num_seqs must equal 8")
    if int(rollout.agent.num_workers) != 8:
        raise SystemExit("rollout.agent.num_workers must equal 8")

    if prompt_length != 4096:
        raise SystemExit("GRPO max_prompt_length must equal 4096")
    if response_length != MAX_SAFE_RESPONSE_LENGTH:
        raise SystemExit(
            "unsafe or inconsistent GRPO response budget: "
            f"max_response_length must equal {MAX_SAFE_RESPONSE_LENGTH}, got {response_length}"
        )
    if total_length != MAX_SAFE_SEQUENCE_LENGTH:
        raise SystemExit(
            "unsafe or inconsistent GRPO sequence budget: "
            f"max_prompt_length + max_response_length = {total_length}, "
            f"required value is {MAX_SAFE_SEQUENCE_LENGTH}"
        )
    for name, value in (
        ("rollout.max_model_len", int(rollout.max_model_len)),
    ):
        if value != MAX_SAFE_SEQUENCE_LENGTH:
            raise SystemExit(
                f"unsafe or inconsistent GRPO memory budget: {name} must equal "
                f"{MAX_SAFE_SEQUENCE_LENGTH}, got {value}"
            )
    if int(rollout.max_num_batched_tokens) != ROLLOUT_BATCHED_TOKEN_BUDGET:
        raise SystemExit(
            "rollout.max_num_batched_tokens must equal "
            f"{ROLLOUT_BATCHED_TOKEN_BUDGET}"
        )
    actor_token_budget = int(actor.ppo_max_token_len_per_gpu)
    if actor_token_budget != ACTOR_DYNAMIC_TOKEN_BUDGET:
        raise SystemExit(
            "actor.ppo_max_token_len_per_gpu must equal actor dynamic token budget "
            f"{ACTOR_DYNAMIC_TOKEN_BUDGET}, got {actor_token_budget}"
        )
    for name, value in (
        (
            "rollout.log_prob_max_token_len_per_gpu",
            int(rollout.log_prob_max_token_len_per_gpu),
        ),
        (
            "ref.log_prob_max_token_len_per_gpu",
            int(reference.log_prob_max_token_len_per_gpu),
        ),
    ):
        if value != LOG_PROB_DYNAMIC_TOKEN_BUDGET:
            raise SystemExit(
                f"{name} must equal log-prob dynamic token budget "
                f"{LOG_PROB_DYNAMIC_TOKEN_BUDGET}, got {value}"
            )
    if not bool(actor.use_dynamic_bsz):
        raise SystemExit("actor.use_dynamic_bsz must be true")
    if actor.ppo_micro_batch_size_per_gpu is not None:
        raise SystemExit("actor.ppo_micro_batch_size_per_gpu must be null with dynamic batching")
    if not bool(rollout.log_prob_use_dynamic_bsz):
        raise SystemExit("rollout.log_prob_use_dynamic_bsz must be true")
    if rollout.log_prob_micro_batch_size_per_gpu is not None:
        raise SystemExit(
            "rollout.log_prob_micro_batch_size_per_gpu must be null with dynamic batching"
        )
    if not bool(reference.log_prob_use_dynamic_bsz):
        raise SystemExit("ref.log_prob_use_dynamic_bsz must be true")
    if reference.log_prob_micro_batch_size_per_gpu is not None:
        raise SystemExit(
            "ref.log_prob_micro_batch_size_per_gpu must be null with dynamic batching"
        )
    if bool(config.trainer.val_before_train):
        raise SystemExit("trainer.val_before_train must be false")
    if int(config.trainer.test_freq) not in {-1, 10}:
        raise SystemExit("trainer.test_freq must be -1 (disabled) or 10")

    print(
        "GRPO training memory budget preflight passed: "
        + json.dumps(
            {
                "max_prompt_length": prompt_length,
                "max_response_length": response_length,
                "max_sequence_length": total_length,
                "train_prompt_batch_size": 2,
                "rollouts_per_prompt": 4,
                "trajectories_per_step": 8,
                "rollout_max_num_seqs": 8,
                "rollout_batched_token_budget": ROLLOUT_BATCHED_TOKEN_BUDGET,
                "agent_workers": 8,
                "environment_slots": 20,
                "ppo_mini_batch_size": 2,
                "actor_dynamic_token_budget": ACTOR_DYNAMIC_TOKEN_BUDGET,
                "log_prob_dynamic_token_budget": LOG_PROB_DYNAMIC_TOKEN_BUDGET,
                "actor_dynamic_batch": True,
                "rollout_log_prob_dynamic_batch": True,
                "reference_dynamic_batch": True,
            },
            sort_keys=True,
        )
    )


def validate_acceleration_config(config):
    model = config.actor_rollout_ref.model
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    reference = config.actor_rollout_ref.ref
    attention = str(model.override_config.get("attn_implementation", ""))
    if not bool(model.enable_gradient_checkpointing):
        raise SystemExit("GRPO requires gradient checkpointing for the 30k contract")
    if not bool(model.get("use_liger", False)):
        raise SystemExit("GRPO acceleration config requires model.use_liger=true")
    if attention != "flash_attention_2":
        raise SystemExit(
            "GRPO acceleration config requires attn_implementation=flash_attention_2"
        )
    if bool(actor.fsdp_config.optimizer_offload):
        raise SystemExit("GRPO acceleration config requires actor optimizer offload=false")
    if not bool(actor.fsdp_config.param_offload):
        raise SystemExit("GRPO memory config requires actor parameter offload=true")
    if bool(reference.fsdp_config.param_offload):
        raise SystemExit(
            "GRPO no-KL config requires unused reference parameter offload=false"
        )
    if bool(actor.use_kl_loss) or float(actor.kl_loss_coef) != 0.0:
        raise SystemExit("GRPO actor KL loss must remain disabled")
    if bool(config.algorithm.use_kl_in_reward):
        raise SystemExit("GRPO reward KL penalty must remain disabled")
    if float(rollout.gpu_memory_utilization) != 0.50:
        raise SystemExit("GRPO rollout.gpu_memory_utilization must equal 0.50")
    if not bool(rollout.free_cache_engine):
        raise SystemExit("GRPO memory config requires rollout.free_cache_engine=true")
    print(
        "GRPO acceleration config preflight passed: "
        + json.dumps(
            {
                "attention": attention,
                "gradient_checkpointing": True,
                "liger": True,
                "actor_param_offload": True,
                "actor_optimizer_offload": False,
                "reference_policy_enabled": False,
                "reference_param_offload": False,
                "free_cache_engine": True,
                "rollout_gpu_memory_utilization": 0.50,
                "flash_linear_attention": "automatic when CUDA FLA kernels are available",
            },
            sort_keys=True,
        )
    )


def main():
    config = compose_runtime_config(sys.argv[1:])
    validate_environment_contract()
    validate_training_memory_budget(config)
    validate_acceleration_config(config)
    required_paths = {"GRPO_TRAIN_FILE": os.environ.get("GRPO_TRAIN_FILE")}
    if config.data.val_files:
        required_paths["GRPO_VAL_FILE"] = os.environ.get("GRPO_VAL_FILE")
    missing = [name for name, value in required_paths.items() if not value or not Path(value).is_file()]
    if missing:
        raise SystemExit("missing GRPO parquet file(s): " + ", ".join(missing))
    validate_training_data_contract(Path(required_paths["GRPO_TRAIN_FILE"]))
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"incompatible Python: expected 3.12, got {sys.version.split()[0]}")

    installed = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            installed[package] = version(package)
        except PackageNotFoundError as exc:
            raise SystemExit(f"missing GRPO dependency: {package}=={expected}") from exc
        if installed[package].split("+", 1)[0] != expected:
            raise SystemExit(
                f"incompatible GRPO dependency: expected {package}=={expected}, got {installed[package]}"
            )
    validate_transformers_revision()

    try:
        import torch
        import verl
        from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.tools.base_tool import BaseTool
        from verl.utils.tracking import Tracking

        from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
        from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
        from shopping_grpo.training.grpo.compat import install_torch_padding_fallback
    except ImportError as exc:
        raise SystemExit(
            "incompatible veRL 0.8 install: required AgentLoop/Tool APIs are unavailable; "
            f"original error: {exc}"
        ) from exc

    verl_source = Path(verl.__file__).resolve()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in the GRPO environment")
    if (
        not issubclass(ShoppingToolAgentLoop, ToolAgentLoop)
        or not issubclass(ShopSimulatorTool, BaseTool)
        or AgentState.TERMINATED.value != "terminated"
        or not hasattr(ToolAgentLoop, "_handle_processing_tools_state")
    ):
        raise SystemExit("incompatible veRL ToolAgentLoop lifecycle API")
    if "qwen3_coder" not in ToolParser._registry:
        raise SystemExit("veRL 0.8 built-in qwen3_coder parser is unavailable")
    if "swanlab" not in Tracking.supported_backend:
        raise SystemExit("veRL 0.8 SwanLab tracking backend is unavailable")
    validate_dynamic_sampling(config, verl_source, installed)
    validate_actor_cache_cleanup_patch(verl_source, installed)
    validate_vllm_stability_patch(verl_source, installed)
    validate_swanlab_tracking(config)
    install_torch_padding_fallback()
    print(
        "GRPO runtime preflight passed: "
        + ", ".join(f"{name}={value}" for name, value in installed.items())
        + f", source={verl_source}"
    )


if __name__ == "__main__":
    main()
