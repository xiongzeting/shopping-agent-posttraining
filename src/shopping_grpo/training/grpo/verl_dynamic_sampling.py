"""Pinned veRL 0.8 rollout integration for shopping dynamic sampling."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.reward import extract_reward

from shopping_grpo.training.grpo.dynamic_sampling import (
    aggregate_shopping_metrics,
    extract_shopping_group_signals,
    select_reward_varying_groups,
)


def _select_attempt(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    reward_extra_infos: dict[str, list[Any]],
    *,
    tolerance: float,
    require_purchase_success: bool,
) -> tuple[DataProto | None, torch.Tensor | None, dict[str, list[Any]], list[int], dict[str, Any]]:
    shopping_infos = reward_extra_infos.get("shopping")
    if shopping_infos is None:
        raise RuntimeError("shopping dynamic sampling requires reward extra field 'shopping'")
    seq_rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()
    terminal_utilities, purchase_success, sampling_invalid, invalid_reasons = (
        extract_shopping_group_signals(shopping_infos)
    )
    selected_indices, stats = select_reward_varying_groups(
        batch.non_tensor_batch["uid"].tolist(),
        seq_rewards,
        terminal_utilities=terminal_utilities,
        purchase_success=purchase_success,
        sampling_invalid=sampling_invalid,
        sampling_invalid_reasons=invalid_reasons,
        tolerance=tolerance,
        require_purchase_success=require_purchase_success,
    )
    # Only ordinary low-variation/no-success groups may be retried.
    # assistant_final remains a valid negative model-error trajectory; genuinely
    # unverifiable trajectories are filtered and never regenerated.
    dropped_prompt_indices = [
        index
        for index, group in enumerate(stats["groups"])
        if not group["kept"] and group.get("retryable", True)
    ]
    if not selected_indices:
        return None, None, {}, dropped_prompt_indices, stats
    selected_extras = {
        key: [values[index] for index in selected_indices]
        for key, values in reward_extra_infos.items()
    }
    return (
        batch.select_idxs(selected_indices),
        reward_tensor[selected_indices],
        selected_extras,
        dropped_prompt_indices,
        stats,
    )


def _concat_reward_extras(parts: list[dict[str, list[Any]]]) -> dict[str, list[Any]]:
    if not parts:
        return {}
    expected_keys = set(parts[0])
    if any(set(part) != expected_keys for part in parts[1:]):
        raise RuntimeError("reward extra fields changed between dynamic-sampling attempts")
    return {
        key: [item for part in parts for item in part[key]]
        for key in sorted(expected_keys)
    }


def finalize_dynamic_sampling(
    trainer: Any,
    prompt_batch: DataProto,
    gen_batch: DataProto,
    batch: DataProto,
    reward_tensor: torch.Tensor,
    reward_extra_infos: dict[str, list[Any]],
    timing_raw: dict[str, float],
    *,
    curr_step_profile: bool,
) -> tuple[DataProto | None, torch.Tensor | None, dict[str, list[Any]], dict[str, float]]:
    """Filter the first rollout and retry only dropped prompt groups."""

    config = trainer.config.get("shopping_dynamic_sampling", {})
    if not bool(config.get("enable", False)):
        return batch, reward_tensor, reward_extra_infos, {}

    max_attempts = int(config.max_num_gen_batches)
    tolerance = float(config.reward_tolerance)
    require_purchase_success = bool(config.get("require_purchase_success", False))
    rollout_n = int(trainer.config.actor_rollout_ref.rollout.n)
    kept_batches: list[DataProto] = []
    kept_rewards: list[torch.Tensor] = []
    kept_extras: list[dict[str, list[Any]]] = []
    aggregate_counts = {
        "generated_group_count": 0,
        "kept_group_count": 0,
        "dropped_group_count": 0,
        "constant_reward_group_count": 0,
        "low_reward_variation_group_count": 0,
        "sampling_invalid_group_count": 0,
        "no_purchase_success_group_count": 0,
    }

    # The trainer batch is fixed at two prompts by veRL, but retries are
    # deliberately performed one prompt-group at a time.  A retry therefore
    # always creates exactly rollout.n (4) trajectories, never 2*rollout.n.
    attempt_batch = batch
    attempt_reward = reward_tensor
    attempt_extras = reward_extra_infos
    attempts = 0
    try:
        # Evaluate the initial 2-prompt batch once.
        while attempts < 1:
            attempts += 1
            selected_batch, selected_reward, selected_extras, dropped_indices, stats = (
                _select_attempt(
                    attempt_batch,
                    attempt_reward,
                    attempt_extras,
                    tolerance=tolerance,
                    require_purchase_success=require_purchase_success,
                )
            )
            aggregate_counts["generated_group_count"] += int(stats["num_groups"])
            aggregate_counts["kept_group_count"] += int(stats["kept_group_count"])
            aggregate_counts["dropped_group_count"] += int(stats["dropped_group_count"])
            aggregate_counts["constant_reward_group_count"] += int(
                stats["all_equal_group_count"]
            )
            aggregate_counts["low_reward_variation_group_count"] += int(
                stats["low_reward_variation_group_count"]
            )
            aggregate_counts["sampling_invalid_group_count"] += int(
                stats["sampling_invalid_group_count"]
            )
            aggregate_counts["no_purchase_success_group_count"] += int(
                stats["no_purchase_success_group_count"]
            )
            if selected_batch is not None:
                kept_batches.append(selected_batch)
                kept_rewards.append(selected_reward)
                kept_extras.append(selected_extras)
            break

        # Retry each dropped prompt independently.  Forbidden terminals are
        # excluded by dropped_indices and are never regenerated.
        for prompt_index in dropped_indices:
            retry_prompt = prompt_batch.select_idxs([prompt_index])
            retry_gen = gen_batch.select_idxs([prompt_index])
            for retry_round in range(1, max_attempts):
                attempts += 1
                retry_uid = np.array([str(uuid.uuid4())], dtype=object)
                retry_prompt.non_tensor_batch["uid"] = retry_uid.copy()
                retry_gen.non_tensor_batch["uid"] = retry_uid.copy()
                retry_gen.meta_info["global_steps"] = trainer.global_steps
                retry_request = retry_gen.repeat(repeat_times=rollout_n, interleave=True)
                if curr_step_profile:
                    trainer.llm_server_manager.start_profile()
                retry_output = trainer.async_rollout_manager.generate_sequences(retry_request)
                if curr_step_profile:
                    trainer.llm_server_manager.stop_profile()
                timing_raw.update(retry_output.meta_info["timing"])
                retry_output.meta_info.pop("timing", None)
                retry_batch = retry_prompt.repeat(repeat_times=rollout_n, interleave=True).union(retry_output)
                if "response_mask" not in retry_batch.batch:
                    responses = retry_batch.batch["responses"]
                    retry_batch.batch["response_mask"] = retry_batch.batch["attention_mask"][:, -responses.size(1):]
                if trainer.use_rm and "rm_scores" not in retry_batch.batch:
                    retry_batch = retry_batch.union(trainer._compute_reward_colocate(retry_batch))
                retry_reward, retry_extras = extract_reward(retry_batch)
                selected_batch, selected_reward, selected_extras, _, retry_stats = _select_attempt(
                    retry_batch, retry_reward, retry_extras,
                    tolerance=tolerance, require_purchase_success=require_purchase_success,
                )
                aggregate_counts["generated_group_count"] += int(retry_stats["num_groups"])
                aggregate_counts["kept_group_count"] += int(retry_stats["kept_group_count"])
                aggregate_counts["dropped_group_count"] += int(retry_stats["dropped_group_count"])
                aggregate_counts["constant_reward_group_count"] += int(retry_stats["all_equal_group_count"])
                aggregate_counts["low_reward_variation_group_count"] += int(retry_stats["low_reward_variation_group_count"])
                aggregate_counts["sampling_invalid_group_count"] += int(retry_stats["sampling_invalid_group_count"])
                aggregate_counts["no_purchase_success_group_count"] += int(retry_stats["no_purchase_success_group_count"])
                if selected_batch is not None:
                    kept_batches.append(selected_batch)
                    kept_rewards.append(selected_reward)
                    kept_extras.append(selected_extras)
                    break
    finally:
        # The normal actor-update path synchronizes weights and wakes rollout
        # replicas again. An empty dynamic-sampling result skips that entire
        # path, so sleeping here would leave vLLM asleep for the next batch and
        # every new request would disappear as "Request not found".
        if kept_batches:
            trainer.checkpoint_manager.sleep_replicas()

    metrics = {
        "sampling/generated_batches": float(attempts),
        "sampling/generated_groups": float(aggregate_counts["generated_group_count"]),
        "sampling/kept_groups": float(aggregate_counts["kept_group_count"]),
        "sampling/dropped_groups": float(aggregate_counts["dropped_group_count"]),
        "sampling/constant_reward_groups": float(
            aggregate_counts["constant_reward_group_count"]
        ),
        "sampling/low_reward_variation_groups": float(
            aggregate_counts["low_reward_variation_group_count"]
        ),
        "sampling/invalid_groups": float(
            aggregate_counts["sampling_invalid_group_count"]
        ),
        "sampling/no_purchase_success_groups": float(
            aggregate_counts["no_purchase_success_group_count"]
        ),
    }
    # Preserve terminal diagnostics even when every group is dropped.  Without
    # this, a 0/4 batch returns before aggregate_shopping_metrics() and the
    # log cannot distinguish real model failure from reward parsing failure.
    attempt_shopping = attempt_extras.get("shopping")
    if attempt_shopping is not None and len(attempt_shopping) > 0:
        metrics.update(aggregate_shopping_metrics(attempt_shopping))
    if not kept_batches:
        return None, None, {}, metrics

    final_batch = kept_batches[0] if len(kept_batches) == 1 else DataProto.concat(kept_batches)
    final_reward = kept_rewards[0] if len(kept_rewards) == 1 else torch.cat(kept_rewards)
    final_extras = _concat_reward_extras(kept_extras)
    metrics.update(aggregate_shopping_metrics(final_extras["shopping"]))
    return final_batch, final_reward, final_extras, metrics
