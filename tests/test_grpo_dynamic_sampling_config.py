"""CPU-only checks for the project dynamic-sampling configuration gate."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_grpo_runtime import (
    ACTOR_CACHE_PATCH_MARKER,
    PATCH_MARKER,
    VLLM_STABILITY_PATCH_MARKER,
    compose_runtime_config,
    validate_acceleration_config,
    validate_actor_cache_cleanup_patch,
    validate_dynamic_sampling,
    validate_training_memory_budget,
    validate_vllm_stability_patch,
)


@unittest.skipUnless(
    importlib.util.find_spec("verl") is not None,
    "requires verl==0.8.0 configuration package",
)
class DynamicSamplingConfigTest(unittest.TestCase):
    def test_training_memory_budget_enforces_30k_context_and_split_dynamic_budgets(self):
        config = compose_runtime_config([])
        validate_training_memory_budget(config)
        validate_acceleration_config(config)
        self.assertEqual(config.data.train_batch_size, 2)
        self.assertEqual(config.data.max_response_length, 25904)
        self.assertEqual(config.actor_rollout_ref.rollout.n, 4)
        self.assertEqual(config.actor_rollout_ref.rollout.max_num_seqs, 8)
        self.assertEqual(config.actor_rollout_ref.rollout.agent.num_workers, 8)
        self.assertEqual(config.actor_rollout_ref.actor.ppo_mini_batch_size, 2)
        self.assertEqual(config.actor_rollout_ref.rollout.max_model_len, 30000)
        self.assertEqual(config.actor_rollout_ref.rollout.max_num_batched_tokens, 60000)
        self.assertEqual(config.actor_rollout_ref.rollout.gpu_memory_utilization, 0.50)
        self.assertTrue(config.actor_rollout_ref.actor.use_dynamic_bsz)
        self.assertIsNone(config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu)
        self.assertTrue(
            config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
        )
        self.assertIsNone(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
        )
        self.assertTrue(config.actor_rollout_ref.ref.log_prob_use_dynamic_bsz)
        self.assertIsNone(config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu)
        self.assertEqual(config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu, 30000)
        self.assertEqual(
            config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu, 36000
        )
        self.assertEqual(config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu, 36000)
        self.assertTrue(config.actor_rollout_ref.model.enable_gradient_checkpointing)
        self.assertTrue(config.actor_rollout_ref.model.use_liger)
        self.assertTrue(config.actor_rollout_ref.actor.fsdp_config.param_offload)
        self.assertFalse(config.actor_rollout_ref.actor.fsdp_config.optimizer_offload)
        self.assertFalse(config.actor_rollout_ref.ref.fsdp_config.param_offload)
        self.assertFalse(config.actor_rollout_ref.actor.use_kl_loss)
        self.assertFalse(config.algorithm.use_kl_in_reward)
        self.assertEqual(
            config.actor_rollout_ref.model.override_config.attn_implementation,
            "flash_attention_2",
        )
        self.assertEqual(config.trainer.total_epochs, 1)
        self.assertEqual(config.trainer.total_training_steps, 100)
        self.assertEqual(config.trainer.save_freq, 5)
        self.assertEqual(config.trainer.test_freq, -1)
        self.assertFalse(config.trainer.val_before_train)

    def test_training_memory_budget_rejects_unsafe_overrides(self):
        unsafe_response = compose_runtime_config(["data.max_response_length=25000"])
        with self.assertRaisesRegex(SystemExit, "GRPO response budget"):
            validate_training_memory_budget(unsafe_response)

        wrong_prompt_batch = compose_runtime_config(["data.train_batch_size=4"])
        with self.assertRaisesRegex(SystemExit, "train_batch_size must equal 2"):
            validate_training_memory_budget(wrong_prompt_batch)

        wrong_token_budget = compose_runtime_config(
            ["actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40000"]
        )
        with self.assertRaisesRegex(SystemExit, "actor dynamic token budget 30000"):
            validate_training_memory_budget(wrong_token_budget)

        enabled_validation = compose_runtime_config(["trainer.test_freq=25"])
        with self.assertRaisesRegex(SystemExit, "test_freq must be -1 .* or 10"):
            validate_training_memory_budget(enabled_validation)

        wrong_rollout_group = compose_runtime_config(["actor_rollout_ref.rollout.n=8"])
        with self.assertRaisesRegex(SystemExit, "rollout.n must equal 4"):
            validate_training_memory_budget(wrong_rollout_group)

        fixed_actor = compose_runtime_config(
            ["actor_rollout_ref.actor.use_dynamic_bsz=false"]
        )
        with self.assertRaisesRegex(SystemExit, "actor.use_dynamic_bsz must be true"):
            validate_training_memory_budget(fixed_actor)

        fixed_rollout_log_prob = compose_runtime_config(
            ["actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false"]
        )
        with self.assertRaisesRegex(
            SystemExit, "rollout.log_prob_use_dynamic_bsz must be true"
        ):
            validate_training_memory_budget(fixed_rollout_log_prob)

    def test_acceleration_contract_keeps_optimizer_and_reference_offload_disabled(self):
        optimizer_offload = compose_runtime_config(
            ["actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"]
        )
        with self.assertRaisesRegex(SystemExit, "optimizer offload=false"):
            validate_acceleration_config(optimizer_offload)

        actor_param_resident = compose_runtime_config(
            ["actor_rollout_ref.actor.fsdp_config.param_offload=false"]
        )
        with self.assertRaisesRegex(SystemExit, "actor parameter offload=true"):
            validate_acceleration_config(actor_param_resident)

        reference_offload = compose_runtime_config(
            ["actor_rollout_ref.ref.fsdp_config.param_offload=true"]
        )
        with self.assertRaisesRegex(SystemExit, "reference parameter offload=false"):
            validate_acceleration_config(reference_offload)

    def test_hydra_overrides_resolve_project_top_level_config(self):
        config = compose_runtime_config(
            [
                "shopping_dynamic_sampling.enable=true",
                "shopping_dynamic_sampling.metric=seq_reward",
                "shopping_dynamic_sampling.max_num_gen_batches=2",
                "shopping_dynamic_sampling.reward_tolerance=0.025",
            ]
        )
        self.assertTrue(config.shopping_dynamic_sampling.enable)
        self.assertEqual(config.shopping_dynamic_sampling.metric, "seq_reward")
        self.assertEqual(config.shopping_dynamic_sampling.max_num_gen_batches, 2)
        self.assertEqual(config.shopping_dynamic_sampling.reward_tolerance, 0.025)
        self.assertTrue(config.algorithm.rollout_correction.bypass_mode)
        self.assertTrue(config.actor_rollout_ref.rollout.calculate_log_probs)

    def test_enabled_config_requires_installed_patch_marker(self):
        config = compose_runtime_config(["shopping_dynamic_sampling.enable=true"])
        with tempfile.TemporaryDirectory() as temp_dir:
            verl_source = Path(temp_dir) / "verl" / "__init__.py"
            trainer_source = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
            trainer_source.parent.mkdir(parents=True)
            verl_source.write_text("", encoding="utf-8")
            trainer_source.write_text("# unpatched\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "patch marker is missing"):
                validate_dynamic_sampling(config, verl_source, {"verl": "0.8.0"})

            trainer_source.write_text(f"# {PATCH_MARKER}\n", encoding="utf-8")
            validate_dynamic_sampling(config, verl_source, {"verl": "0.8.0"})

            three_attempts = compose_runtime_config(
                ["shopping_dynamic_sampling.max_num_gen_batches=3"]
            )
            with self.assertRaisesRegex(SystemExit, "must equal 2"):
                validate_dynamic_sampling(
                    three_attempts,
                    verl_source,
                    {"verl": "0.8.0"},
                )

    def test_actor_cache_cleanup_requires_installed_patch_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            verl_source = Path(temp_dir) / "verl" / "__init__.py"
            worker_source = verl_source.parent / "workers" / "engine_workers.py"
            worker_source.parent.mkdir(parents=True)
            verl_source.write_text("", encoding="utf-8")
            worker_source.write_text("# unpatched\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "cache cleanup patch is missing"):
                validate_actor_cache_cleanup_patch(
                    verl_source,
                    {"verl": "0.8.0"},
                )

            worker_source.write_text(
                f"# {ACTOR_CACHE_PATCH_MARKER}\n",
                encoding="utf-8",
            )
            validate_actor_cache_cleanup_patch(
                verl_source,
                {"verl": "0.8.0"},
            )

    def test_vllm_stability_requires_patch_marker_and_timeout_environment(self):
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
                    "SHOPPING_GRPO_VLLM_GENERATION_TIMEOUT_SECONDS": "10000",
                },
            ):
                validate_vllm_stability_patch(verl_source, {"verl": "0.8.0"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
