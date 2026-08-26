#!/usr/bin/env python3
"""对验收后的 Shopping tool-calling 数据进行最小 LoRA SFT。"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time as _time
import traceback
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from shopping_grpo.training.sft.dataset import load_supervised_examples

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # Qwen3.5 的大多数文本层是 Gated DeltaNet，不能遗漏其线性注意力投影。
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


def parse_args():
    parser = argparse.ArgumentParser(description="使用 Transformers + PEFT 执行 Shopping LoRA SFT")
    parser.add_argument("--model", required=True, help="Hugging Face 模型名或本地模型目录")
    parser.add_argument("--train", type=Path, required=True, help="训练 SFT JSONL")
    parser.add_argument("--validation", type=Path, default=None, help="可选验证 SFT JSONL")
    parser.add_argument("--output", type=Path, required=True, help="LoRA adapter 输出目录")
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="独立运行记录目录；默认写入 <output>/run-records",
    )
    parser.add_argument(
        "--training-contract",
        type=Path,
        default=None,
        help="首次训练 preflight 契约；会复制进 adapter 供恢复与合并校验",
    )
    # 35k 保留当前完整轨迹；正式训练前仍由显存门与 BF16/SDPA smoke test 保护。
    parser.add_argument("--max-length", type=int, default=30000)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="模型与训练精度；auto 在 CUDA 上优先 bf16，其次 fp16，CPU 使用 fp32。",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="兼容旧命令；等价于 --dtype bf16，不能与其他 --dtype 同时使用。",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="可选模型 revision；本地路径通常不需要。",
    )
    parser.add_argument("--liger-kernel", action="store_true", help="启用 Liger 融合 loss，避免全序列 logits 常驻")
    parser.add_argument(
        "--attention-implementation",
        choices=("auto", "sdpa", "flash_attention_2"),
        default="auto",
        help="注意力后端；sdpa 使用 PyTorch 原生内存高效实现，不要求编译 FlashAttention 2。",
    )
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
        default=None,
        help="可选的 Transformers 模型并行映射；用于单进程跨多张 GPU 分摊模型层。",
    )
    parser.add_argument("--qlora", action="store_true", help="以 NF4 4-bit 加载基座，并按 PEFT 标准预处理")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--recipe-variant",
        default="canonical",
        help="写入记录的 recipe 名称；canonical 以外均视为显式实验变体",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=-1, help="最大训练步数（-1=完整 epoch）；用于冒烟测试")
    parser.add_argument("--swanlab", action="store_true", help="启用 SwanLab 训练监控")
    parser.add_argument("--swanlab-project", default="shopping-grpo", help="SwanLab project 名")
    parser.add_argument("--swanlab-run-name", default=None, help="SwanLab run 名；默认自动生成")
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local"),
        default="online",
        help="SwanLab 在线同步或只保存在本地；仅 --swanlab 时生效。",
    )
    return parser.parse_args()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        stream.flush()


def _safe_environment_snapshot():
    """只记录影响离线训练与设备选择的非敏感环境变量。"""

    names = (
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "PYTHONUNBUFFERED",
        "SWANLAB_PROJECT",
        "SWANLAB_LOG_DIR",
        "SWANLAB_MODE",
        "SWANLAB_INTERACTIVE",
        "SFT_MIN_GPU_MEMORY_GIB",
        "SFT_MIN_FREE_GPU_MEMORY_GIB",
        "SFT_MIN_FREE_DISK_GIB",
    )
    return {name: os.environ.get(name) for name in names}


def _recipe_record(args):
    expected = {
        "max_length": 30000,
        "epochs": 3,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.03,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "dtype": "bf16",
        "gradient_checkpointing": False,
        "attention_implementation": "flash_attention_2",
        "device_map": None,
        "liger_kernel": True,
        "qlora": False,
        "max_steps": -1,
        "seed": 42,
        "target_modules": list(DEFAULT_TARGET_MODULES),
    }
    actual = {
        key: list(getattr(args, key)) if key == "target_modules" else getattr(args, key)
        for key in expected
    }
    deviations = {
        key: {"expected": expected[key], "actual": value}
        for key, value in actual.items()
        if value != expected[key]
    }
    if args.recipe_variant != "canonical":
        deviations["recipe_variant"] = {
            "expected": "canonical",
            "actual": args.recipe_variant,
        }
    return {
        "name": args.recipe_variant,
        "canonical": not deviations,
        "effective": actual,
        "deviations": deviations,
    }


def _runtime_snapshot(torch, args=None):
    packages = {}
    package_names = [
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "accelerate",
        "swanlab",
    ]
    if args is not None and args.liger_kernel:
        package_names.append("liger-kernel")
    if args is not None and args.attention_implementation == "flash_attention_2":
        package_names.append("flash-attn")
    if args is not None and args.qlora:
        package_names.append("bitsandbytes")
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    cuda = {
        "available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        cuda["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_properties(index).name,
                "total_memory_gib": round(
                    torch.cuda.get_device_properties(index).total_memory / 1024**3, 2
                ),
                "capability": [
                    torch.cuda.get_device_properties(index).major,
                    torch.cuda.get_device_properties(index).minor,
                ],
            }
            for index in range(torch.cuda.device_count())
        ]
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": packages,
        "cuda": cuda,
    }


def _adapter_artifacts(output):
    records = []
    for path in sorted(Path(output).iterdir()):
        if (
            not path.is_file()
            or path.name.startswith("optimizer")
            or path.name == "train_summary.json"
        ):
            continue
        record = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        records.append(record)
    return records


def _training_dependencies():
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForMultimodalLM,
            AutoProcessor,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit("缺少训练依赖。请执行：uv sync --extra sft") from exc
    return (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )


def _model_load_kwargs(args, dtype, bits_and_bytes_config):
    """构造可审计的模型加载参数；加速功能必须显式开启。"""
    kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if args.revision:
        kwargs["revision"] = args.revision
    if args.attention_implementation != "auto":
        kwargs["attn_implementation"] = args.attention_implementation
    if args.device_map:
        kwargs["device_map"] = args.device_map
    if args.qlora:
        kwargs["quantization_config"] = bits_and_bytes_config(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return kwargs


def _prepare_model_for_training(model, args, prepare_model_for_kbit_training):
    """按 PEFT 推荐顺序准备量化模型与梯度检查点。"""
    if args.qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if not args.qlora and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def _validate_optional_training_dependencies(args):
    """仅在所选实验需要时检查可选加速包，保持基础 LoRA 环境轻量。"""
    if args.qlora:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "--qlora 需要 bitsandbytes；请执行："
                "uv sync --extra sft --extra sft-accelerated"
            ) from exc
    if args.liger_kernel:
        try:
            import liger_kernel  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "--liger-kernel 需要 liger-kernel；请执行："
                "uv sync --extra sft --extra sft-accelerated"
            ) from exc
    if args.attention_implementation == "flash_attention_2":
        if platform.system() != "Linux":
            raise SystemExit(
                "--attention-implementation flash_attention_2 requires Linux/CUDA"
            )
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "Flash Attention 2 requires flash-attn; run: "
                "SFT_INSTALL_FLASH_ATTN=1 bash scripts/setup_sft.sh"
            ) from exc


def _resolve_dtype(args, torch):
    """Resolve one explicit dtype for model loading and TrainingArguments."""

    requested = args.dtype
    if args.bf16:
        if requested not in {"auto", "bf16"}:
            raise SystemExit("--bf16 cannot be combined with a non-bf16 --dtype")
        requested = "bf16"
    if requested == "auto":
        if torch.cuda.is_available():
            requested = (
                "bf16"
                if torch.cuda.is_bf16_supported()
                else "fp16"
            )
        else:
            requested = "fp32"
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return requested, mapping[requested]


def _swanlab_config(args):
    """通过环境变量配置官方 Transformers SwanLab callback。"""
    if not args.swanlab:
        return "none", None
    try:
        import swanlab  # noqa: F401 - 仅验证可选依赖存在。
    except ImportError as exc:
        raise SystemExit("缺少 SwanLab。请执行：uv sync --extra sft") from exc

    run_name = args.swanlab_run_name or (
        f"lora-r{args.lora_r}-bs{args.per_device_train_batch_size}"
        f"x{args.gradient_accumulation_steps}-lr{args.learning_rate}"
    )
    if args.swanlab_mode == "online" and not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit(
            "SwanLab online 模式要求预先设置 SWANLAB_API_KEY；"
            "租卡环境禁止交互式登录。"
        )
    os.environ["SWANLAB_PROJECT"] = args.swanlab_project
    os.environ["SWANLAB_MODE"] = args.swanlab_mode
    os.environ["SWANLAB_INTERACTIVE"] = "false"
    return "swanlab", run_name


def _loss_only_eval_trainer_class(trainer_base, enable_skip_logits):
    """构造只在 loss-only 验证时显式跳过完整词表 logits 的 Trainer。

    Qwen3.5 的 Liger forward 默认只在 ``model.training`` 时启用融合
    LM-head + cross-entropy；Trainer 验证会先调用 ``model.eval()``，即使最终
    只需要 eval_loss，也会物化 ``[batch, sequence, vocab]`` logits。20K 上下文
    和 248K 词表会因此产生约 20 GiB 的瞬时 FP32 张量。

    ``skip_logits`` 是 Liger Qwen3.5 forward 的公开参数。这里只在 Trainer 已经
    明确 ``prediction_loss_only=True`` 且输入含 labels 时传入，不改变训练前向，
    也不影响需要 predictions/metrics 的评估。
    """

    class LossOnlyEvalTrainer(trainer_base):
        def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=None,
        ):
            if enable_skip_logits and prediction_loss_only and inputs.get("labels") is not None:
                inputs = dict(inputs)
                inputs["skip_logits"] = True
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )

    return LossOnlyEvalTrainer


def _load_preprocessing_components(
    model_name,
    auto_config,
    auto_tokenizer,
    auto_processor,
    revision=None,
):
    """按模型配置选择 chat template 的持有者。

    Qwen3.5 是带视觉编码器的条件生成模型，官方模板由 processor 提供；本项目
    当前数据仅含文本和工具调用，因此 labels 仍用 processor.tokenizer 的 token id。
    其他纯文本因果模型保持原来的 tokenizer 路径。
    """
    load_kwargs = {"trust_remote_code": True}
    if revision:
        load_kwargs["revision"] = revision
    config = auto_config.from_pretrained(model_name, **load_kwargs)
    is_multimodal = str(getattr(config, "model_type", "")).startswith("qwen3_5")
    if is_multimodal:
        processor = auto_processor.from_pretrained(model_name, **load_kwargs)
        template = processor if getattr(processor, "chat_template", None) else processor.tokenizer
        return processor.tokenizer, template, True
    tokenizer = auto_tokenizer.from_pretrained(model_name, **load_kwargs)
    return tokenizer, tokenizer, False


def _torch_dataset(examples, torch):
    class TokenizedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            example = examples[index]
            return {
                "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(example["labels"], dtype=torch.long),
            }

    return TokenizedDataset()


def _collate(batch, pad_token_id, torch):
    """右侧 padding，labels 的 padding 永远不参与 loss。"""
    max_length = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    labels = torch.full((len(batch), max_length), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        length = item["input_ids"].size(0)
        input_ids[row, :length] = item["input_ids"]
        attention_mask[row, :length] = item["attention_mask"]
        labels[row, :length] = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _run_training(args, record_dir, metrics_path, started_at, start_time, phase_state):
    phase_state["phase"] = "validate_arguments"
    if args.max_length < 1 or args.epochs <= 0:
        raise SystemExit("--max-length 与 --epochs 必须为正数")
    _validate_optional_training_dependencies(args)
    phase_state["phase"] = "load_dependencies"
    (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    ) = _training_dependencies()
    set_seed(args.seed)
    report_to, run_name = _swanlab_config(args)
    if report_to == "swanlab":
        os.environ["SWANLAB_LOG_DIR"] = str((record_dir / "swanlab").resolve())
    progress_state = {"initial_global_step": 0}
    _write_json(
        record_dir / "resolved_training_config.json",
        {
            "schema_version": "shopping-sft-training-config-v1",
            "created_at": started_at,
            "arguments": vars(args),
            "recipe": _recipe_record(args),
            "data": {
                "train": {
                    "path": str(args.train.resolve()),
                    "sha256": _sha256(args.train),
                },
                "validation": {
                    "path": str(args.validation.resolve()),
                    "sha256": _sha256(args.validation),
                }
                if args.validation
                else None,
            },
            "runtime": _runtime_snapshot(torch, args),
            "environment": _safe_environment_snapshot(),
        },
    )

    # --- Progress callback: 独立持久化，避免指标只留在 stdout。 ---
    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.step_start = None
            self.epoch_start = None
            self.last_log_time = _time.time()
            self.last_log_step = 0

        def _event(self, event, state, payload=None):
            if not state.is_world_process_zero:
                return
            gpu = {}
            if torch.cuda.is_available():
                gpu = {
                    "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
                    "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
                    "peak_allocated_gib": round(
                        torch.cuda.max_memory_allocated() / 1024**3, 3
                    ),
                    "peak_reserved_gib": round(
                        torch.cuda.max_memory_reserved() / 1024**3, 3
                    ),
                }
            _append_jsonl(
                metrics_path,
                {
                    "timestamp": _utc_now(),
                    "event": event,
                    "global_step": state.global_step,
                    "max_steps": state.max_steps,
                    "epoch": state.epoch,
                    "gpu": gpu,
                    **(payload or {}),
                },
            )

        def on_train_begin(self, args, state, control, **kwargs):
            self.last_log_time = _time.time()
            self.last_log_step = state.global_step
            progress_state["initial_global_step"] = state.global_step
            self._event("train_begin", state)
            return control

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_start = _time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or not logs:
                return control
            now = _time.time()
            step_delta = max(1, state.global_step - self.last_log_step)
            seconds_per_step = (now - self.last_log_time) / step_delta
            self.last_log_time = now
            self.last_log_step = state.global_step
            eta_seconds = max(0, state.max_steps - state.global_step) * seconds_per_step
            self._event(
                "log",
                state,
                {
                    "metrics": dict(logs),
                    "seconds_per_optimizer_step": round(seconds_per_step, 3),
                    "eta_seconds": round(eta_seconds, 1),
                },
            )
            if "loss" in logs:
                gpu_mem = (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if torch.cuda.is_available()
                    else 0
                )
                print(
                    f"[step {state.global_step}/{state.max_steps}] "
                    f"loss={float(logs['loss']):.4f} step_t={seconds_per_step:.1f}s "
                    f"GPU_peak={gpu_mem:.1f}GiB ETA={eta_seconds/60:.0f}min"
                )
            elif "eval_loss" in logs:
                print(
                    f"[eval step {state.global_step}] "
                    f"loss={float(logs['eval_loss']):.4f} epoch={state.epoch}"
                )
            return control

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.epoch_start = _time.time()
            self._event("epoch_begin", state)
            print(f"\n{'='*60}\n  EPOCH {int(state.epoch)} 开始  steps={state.max_steps}\n{'='*60}")

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch_time = _time.time() - self.epoch_start if self.epoch_start else 0
            self._event("epoch_end", state, {"epoch_time_seconds": round(epoch_time, 1)})
            print(f"  EPOCH {int(state.epoch)} 完成  耗时={epoch_time/60:.1f}min")

        def on_save(self, args, state, control, **kwargs):
            self._event("checkpoint_saved", state)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self._event("train_end", state)
            return control

    phase_state["phase"] = "load_processor"
    tokenizer, chat_template, is_multimodal = _load_preprocessing_components(
        args.model,
        auto_config=AutoConfig,
        auto_tokenizer=AutoTokenizer,
        auto_processor=AutoProcessor,
        revision=args.revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Phase 1: 加载训练数据 ----
    print(f"\n{'='*60}")
    print(f"  Phase 1/3: 加载 & Tokenize 训练数据 (max_length={args.max_length})")
    print(f"{'='*60}")
    phase_state["phase"] = "tokenize_train"
    train_examples, train_stats = load_supervised_examples(
        args.train,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=args.max_length,
    )
    print("train_data=", train_stats)
    if not train_examples:
        raise SystemExit("训练集没有可用样本；请检查 data/sft/ 中的 JSONL 格式")
    validation_examples = []
    validation_stats = {"total": 0, "kept": 0, "dropped": 0}
    if args.validation:
        phase_state["phase"] = "tokenize_validation"
        validation_examples, validation_stats = load_supervised_examples(
            args.validation,
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=args.max_length,
        )
        print("validation_data=", validation_stats)
        if not validation_examples:
            raise SystemExit("验证集没有可用样本；请调整划分或 --max-length")
    _write_json(
        record_dir / "tokenization.json",
        {
            "schema_version": "shopping-sft-tokenization-v1",
            "created_at": _utc_now(),
            "max_length": args.max_length,
            "train": train_stats,
            "validation": validation_stats,
        },
    )
    _append_jsonl(
        metrics_path,
        {
            "timestamp": _utc_now(),
            "event": "tokenization_completed",
            "train": train_stats,
            "validation": validation_stats,
        },
    )

    dtype_name, dtype = _resolve_dtype(args, torch)
    model_class = AutoModelForMultimodalLM if is_multimodal else AutoModelForCausalLM

    # ---- Phase 2: 加载模型 + LoRA ----
    print(f"\n{'='*60}")
    print("  Phase 2/3: 加载模型与 LoRA")
    print(f"{'='*60}")
    print(f"  model={args.model}")
    print(f"  revision={args.revision or 'default/local'}")
    print(f"  dtype={dtype_name}")
    print(f"  attention_implementation={args.attention_implementation}")
    print(f"  device_map={args.device_map or 'single_device'}")
    print(f"  qlora={args.qlora}")
    print(f"  lora_r={args.lora_r} lora_alpha={args.lora_alpha}")
    print(f"  lora_targets={','.join(args.target_modules)}")
    phase_state["phase"] = "load_base_model"
    model = model_class.from_pretrained(
        args.model,
        **_model_load_kwargs(
            args,
            dtype=dtype,
            bits_and_bytes_config=BitsAndBytesConfig,
        ),
    )
    model = _prepare_model_for_training(
        model,
        args,
        prepare_model_for_kbit_training=prepare_model_for_kbit_training,
    )
    phase_state["phase"] = "initialize_lora"
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=list(args.target_modules),
        ),
    )
    model.print_trainable_parameters()

    if report_to == "swanlab":
        print(
            f"[SwanLab] project={args.swanlab_project} run={run_name} "
            f"mode={os.environ['SWANLAB_MODE']}"
        )
    phase_state["phase"] = "build_trainer"
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=dtype_name == "bf16",
        fp16=dtype_name == "fp16",
        gradient_checkpointing=args.gradient_checkpointing,
        use_liger_kernel=args.liger_kernel,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch" if validation_examples else "no",
        report_to=report_to,
        run_name=run_name,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    _write_json(
        record_dir / "effective_training_arguments.json",
        {
            "schema_version": "shopping-sft-training-arguments-v1",
            "created_at": _utc_now(),
            "arguments": training_args.to_dict(),
            "recipe": _recipe_record(args),
            "environment": _safe_environment_snapshot(),
        },
    )
    trainer_class = _loss_only_eval_trainer_class(
        Trainer,
        enable_skip_logits=args.liger_kernel and is_multimodal,
    )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=_torch_dataset(train_examples, torch),
        eval_dataset=_torch_dataset(validation_examples, torch) if validation_examples else None,
        data_collator=partial(_collate, pad_token_id=tokenizer.pad_token_id, torch=torch),
        callbacks=[ProgressCallback()],
    )
    phase_state["phase"] = "training"
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    phase_state["phase"] = "save_adapter"
    trainer.save_model(str(args.output))
    chat_template.save_pretrained(str(args.output))
    trainer.save_state()
    if args.training_contract:
        shutil.copy2(args.training_contract, args.output / "training_contract.json")

    # --- 训练完成摘要 ---
    total_time = _time.time() - start_time
    gpu_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    log_history = list(trainer.state.log_history)
    eval_history = [
        {
            "step": entry.get("step"),
            "epoch": entry.get("epoch"),
            "eval_loss": entry.get("eval_loss"),
            "eval_runtime": entry.get("eval_runtime"),
            "eval_samples_per_second": entry.get("eval_samples_per_second"),
        }
        for entry in log_history
        if "eval_loss" in entry
    ]
    loss_history = [entry for entry in log_history if "loss" in entry and "step" in entry]
    weighted_loss_sum = 0.0
    weighted_loss_steps = 0
    previous_loss_step = 0
    for entry in loss_history:
        current_step = int(entry["step"])
        step_count = max(0, current_step - previous_loss_step)
        weighted_loss_sum += float(entry["loss"]) * step_count
        weighted_loss_steps += step_count
        previous_loss_step = current_step
    logged_step_weighted_loss = (
        weighted_loss_sum / weighted_loss_steps if weighted_loss_steps else None
    )
    initial_global_step = int(progress_state["initial_global_step"])
    attempt_optimizer_steps = trainer.state.global_step - initial_global_step
    effective_train_loss = (
        logged_step_weighted_loss
        if initial_global_step and logged_step_weighted_loss is not None
        else result.training_loss
    )
    trainable_parameters, total_parameters = model.get_nb_trainable_parameters()

    train_summary = {
        "schema_version": "shopping-sft-train-summary-v1",
        "status": "completed",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "tokenization": {
            "train": train_stats,
            "validation": validation_stats,
        },
        "train_loss": effective_train_loss,
        "train_loss_source": (
            "trainer_train_output"
            if initial_global_step == 0
            else (
                "step_weighted_trainer_log_history"
                if logged_step_weighted_loss is not None
                else "trainer_train_output_resume_scope"
            )
        ),
        "trainer_reported_train_loss": result.training_loss,
        "logged_step_weighted_train_loss": logged_step_weighted_loss,
        "metrics": result.metrics,
        "initial_global_step": initial_global_step,
        "attempt_optimizer_steps": attempt_optimizer_steps,
        "optimizer_steps": trainer.state.global_step,
        "eval_history": eval_history,
        "last_eval": eval_history[-1] if eval_history else None,
        "peak_gpu_memory_gib": round(gpu_peak, 2),
        "total_time_minutes": round(total_time / 60, 1) if total_time else None,
        "parameters": {
            "trainable": trainable_parameters,
            "total": total_parameters,
            "trainable_percent": round(100 * trainable_parameters / total_parameters, 6),
        },
        "monitoring": {
            "backend": report_to,
            "project": args.swanlab_project if args.swanlab else None,
            "run_name": run_name,
            "mode": args.swanlab_mode if args.swanlab else None,
            "effective_mode": os.environ.get("SWANLAB_MODE") if args.swanlab else None,
        },
        "recipe": _recipe_record(args),
        "acceleration": {
            "dtype": dtype_name,
            "liger_kernel": args.liger_kernel,
            "attention_implementation": args.attention_implementation,
            "device_map": args.device_map,
            "qlora": args.qlora,
        },
        "data": {
            "train": {"path": str(args.train.resolve()), "sha256": _sha256(args.train)},
            "validation": {
                "path": str(args.validation.resolve()),
                "sha256": _sha256(args.validation),
            }
            if args.validation
            else None,
        },
        "runtime": _runtime_snapshot(torch, args),
        "training_contract": {
            "path": str((args.output / "training_contract.json").resolve()),
            "sha256": _sha256(args.output / "training_contract.json"),
        }
        if args.training_contract
        else None,
        "adapter_artifacts": _adapter_artifacts(args.output),
        "arguments": vars(args),
    }

    print(f"\n{'='*60}")
    print("  训练完成")
    displayed_train_loss = train_summary["train_loss"]
    print(
        f"  train_loss={displayed_train_loss:.4f} "
        f"source={train_summary['train_loss_source']}"
    )
    if initial_global_step:
        print(
            f"  resumed_from_step={initial_global_step} "
            f"attempt_steps={attempt_optimizer_steps} "
            f"trainer_reported_loss={result.training_loss:.4f}"
        )
    print(f"  eval_loss={eval_history[-1]['eval_loss'] if eval_history else 'N/A'}")
    print(f"  peak_gpu={gpu_peak:.1f} GiB")
    print(f"  adapter → {args.output}")
    print(f"{'='*60}\n")

    _write_json(record_dir / "trainer_log_history.json", log_history)
    _write_json(record_dir / "train_summary.json", train_summary)
    _write_json(args.output / "train_summary.json", train_summary)
    phase_state["phase"] = "completed"

    print(f"LoRA adapter 已保存到 {args.output}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    start_time = _time.time()
    started_at = _utc_now()
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record_dir = args.record_dir or (args.output / "run-records")
    record_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = record_dir / "metrics.jsonl"
    phase_state = {"phase": "attempt_begin"}
    _append_jsonl(
        metrics_path,
        {
            "timestamp": started_at,
            "event": "attempt_begin",
            "arguments": vars(args),
            "recipe": _recipe_record(args),
            "environment": _safe_environment_snapshot(),
        },
    )
    try:
        _run_training(
            args=args,
            record_dir=record_dir,
            metrics_path=metrics_path,
            started_at=started_at,
            start_time=start_time,
            phase_state=phase_state,
        )
    except BaseException as exc:
        failure = {
            "schema_version": "shopping-sft-failure-v1",
            "status": "failed",
            "started_at": started_at,
            "failed_at": _utc_now(),
            "phase": phase_state["phase"],
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "arguments": vars(args),
            "recipe": _recipe_record(args),
            "environment": _safe_environment_snapshot(),
        }
        _write_json(record_dir / "failure.json", failure)
        _append_jsonl(
            metrics_path,
            {
                "timestamp": failure["failed_at"],
                "event": "attempt_failed",
                "phase": failure["phase"],
                "exception_type": failure["exception_type"],
                "message": failure["message"],
            },
        )
        raise
    _append_jsonl(
        metrics_path,
        {
            "timestamp": _utc_now(),
            "event": "attempt_completed",
            "phase": phase_state["phase"],
        },
    )


if __name__ == "__main__":
    main()
