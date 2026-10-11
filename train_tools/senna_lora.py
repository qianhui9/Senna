"""Single-GPU, 16-bit LoRA planning fine-tuning; no DeepSpeed or remote downloads."""

import argparse
from collections import Counter
import fcntl
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model

from data_tools.senna_inputs import CAMERAS, check_model, validate_sample
from data_tools.senna_qa_utils import load_images
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.senna.senna_llava_llama import SennaLlavaLlamaForCausalLM
from train_tools.lora_io import MANIFEST, base_identity, digest, latest_complete_checkpoint, load_visual_weights, mark_complete, save_visual_weights

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", required=True, help="Held-out input JSON, used to prevent leakage")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 instead of BF16")
    parser.add_argument("--freeze-visual-adapters", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="CPU-only full input/tokenization checks")
    return parser.parse_args()


def encode_planning(sample, tokenizer, max_length):
    """Use exactly the inference prompt, supervising only the assistant answer/EOS."""
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], sample["conversations"][0]["value"])
    conv.append_message(conv.roles[1], None)
    prefix = tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors="pt")
    conv.messages[-1][1] = sample["conversations"][1]["value"].strip()
    ids = tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors="pt")
    if not torch.equal(ids[:len(prefix)], prefix):
        raise ValueError(f"Assistant token boundary mismatch: {sample['token']}")
    if (ids == IMAGE_TOKEN_INDEX).sum().item() != 6:
        raise ValueError("Six image placeholders are required")
    expanded_length = len(ids) + 6 * (128 - 1)
    if expanded_length > max_length:
        raise ValueError(f"{sample['token']}: expanded length {expanded_length} > {max_length}; increase --max-length")
    labels = ids.clone()
    labels[:len(prefix)] = IGNORE_INDEX
    if not (labels != IGNORE_INDEX).any() or ids[-1].item() != tokenizer.eos_token_id:
        raise ValueError("Assistant answer/EOS is missing; refusing an all-masked loss")
    return ids, labels, expanded_length


def load_records(train_path, eval_path, max_samples=None):
    records = json.loads(Path(train_path).read_text())
    held_out = json.loads(Path(eval_path).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Training data must be a nonempty planning JSON list")
    tokens = [row["token"] for row in records]
    if len(set(tokens)) != len(tokens):
        raise ValueError("Duplicate training sample tokens")
    if set(tokens) & {row["token"] for row in held_out}:
        raise ValueError("Training/validation sample leakage")
    if {row["scene_token"] for row in records} & {row["scene_token"] for row in held_out}:
        raise ValueError("Training/validation scene leakage")
    for row in tqdm(records, desc="Training input checks"):
        validate_sample(row)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        records = records[:max_samples]
    return records


class PlanningDataset(Dataset):
    def __init__(self, records, tokenizer, processor, model_config, max_length):
        self.records, self.processor, self.model_config = records, processor, model_config
        self.encoded = [encode_planning(row, tokenizer, max_length)
                        for row in tqdm(records, desc="Checking assistant labels and expanded lengths")]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        ids, labels, _ = self.encoded[index]
        images = process_images(load_images(self.records[index]["images"]), self.processor, self.model_config)
        return {"input_ids": ids, "labels": labels, "images": images}


class PlanningCollator:
    def __init__(self, pad_token):
        self.pad_token = pad_token

    def __call__(self, records):
        ids = torch.nn.utils.rnn.pad_sequence([r["input_ids"] for r in records], batch_first=True, padding_value=self.pad_token)
        labels = torch.nn.utils.rnn.pad_sequence([r["labels"] for r in records], batch_first=True, padding_value=IGNORE_INDEX)
        return {"input_ids": ids, "labels": labels, "attention_mask": ids.ne(self.pad_token),
                "images": torch.stack([r["images"] for r in records])}


class SennaLoraTrainer(Trainer):
    def __init__(self, *args, manifest=None, **kwargs):
        self.senna_manifest = manifest
        super().__init__(*args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        directory = Path(output_dir or self.args.output_dir)
        super()._save(str(directory), state_dict=state_dict)
        save_visual_weights(self.model, directory)
        self.model.get_base_model().config.save_pretrained(directory)
        (directory / MANIFEST).write_text(json.dumps(self.senna_manifest, indent=2))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        load_visual_weights(model or self.model, resume_from_checkpoint)

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        if self.args.should_save:
            mark_complete(Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}")


class FiniteLoss(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and not math.isfinite(logs["loss"]):
            raise RuntimeError("Nonfinite loss; aborting training")


def manifest_for(args, count):
    settings = {k: v for k, v in vars(args).items()
                if k not in ("resume", "validate_only", "output_dir", "train_data", "eval_data", "model_path")}
    return {"schema_version": 1, "base_model": base_identity(args.model_path),
            "train_sha256": digest(args.train_data), "eval_sha256": digest(args.eval_data),
            "train_data": str(Path(args.train_data).resolve()), "eval_data": str(Path(args.eval_data).resolve()),
            "selected_samples": count, "settings": settings, "conv_mode": "llava_v1",
            "camera_order": list(CAMERAS), "targets": list(TARGETS),
            "implementation_sha256": {name: digest(Path(__file__).parent / name)
                                      for name in ("senna_lora.py", "lora_io.py")}}


def train(args):
    for key in ("batch_size", "gradient_accumulation", "lora_r", "lora_alpha", "max_length", "save_steps"):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.epochs <= 0 or args.learning_rate <= 0 or args.workers < 0 or args.max_steps == 0:
        raise ValueError("Invalid epoch/learning rate/worker/max-step setting")
    check_model(args.model_path)
    records = load_records(args.train_data, args.eval_data, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, local_files_only=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    config = json.loads((Path(args.model_path) / "config.json").read_text())
    processor = CLIPImageProcessor.from_pretrained(config["mm_vision_tower"], local_files_only=True)
    from types import SimpleNamespace
    dataset = PlanningDataset(records, tokenizer, processor, SimpleNamespace(**config), args.max_length)
    print(f"Preflight OK: {len(records)} train samples; no validation scene/token overlap; "
          f"max expanded length={max(x[2] for x in dataset.encoded)}", flush=True)
    print("Training GT distribution:", dict(Counter(row["conversations"][1]["value"].strip() for row in records)), flush=True)
    if args.validate_only:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")
    if not args.fp16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable; rerun with --fp16")
    output = Path(args.output_dir)
    expected = manifest_for(args, len(records))
    manifest_path = output / MANIFEST
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Run exists; use --resume or a new --output-dir")
        if json.loads(manifest_path.read_text()) != expected:
            raise ValueError("Resume settings/data/base model/code differ; choose a new output directory")
        checkpoint = latest_complete_checkpoint(output)
        if not checkpoint:
            raise ValueError("No resumable checkpoint; choose a new output directory")
        if json.loads((Path(checkpoint) / MANIFEST).read_text()) != expected:
            raise ValueError("Checkpoint identity differs from this run")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Output directory is not empty")
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = None
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    model = SennaLlavaLlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype,
                                                    low_cpu_mem_usage=True, local_files_only=True)
    model.config.use_cache = False
    model.config.tokenizer_model_max_length = args.max_length
    model.config.tokenizer_padding_side = "right"
    tower = model.get_vision_tower()
    tower.load_model()
    tower.to(dtype=dtype)
    tower.requires_grad_(False)
    # CLIP also has q_proj/k_proj/v_proj names. Bare suffixes would silently
    # inject trainable LoRA into the frozen vision encoder as well.
    target_modules = [name for name, module in model.named_modules()
                      if name.startswith("model.layers.") and isinstance(module, torch.nn.Linear)
                      and name.rsplit(".", 1)[-1] in TARGETS]
    if len(target_modules) != model.config.num_hidden_layers * len(TARGETS):
        raise ValueError("Unexpected Senna language-layer LoRA target layout")
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                           lora_dropout=0.05, target_modules=target_modules,
                                           bias="none", task_type="CAUSAL_LM"))
    for name, parameter in model.get_base_model().named_parameters():
        if name.startswith(("model.mm_projector.", "model.img_adapter.")):
            parameter.requires_grad_(not args.freeze_visual_adapters)
    # Keep small trainable weights and Adam states in FP32. BF16 remains the
    # frozen backbone/storage and autocast compute precision, not optimizer precision.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    if any(p.requires_grad for p in model.get_base_model().get_vision_tower().parameters()):
        raise RuntimeError("CLIP must remain frozen")
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, warmup_ratio=0.03, lr_scheduler_type="cosine",
        weight_decay=0, bf16=not args.fp16, fp16=args.fp16, gradient_checkpointing=True,
        optim="adamw_torch", max_grad_norm=1.0, remove_unused_columns=False,
        logging_steps=1, logging_nan_inf_filter=False, evaluation_strategy="no",
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2, save_safetensors=False,
        dataloader_num_workers=args.workers, report_to=["tensorboard"], seed=args.seed,
        data_seed=args.seed, label_names=["labels"],
    )
    trainer = SennaLoraTrainer(model=model, tokenizer=tokenizer, args=training_args, manifest=expected,
                              train_dataset=dataset, data_collator=PlanningCollator(tokenizer.pad_token_id),
                              callbacks=[FiniteLoss()])
    manifest_path.write_text(json.dumps(expected, indent=2))
    previous_step = (json.loads((Path(checkpoint) / "trainer_state.json").read_text())["global_step"]
                     if checkpoint else 0)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    if checkpoint and trainer.state.global_step == previous_step:
        print(f"TRAINING ALREADY COMPLETE at step {previous_step}; existing final weights/metrics preserved", flush=True)
        return
    trainer.save_model()
    trainer.save_state()
    metrics = dict(result.metrics)
    metrics.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    (output / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("TRAINING COMPLETE", json.dumps(metrics), flush=True)


def main():
    args = parse_args()
    lock_path = Path(str(Path(args.output_dir).resolve()) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(args)


if __name__ == "__main__":
    main()
