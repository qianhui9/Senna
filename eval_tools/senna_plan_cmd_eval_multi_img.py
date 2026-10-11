"""Evaluate Senna meta-actions with validated inputs and resumable predictions."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from tqdm import tqdm

from data_tools.senna_inputs import SPEED_PLANS, PATH_PLANS, check_model, parse_plan, validate_sample
from data_tools.senna_qa_utils import eval_multi_img_model_wo_init
from llava.model.builder import load_senna_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lora-path", help="Senna LoRA checkpoint directory; --model-path remains the base model")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--resume", action="store_true", help="Reuse completed predictions with identical run inputs")
    parser.add_argument("--allow-legacy-inputs", action="store_true", help="Controlled baseline/ablation only")
    parser.add_argument("--validate-only", action="store_true", help="Check every input without loading the model")
    return parser.parse_args()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_metadata(args, count):
    model = Path(args.model_path).resolve()
    vision = Path(json.loads((model / "config.json").read_text())["mm_vision_tower"])
    metadata = {
        "schema_version": 2,
        "eval_data_path": str(Path(args.eval_data_path).resolve()),
        "eval_data_sha256": digest_file(args.eval_data_path),
        "model_path": str(model),
        "model_config_sha256": digest_file(model / "config.json"),
        "implementation_sha256": {
            name: digest_file(Path(__file__).resolve().parents[1] / name)
            for name in ("data_tools/senna_inputs.py", "data_tools/senna_qa_utils.py",
                         "eval_tools/senna_plan_cmd_eval_multi_img.py")
        },
        "vision_config_sha256": digest_file(vision / "config.json"),
        "vision_weight": {"size": (vision / "pytorch_model.bin").stat().st_size,
                          "mtime_ns": (vision / "pytorch_model.bin").stat().st_mtime_ns},
        "model_files": {
            name: {"size": (model / name).stat().st_size, "mtime_ns": (model / name).stat().st_mtime_ns}
            for name in ("pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin", "tokenizer.model")
        },
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "allow_legacy_inputs": args.allow_legacy_inputs,
        "selected_samples": count,
        "conv_mode": "llava_v1", "temperature": 0, "num_beams": 1,
        "metric": "strict_speed_path_exact_match",
    }
    if args.lora_path:
        from train_tools.lora_io import MANIFEST, EXTRA_WEIGHTS
        adapter = Path(args.lora_path).resolve()
        metadata["lora"] = {"path": str(adapter), "files": {
            name: digest_file(adapter / name)
            for name in (MANIFEST, "adapter_config.json", "adapter_model.bin", EXTRA_WEIGHTS)},
            "loader_sha256": digest_file(Path(__file__).resolve().parents[1] / "train_tools/lora_io.py")}
    return metadata


def read_checkpoint(path, metadata):
    records = []
    with path.open("r+b") as file:
        first = json.loads(file.readline())
        if first != {"type": "metadata", "value": metadata}:
            raise ValueError("Resume input/model/settings differ from saved run; choose a new output path.")
        while True:
            offset = file.tell()
            line = file.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                file.seek(offset)
                file.truncate()
                print("Discarded an interrupted final checkpoint line; its sample will be retried.")
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if file.read():
                    raise ValueError("Invalid checkpoint record before the end of the file")
                # Recover only an interrupted final JSONL write.
                file.seek(offset)
                file.truncate()
                print("Discarded an interrupted final checkpoint line; its sample will be retried.")
                break
            if record.get("type") != "prediction":
                raise ValueError("Unexpected checkpoint record type")
            records.append(record["value"])
    return records


def summarize(records):
    speed_confusion, path_confusion = Counter(), Counter()
    joint_total, joint_correct = Counter(), Counter()
    invalid = correct = 0
    errors = {}
    for record in records:
        gt_speed, gt_path = parse_plan(record["gt"])
        pred_speed, pred_path = parse_plan(record["pred"])
        valid = pred_speed is not None
        invalid += int(not valid)
        speed_confusion[gt_speed, pred_speed or "INVALID"] += 1
        path_confusion[gt_path, pred_path or "INVALID"] += 1
        key = gt_speed + "_" + gt_path
        joint_total[key] += 1
        is_correct = (gt_speed, gt_path) == (pred_speed, pred_path)
        correct += int(is_correct)
        joint_correct[key] += int(is_correct)
        if not is_correct:
            errors[record["token"]] = {"gt": record["gt"], "pred": record["pred"]}

    class_metrics = {}
    for classes, confusion in ((SPEED_PLANS, speed_confusion), (PATH_PLANS, path_confusion)):
        for key in classes:
            support = sum(value for (gt, pred), value in confusion.items() if gt == key)
            tp = confusion[key, key]
            fp = sum(value for (gt, pred), value in confusion.items() if pred == key and gt != key)
            fn = support - tp
            class_metrics[key] = {
                "support": support, "predicted": tp + fp, "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0,
                "recall": tp / support if support else 0,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            }
    count = len(records)
    speed_correct = sum(speed_confusion[key, key] for key in SPEED_PLANS)
    path_correct = sum(path_confusion[key, key] for key in PATH_PLANS)
    lines = [
        f"Total Number: {count}", f"Correct Number: {correct}",
        f"Planning Accuracy: {100 * correct / count:.2f}%",
        f"Speed Accuracy: {100 * speed_correct / count:.2f}%",
        f"Path Accuracy: {100 * path_correct / count:.2f}%",
        f"Invalid Outputs: {invalid}",
    ]
    for key, value in sorted(joint_total.items()):
        lines.append(f"{key}: num: {value}, correct num: {joint_correct[key]}, {100 * joint_correct[key] / value:.2f}%")
    def confusion_dict(classes, confusion):
        return {gt: {pred: confusion[gt, pred] for pred in (*classes, "INVALID")} for gt in classes}
    return {
        "summary": "\n".join(lines),
        "f1_score": {key: value["f1"] for key, value in class_metrics.items()},
        "metrics": {
            "total": count, "correct": correct, "planning_accuracy": correct / count,
            "speed_accuracy": speed_correct / count, "path_accuracy": path_correct / count,
            "invalid_outputs": invalid, "class_metrics": class_metrics,
            "speed_confusion": confusion_dict(SPEED_PLANS, speed_confusion),
            "path_confusion": confusion_dict(PATH_PLANS, path_confusion),
            "majority_joint_baseline": max(joint_total.values()) / count,
        },
        "errors": errors, "predictions": records,
    }


def evaluate(args):
    with open(args.eval_data_path, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Expected a list of Senna planning records")
    samples = [sample for sample in data if "SPEED plan" in sample["conversations"][0]["value"]]
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[:args.max_samples]
    if not samples:
        raise ValueError("No planning samples")
    tokens = [sample["token"] for sample in samples]
    if len(set(tokens)) != len(tokens):
        raise ValueError("Duplicate sample tokens in planning inputs")
    for sample in samples:
        validate_sample(sample, allow_legacy=args.allow_legacy_inputs)
    check_model(args.model_path)
    if args.lora_path:
        from train_tools.lora_io import check_lora
        check_lora(args.lora_path, args.model_path)
    print(f"Preflight OK: {len(samples)} samples, {6 * len(samples)} images.", flush=True)
    if args.validate_only:
        return

    output = Path(args.save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(str(output) + ".jsonl")
    metadata = run_metadata(args, len(samples))
    records = read_checkpoint(checkpoint, metadata) if args.resume and checkpoint.exists() else []
    if not args.resume and (output.exists() or checkpoint.exists()):
        raise FileExistsError("Output already exists; use --resume or choose a new --save-path.")
    by_token = {sample["token"]: sample for sample in samples}
    seen = set()
    for record in records:
        if record["token"] in seen or record["token"] not in by_token:
            raise ValueError("Duplicate or foreign token in checkpoint")
        if record["gt"] != by_token[record["token"]]["conversations"][1]["value"].strip():
            raise ValueError("Ground truth differs from checkpoint")
        seen.add(record["token"])
    if not checkpoint.exists():
        with checkpoint.open("x", encoding="utf-8") as file:
            file.write(json.dumps({"type": "metadata", "value": metadata}) + "\n")
    pending = [sample for sample in samples if sample["token"] not in seen]
    if pending:
        tokenizer, model, processor, _ = load_senna_pretrained_model(
            args.model_path, None, model_name="llava", device_map=args.gpu
        )
        if args.lora_path:
            from train_tools.lora_io import load_lora_for_inference
            model = load_lora_for_inference(model, args.lora_path, args.model_path)
        model.eval()
        with checkpoint.open("a", encoding="utf-8") as file:
            for sample in tqdm(pending, initial=len(records), total=len(samples), desc="Planning inference"):
                inference_args = argparse.Namespace(
                    model_path=args.model_path, model_base=None,
                    query=sample["conversations"][0]["value"], conv_mode="llava_v1",
                    image_file=sample["images"], sep=",", temperature=0, top_p=None, num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                )
                start = time.monotonic()
                answer = eval_multi_img_model_wo_init(inference_args, tokenizer, model, processor)
                record = {
                    "token": sample["token"], "gt": sample["conversations"][1]["value"].strip(),
                    "pred": answer, "navigation": sample.get("navigation"),
                    "elapsed_seconds": time.monotonic() - start,
                }
                file.write(json.dumps({"type": "prediction", "value": record}, ensure_ascii=False) + "\n")
                file.flush()
                records.append(record)
    result = summarize(records)
    result["metadata"] = metadata
    temporary = Path(str(output) + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    os.replace(temporary, output)
    print("\n" + result["summary"])
    print("\nF1 Score")
    for key, value in result["f1_score"].items():
        print(f"{key}: {value:.6f}")
    print(f"\nEval results saved to {output}\nPer-sample checkpoint: {checkpoint}")


def main():
    args = parse_args()
    # Prevent two jobs from appending to the same checkpoint.
    output = Path(args.save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(output) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate(args)


if __name__ == "__main__":
    main()
