"""Original six-view Senna requests, equivalence checks, and CUDA profiling."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

from .analyze_temporal_redundancy import load_analysis_image, similarity_metrics
from .utils.data_utils import CAMERAS, LOG, digest, mark, read_csv, write_csv, write_json
from .utils.profiling_utils import original_request, timed_request
from .utils.stats_utils import distribution


def select_samples(records, manifest, cfg):
    lookup = {(r["sample_token"], r["camera_channel"]): r for r in manifest if str(r["is_key_frame"]).lower() == "true"}
    buckets = defaultdict(list)
    linked = {}
    for sample in records:
        rows = [lookup.get((sample["token"], c)) for c in CAMERAS]
        valid = all(r and Path(r["image_path"]).resolve() == Path(p).resolve() for r, p in zip(rows, sample["images"]))
        if valid:
            front = rows[0]
            linked[sample["token"]] = {"scene_name": front["scene_name"], "scene_token": front["scene_token"], "timestamp_us": int(front["timestamp_us"]), "frames": rows}
            buckets[front["scene_token"]].append(sample)
    if not buckets:
        # No reliable linkage: retain the evaluation workload, report separately.
        for sample in records:
            buckets[sample.get("scene_token", "unknown")].append(sample)
    keys = sorted(buckets)
    random.Random(cfg["experiment"]["seed"]).shuffle(keys)
    keys = keys[:cfg["experiment"]["max_scenes"]]
    original_indices = {s["token"]: i for i, s in enumerate(records)}
    for k in keys:
        buckets[k].sort(key=lambda s: (linked.get(s["token"], {}).get("timestamp_us", original_indices[s["token"]]), s["token"]))
    selected, offset = [], 0
    while len(selected) < cfg["experiment"]["max_profile_samples"]:
        available = False
        for key in keys:
            for sample in buckets[key][offset:offset + 2]:
                available = True
                selected.append(sample)
                if len(selected) == cfg["experiment"]["max_profile_samples"]:
                    return selected, linked
        if not available:
            break
        offset += 2
    return selected, linked


def inference_args(sample, cfg):
    return argparse.Namespace(model_path=cfg["senna"]["checkpoint_path"], model_base=None, query=sample["conversations"][0]["value"], conv_mode="llava_v1", image_file=sample["images"], sep=",", temperature=0, top_p=None, num_beams=1, max_new_tokens=cfg["senna"]["max_new_tokens"])


def profile_summary(rows):
    timing = [r for r in rows if r["measurement_mode"] == "timing"]
    metrics = ("image_file_loading_ms", "image_preprocessing_ms", "host_to_device_ms", "vision_tower_ms", "projector_ms", "img_adapter_ms", "multimodal_preparation_ms", "llm_generation_ms", "other_ms", "end_to_end_ms", "vision_latency_ratio", "projector_latency_ratio", "visual_path_latency_ratio", "llm_generation_latency_ratio", "ideal_visual_path_speedup_bound")
    return {"timing_records": len(timing), "unique_samples": len({r["sample_token"] for r in timing}), "metrics": {key: distribution([r.get(key) for r in timing]) for key in metrics}, "detailed_records_excluded": sum(r["measurement_mode"] != "timing" for r in rows), "speedup_caveat": "Ideal bound removes measured Vision Tower + Projector + img_adapter only, assumes zero added management cost; not an achievable or measured speedup", "prefill_decode": "Not separately measured; llm_generation_ms covers original autoregressive generation after multimodal preparation"}


def matched_results(samples, linked, rows, cfg, out):
    timed = defaultdict(list)
    for row in rows:
        if row["measurement_mode"] == "timing":
            timed[row["sample_token"]].append(row)
    by_scene = defaultdict(list)
    for sample in samples:
        token = sample["token"]
        if token in linked and token in timed:
            by_scene[linked[token]["scene_token"]].append(sample)
    result, issues = [], []
    for scene, seq in sorted(by_scene.items()):
        seq.sort(key=lambda s: linked[s["token"]]["timestamp_us"])
        for prev, current in zip(seq, seq[1:]):
            link = linked[current["token"]]
            records = timed[current["token"]]
            for camera, previous_frame, current_frame in zip(CAMERAS, linked[prev["token"]]["frames"], link["frames"]):
                try:
                    a, _ = load_analysis_image(previous_frame["image_path"], cfg["experiment"]["analysis_resize"])
                    b, _ = load_analysis_image(current_frame["image_path"], cfg["experiment"]["analysis_resize"])
                    metrics, _ = similarity_metrics(a, b, cfg["experiment"]["patch_grid"], cfg["experiment"]["ssim_thresholds"])
                    result.append({"scene_name": link["scene_name"], "scene_token": scene, "camera_channel": camera, "previous_sample_token": prev["token"], "current_sample_token": current["token"], "previous_image_path": previous_frame["image_path"], "current_image_path": current_frame["image_path"], "previous_timestamp_us": previous_frame["timestamp_us"], "current_timestamp_us": current_frame["timestamp_us"], "actual_delta_t_ms": (int(current_frame["timestamp_us"]) - int(previous_frame["timestamp_us"])) / 1000, "ssim": metrics["ssim"], "current_vision_tower_ms": distribution([r["vision_tower_ms"] for r in records])["mean"], "current_end_to_end_ms": distribution([r["end_to_end_ms"] for r in records])["mean"], "current_vision_tower_calls": distribution([r["vision_tower_calls"] for r in records])["mean"], "current_visual_tokens_total": distribution([r["visual_tokens_total"] for r in records])["mean"], "linkage": "all six exact keyframe paths verified against sample_data manifest; timing belongs to current six-view request"})
                except (OSError, ValueError) as exc:
                    issues.append({"current_sample_token": current["token"], "camera_channel": camera, "error": str(exc)})
    write_csv(out / "matched_profile_pairs.csv", result)
    write_json(out / "matched_profile_issues.json", issues)


def run(cfg, args, out):
    import torch
    import transformers
    from data_tools.senna_inputs import check_model, parse_plan, validate_sample
    from llava.model.builder import load_senna_pretrained_model

    status_path = out / "profile_status.json"
    previous_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    desired_modes = {"timing", "detailed"} if args.profile_mode == "both" else {args.profile_mode}
    if args.existing in ("skip", "resume") and previous_status.get("status") == "complete" and desired_modes <= set(previous_status.get("completed_modes", [])):
        LOG.info("Skipping completed profiling modes")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Data analysis can run independently with --mode data")
    check_model(cfg["senna"]["checkpoint_path"])
    records = json.loads(Path(cfg["senna"]["eval_json_path"]).read_text())
    records = [s for s in records if "SPEED plan" in s["conversations"][0]["value"]]
    samples, linked = select_samples(records, read_csv(out / "camera_manifest.csv"), cfg)
    if not samples or len({s["token"] for s in samples}) != len(samples):
        raise ValueError("No planning samples or duplicate tokens")
    for sample in samples:
        validate_sample(sample)
    torch.manual_seed(cfg["experiment"]["seed"])
    torch.cuda.manual_seed_all(cfg["experiment"]["seed"])
    device = cfg["senna"]["device"]
    if not device.startswith("cuda"):
        raise ValueError("Original fp16 profiling requires CUDA")
    torch.cuda.set_device(torch.device(device))
    start = time.perf_counter()
    torch.cuda.init()
    torch.cuda.synchronize()
    initialization_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    tokenizer, model, processor, context_len = load_senna_pretrained_model(cfg["senna"]["checkpoint_path"], None, model_name="llava", device_map="auto" if cfg["senna"]["device_map"] == "auto" else device)
    model.eval()
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - start) * 1000
    checkpoint = Path(cfg["senna"]["checkpoint_path"])
    vision_path = Path(model.config.mm_vision_tower)
    write_json(out / "model_metadata.json", {"model_loading_ms": load_ms, "cuda_initialization_ms": initialization_ms, "class": type(model).__name__, "model_config": model.config.to_dict(), "vision_config": model.get_vision_tower().config.to_dict(), "projector": str(model.get_model().mm_projector), "img_adapter": str(model.get_model().img_adapter), "processor": processor.to_dict(), "context_length": context_len, "torch": torch.__version__, "transformers": transformers.__version__, "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "weight_files": {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in checkpoint.iterdir() if p.is_file()}, "vision_config_sha256": digest(vision_path / "config.json") if vision_path.is_dir() else None, "warmup_iterations": cfg["experiment"]["warmup_iterations"], "generation": vars(inference_args(samples[0], cfg))})
    for i in range(cfg["experiment"]["warmup_iterations"]):
        original_request(inference_args(samples[i % len(samples)], cfg), tokenizer, model, processor)
    LOG.info("Model loaded in %.2fs; warmup complete", load_ms / 1000)

    validations = []
    for sample in samples[:cfg["senna"]["validation_samples"]]:
        request = inference_args(sample, cfg)
        plain, plain_ids, plain_input = original_request(request, tokenizer, model, processor, True)
        measured, measured_ids, measured_input, _ = timed_request(request, tokenizer, model, processor, True)
        valid = plain == measured and plain_ids == measured_ids and plain_input == measured_input and parse_plan(plain) == parse_plan(measured)
        validations.append({"sample_token": sample["token"], "passed": valid, "original_output": plain, "profiled_output": measured, "original_ids": plain_ids, "profiled_ids": measured_ids, "original_input": plain_input, "profiled_input": measured_input, "prediction_equal": parse_plan(plain) == parse_plan(measured)})
    write_json(out / "output_equivalence.json", validations)
    if not all(r["passed"] for r in validations):
        raise RuntimeError("Instrumented inference differs from original; latency results are not accepted")
    expected_outputs = {r["sample_token"]: (r["original_output"], r["original_ids"]) for r in validations}

    rows = read_csv(out / "profiling_samples.csv") if args.existing in ("skip", "resume") else []
    done = {(r["sample_token"], int(r["repeat"]), r["measurement_mode"]) for r in rows}
    for mode in sorted(desired_modes):
        mode_samples = samples if mode == "timing" else samples[:cfg["senna"]["detailed_samples"]]
        repeats = cfg["experiment"]["profile_repeats"] if mode == "timing" else 1
        for sample in mode_samples:
            for repeat in range(repeats):
                if (sample["token"], repeat, mode) in done:
                    continue
                request = inference_args(sample, cfg)
                if mode == "detailed":
                    trace_dir = out / "profiler_trace"
                    trace_dir.mkdir(exist_ok=True)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=False) as prof:
                        answer, ids, _, metrics = timed_request(request, tokenizer, model, processor)
                    prof.export_chrome_trace(str(trace_dir / (sample["token"] + ".json")))
                    (trace_dir / (sample["token"] + "_operators.txt")).write_text(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
                else:
                    answer, ids, _, metrics = timed_request(request, tokenizer, model, processor)
                if sample["token"] in expected_outputs and (answer, ids) != expected_outputs[sample["token"]]:
                    raise RuntimeError("A measured request differs from its verified original output")
                link = linked.get(sample["token"], {})
                row = {"sample_token": sample["token"], "scene_name": link.get("scene_name", ""), "scene_token": link.get("scene_token", sample.get("scene_token", "unmatched")), "timestamp_us": link.get("timestamp_us", ""), "repeat": repeat, "measurement_mode": mode, "camera_order": json.dumps(CAMERAS), "image_paths": json.dumps(sample["images"]), "prompt_sha256": __import__("hashlib").sha256(request.query.encode()).hexdigest(), "prediction": answer, "prediction_plan": json.dumps(parse_plan(answer)), "output_ids": json.dumps(ids), **metrics}
                rows.append(row)
                write_csv(out / "profiling_samples.csv", rows)
                LOG.info("%s %s repeat=%d: %.1fms, vision calls=%d, visual tokens=%d", mode, sample["token"], repeat, metrics["end_to_end_ms"], metrics["vision_tower_calls"], metrics["visual_tokens_total"])
    write_json(out / "profiling_summary.json", profile_summary(rows))
    matched_results(samples, linked, rows, cfg, out)
    completed_modes = set(previous_status.get("completed_modes", [])) | desired_modes if args.existing != "overwrite" else desired_modes
    mark(out, "profile", completed_modes=sorted(completed_modes), equivalence_samples=len(validations), samples=len(samples))


if __name__ == "__main__":
    from .run import main
    raise SystemExit(main("profile"))
