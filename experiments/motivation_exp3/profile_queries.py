"""Original Senna inference with Experiment 1's unmodified instrumentation."""
import contextlib
import json
import hashlib
import os
from pathlib import Path
import subprocess

from data_tools.senna_inputs import CAMERAS, IMAGE_PROMPT, check_model, parse_plan
from experiments.motivation_exp1.profile_senna import inference_args, valid_timing_values
from experiments.motivation_exp1.utils.data_utils import write_csv, write_json
from experiments.motivation_exp1.utils.profiling_utils import original_request, timed_request
from .build_query_sequences import canonical, model_sample


def strict_correct(answer, query):
    kind = query["query_type"]
    if kind == "meta_action_planning":
        parsed = parse_plan(answer)
        return parsed != (None, None) and parsed == parse_plan(query["ground_truth"])
    if kind == "motion_prediction":
        if not answer.strip().startswith("Object 1:"):
            return False
        parsed = parse_plan(answer.strip()[len("Object 1:"):].strip())
        return parsed != (None, None) and parsed == parse_plan(query["ground_truth"][len("Object 1:"):].strip())
    return answer.strip().rstrip(".").casefold() == query["ground_truth"].casefold()


def load_journal(path):
    if not path.exists():
        return []
    content = path.read_bytes()
    # Each complete line is committed; discard only an interrupted final write.
    if content and not content.endswith(b"\n"):
        content = content[:content.rfind(b"\n") + 1]
        path.write_bytes(content)
    rows = [json.loads(line) for line in content.splitlines()]
    for row in rows:
        if "payload_sha256" in row:
            payload = {k: v for k, v in row.items() if k != "payload_sha256"}
            if hashlib.sha256(canonical(payload).encode()).hexdigest() != row["payload_sha256"]:
                raise ValueError("Corrupt profile checkpoint record")
    keys = [(r["record_id"], r["repeat"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate profile checkpoint records")
    return rows


def hardware_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                             "--format=csv,noheader"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def profile(rows, cfg, out):
    journal = out / "query_profile_samples.jsonl"
    measured = load_journal(journal)
    done = {(r["record_id"], r["repeat"]) for r in measured}
    expected = {(r["record_id"], repeat) for r in rows for repeat in range(cfg["senna"]["profile_repeats"])}
    if not done <= expected:
        raise ValueError("Profile checkpoint contains unknown records")
    if done == expected:
        write_csv(out / "query_profile_samples.csv", measured)
        return measured
    import torch
    import transformers
    from llava.model.builder import load_senna_pretrained_model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --mode data for offline analysis")
    check_model(cfg["senna"]["checkpoint_path"])
    for q in rows:
        sample = model_sample(q)
        if len(sample["images"]) != 6 or not sample["conversations"][0]["value"].startswith(IMAGE_PROMPT):
            raise ValueError("Invalid original six-view input")
        if tuple(Path(p).parent.name for p in sample["images"]) != CAMERAS:
            raise ValueError("Camera order differs from original protocol")
        if any(not Path(p).is_file() for p in sample["images"]):
            raise FileNotFoundError("Missing image")
    torch.manual_seed(cfg["experiment"]["seed"])
    torch.cuda.manual_seed_all(cfg["experiment"]["seed"])
    torch.cuda.set_device(torch.device(cfg["senna"]["device"]))
    wrapped_cfg = {"senna": cfg["senna"]}
    requests = lambda q: inference_args(model_sample(q), wrapped_cfg)
    # The original helper prints inferred-mode warnings; retain them in a log.
    with (out / "original_inference.log").open("a") as log, contextlib.redirect_stdout(log):
        before = hardware_snapshot()
        tokenizer, model, processor, context_len = load_senna_pretrained_model(
            cfg["senna"]["checkpoint_path"], None, model_name="llava", device_map=cfg["senna"]["device"])
        model.eval()
        representatives = list({q["query_type"]: q for q in rows}.values())
        for i in range(cfg["senna"]["warmup_iterations"]):
            original_request(requests(representatives[i % len(representatives)]), tokenizer, model, processor)
        equivalence = []
        for q in representatives:
            plain, ids, inputs = original_request(requests(q), tokenizer, model, processor, True)
            instrumented, timed_ids, timed_inputs, _ = timed_request(requests(q), tokenizer, model, processor, True)
            equivalence.append({"query_type": q["query_type"], "sample_token": q["sample_token"],
                                "passed": plain == instrumented and ids == timed_ids and inputs == timed_inputs,
                                "original_answer": plain, "instrumented_answer": instrumented,
                                "original_input": inputs, "instrumented_input": timed_inputs})
        write_json(out / "query_output_equivalence.json", equivalence)
        if not all(e["passed"] for e in equivalence):
            raise RuntimeError("Instrumented inference differs from original")
        metadata = {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(), "device": str(model.device), "context_length": context_len,
                    "hardware_before_loading": before, "warmup_iterations": cfg["senna"]["warmup_iterations"],
                    "generation": vars(requests(rows[0])), "model_config": model.config.to_dict(),
                    "measurement": "Original E1 timed_request; warmup/equivalence excluded; no reuse between requests",
                    "concurrent_load_policy": cfg.get("measurement_context", "inspect hardware snapshot"),
                    "prediction_note": "Motion velocity uses past annotations; navigation is separately marked offline_future_route_6s"}
        write_json(out / "query_model_metadata.json", metadata)
        try:
            with journal.open("a") as handle:
                for index, q in enumerate(rows):
                    for repeat in range(cfg["senna"]["profile_repeats"]):
                        if (q["record_id"], repeat) in done:
                            continue
                        answer, _, _, timing = timed_request(requests(q), tokenizer, model, processor)
                        timing = valid_timing_values([{**timing, "measurement_mode": "timing"}])[0]
                        record = {"record_id": q["record_id"], "repeat": repeat, "scene_token": q["scene_token"],
                                  "sample_token": q["sample_token"], "sequence_id": q["sequence_id"],
                                  "sequence_length": q["sequence_length"], "workload": q["workload"],
                                  "query_type": q["query_type"], "raw_answer": answer, "ground_truth": q["ground_truth"],
                                  "strict_correct": strict_correct(answer, q), "generated_token_count": timing["output_tokens"], **timing}
                        record["payload_sha256"] = hashlib.sha256(canonical(record).encode()).hexdigest()
                        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        measured.append(record)
                        print(f"Profiled {len(measured)}/{len(expected)}: {q['query_type']}", file=log, flush=True)
        finally:
            write_csv(out / "query_profile_samples.csv", measured)
            metadata["hardware_after_measurements"] = hardware_snapshot()
            metadata["completed_measurements"] = len(measured)
            write_json(out / "query_model_metadata.json", metadata)
    return measured
