"""Experiment 3 semantic queries; the old feature pipeline remains run_exp3.py."""
import argparse
import copy
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.motivation_exp3"

import yaml

from experiments.motivation_exp1.utils.data_utils import ROOT, digest, load_nuscenes, write_json, write_csv
from .build_query_sequences import QUERY_TYPES, build, canonical
from .analyze_query_dependencies import analyze
from .profile_queries import profile, load_journal, hardware_snapshot
from .query_reporting import aggregate, plot, report

BUILD_FILES = ("queries.json", "base_queries.json", "sequences.json", "query_traces.csv", "query_dependencies.csv", "excluded_queries.csv", "query_build_summary.json")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    p.add_argument("--mode", choices=["all", "data", "profile", "report"], default="all")
    p.add_argument("--existing", choices=["resume", "overwrite"], default="resume")
    p.add_argument("--output-dir")
    p.add_argument("--dataroot")
    p.add_argument("--checkpoint-path")
    p.add_argument("--device")
    p.add_argument("--max-scenes", type=int)
    p.add_argument("--sequence-lengths", nargs="+", type=int)
    p.add_argument("--warmup-iterations", type=int)
    p.add_argument("--profile-repeats", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    policy = p.add_mutually_exclusive_group()
    policy.add_argument("--allow-shared-gpu", action="store_true", help="Measure under concurrent load and explicitly label that condition")
    policy.add_argument("--wait-for-idle", action="store_true", help="Wait for two idle hardware snapshots before loading the model")
    p.add_argument("--gpu-poll-seconds", type=int, default=30)
    return p


def configuration(args):
    if not 1 <= args.gpu_poll_seconds <= 60:
        raise ValueError("GPU poll interval must be in [1,60] seconds")
    cfg = yaml.safe_load(Path(args.config).read_text())["semantic_queries"]
    destinations = {"output_dir": ("output", "directory"), "dataroot": ("nuscenes", "dataroot"),
                    "checkpoint_path": ("senna", "checkpoint_path"), "device": ("senna", "device"),
                    "max_scenes": ("experiment", "max_scenes"), "sequence_lengths": ("experiment", "sequence_lengths"),
                    "warmup_iterations": ("senna", "warmup_iterations"), "profile_repeats": ("senna", "profile_repeats")}
    for name, (section, key) in destinations.items():
        if getattr(args, name) is not None:
            cfg[section][key] = getattr(args, name)
    if not args.checkpoint_path:
        cfg["senna"]["checkpoint_path"] = os.environ.get("SENNA_MODEL_PATH", cfg["senna"]["checkpoint_path"])
    if not args.dataroot:
        cfg["nuscenes"]["dataroot"] = os.environ.get("NUSCENES_ROOT", cfg["nuscenes"]["dataroot"])
    if args.smoke_test:
        cfg["experiment"]["max_scenes"] = min(2, cfg["experiment"]["max_scenes"])
        cfg["experiment"]["sequence_lengths"] = [5]
        cfg["senna"]["warmup_iterations"] = 2
        cfg["senna"]["profile_repeats"] = 1
        if not args.output_dir:
            cfg["output"]["directory"] += "_smoke"
    for section, key in (("output", "directory"), ("nuscenes", "dataroot"), ("senna", "checkpoint_path")):
        cfg[section][key] = str((ROOT / Path(cfg[section][key]).expanduser()).resolve())
    e, s = cfg["experiment"], cfg["selection"]
    if (e["max_scenes"] < 1 or e["bootstrap_iterations"] < 1 or not e["sequence_lengths"]
            or any(n < 2 for n in e["sequence_lengths"]) or len(set(e["sequence_lengths"])) != len(e["sequence_lengths"])):
        raise ValueError("Require positive scenes/bootstrap and unique sequence lengths >=2")
    if (not e["query_types"] or len(set(e["query_types"])) != len(e["query_types"])
            or not set(e["query_types"]) <= set(QUERY_TYPES) or not e["workloads"] or not set(e["workloads"]) <= {"fixed", "mixed"}):
        raise ValueError("Invalid query types or workloads")
    if any(cfg["senna"][key] < 1 for key in ("warmup_iterations", "profile_repeats", "max_new_tokens")):
        raise ValueError("Warmup, repeats and generation length must be positive")
    if (not 0 < s["existence_distance_m"] <= s["max_distance_m"] or not 0 < s["front_half_angle_degrees"] < 90
            or s["forward_min_m"] < 0 or s["min_sensor_points"] < 0 or s["future_steps"] != 6
            or s["expected_interval_s"] != .5 or not 0 <= s["interval_tolerance_s"] < .5
            or len(s["position_labels"]) != len(s["position_lateral_edges_m"]) + 1
            or s["position_lateral_edges_m"] != sorted(set(s["position_lateral_edges_m"]))):
        raise ValueError("Invalid deterministic selection/discretization; original motion labels require 6 x .5s")
    if e["query_types"] and cfg["nuscenes"]["split"] not in {"val", "train"}:
        raise ValueError("Only official train/val splits supported")
    cfg["measurement_context"] = ("shared GPU allowed; concurrent workload can affect latency, not an exclusive-hardware benchmark"
                                  if args.allow_shared_gpu else "preflight requires <500 MiB used and <5% GPU utilization; no ongoing exclusive reservation")
    return cfg


def file_identity(path):
    p = Path(path)
    return {"path": str(p.resolve()), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}


def identity(cfg):
    model = Path(cfg["senna"]["checkpoint_path"])
    dataroot = Path(cfg["nuscenes"]["dataroot"]) / cfg["nuscenes"]["version"]
    sources = [*Path(__file__).parent.glob("*quer*.py"), Path(__file__).with_name("query_reporting.py"),
               ROOT / "data_tools/senna_inputs.py", ROOT / "data_tools/senna_qa_utils.py",
               ROOT / "eval_tools/prepare_senna_plan_eval.py", ROOT / "experiments/motivation_exp1/profile_senna.py",
               * (ROOT / "experiments/motivation_exp1/utils").glob("*.py"), *(ROOT / "llava").rglob("*.py")]
    configuration = copy.deepcopy(cfg)
    configuration.pop("measurement_context", None)
    signature = {"configuration": configuration, "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(sources))},
                 "model_files": [file_identity(p) for p in sorted(model.iterdir()) if p.is_file()],
                 "nuscenes_tables": [file_identity(p) for p in sorted(dataroot.glob("*.json"))],
                 "model_config_sha256": digest(model / "config.json")}
    vision = Path(json.loads((model / "config.json").read_text())["mm_vision_tower"])
    if vision.is_dir():
        signature["vision_files"] = [file_identity(p) for p in sorted(vision.iterdir()) if p.is_file()]
    return {"sha256": hashlib.sha256(canonical(signature).encode()).hexdigest(), "inputs": signature}


def initialize(cfg, args):
    out = Path(cfg["output"]["directory"])
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = (out.parent / (out.name + ".lock")).open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fingerprint = identity(cfg)
        manifest_path = out / "query_run_manifest.json"
        if args.existing == "overwrite" and manifest_path.exists():
            archive = out.with_name(out.name + "_previous_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
            shutil.move(str(out), archive)
        if out.exists() and any(out.iterdir()) and not manifest_path.exists():
            raise ValueError("Output has unrelated files; choose an empty directory (old feature results are protected)")
        out.mkdir(exist_ok=True)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["identity"]["sha256"] != fingerprint["sha256"]:
                raise ValueError("Configuration, source, dataset or model changed; use a new output directory or --existing overwrite")
            for name, checksum in manifest.get("artifacts", {}).items():
                if not (out / name).is_file() or digest(out / name) != checksum:
                    raise ValueError("Modified or missing completed artifact: " + name)
        else:
            manifest = {"identity": fingerprint, "artifacts": {}}
            write_json(manifest_path, manifest)
        (out / "resolved_query_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        return out, manifest, lock
    except BaseException:
        lock.close()
        raise


def gpu_is_idle(snapshot, device_index):
    gpu = snapshot.splitlines()[device_index].split(",")
    return float(gpu[3].strip().split()[0]) < 500 and float(gpu[5].strip().split()[0]) < 5


def wait_for_idle(cfg, args, out):
    device_index = int(cfg["senna"]["device"].split(":")[-1])
    consecutive = 0
    while consecutive < 2:
        snapshot = hardware_snapshot()
        consecutive = consecutive + 1 if gpu_is_idle(snapshot, device_index) else 0
        write_json(out / "query_invocation_status.json", {
            "mode": args.mode, "phase": "waiting_for_idle_gpu", "device": cfg["senna"]["device"],
            "hardware_snapshot": snapshot, "consecutive_idle_snapshots": consecutive,
            "updated_utc": datetime.now(timezone.utc).isoformat(), "poll_seconds": args.gpu_poll_seconds,
            "note": "No model loaded while waiting; shared-load measurements are not included"})
        if consecutive < 2:
            logging.info("Waiting for idle %s: %s", cfg["senna"]["device"], snapshot.replace("\n", " | "))
            time.sleep(args.gpu_poll_seconds)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    args = parser().parse_args(argv)
    cfg = configuration(args)
    if args.dry_run:
        from data_tools.senna_inputs import check_model
        check_model(cfg["senna"]["checkpoint_path"])
        tables = {name: (Path(cfg["nuscenes"]["dataroot"]) / cfg["nuscenes"]["version"] / (name + ".json")).is_file()
                  for name in ("scene", "sample", "sample_data", "sample_annotation", "ego_pose", "calibrated_sensor")}
        print(json.dumps({"configuration": cfg, "tables": tables, "hardware": hardware_snapshot()}, indent=2))
        return 0 if all(tables.values()) else 1
    out, manifest, lock = initialize(cfg, args)
    try:
        if not manifest.get("build_complete"):
            if args.mode in {"profile", "report"}:
                raise ValueError("Build queries first using --mode data or --mode all")
            nusc = load_nuscenes(cfg)
            from eval_tools.prepare_senna_plan_eval import CachedCanBus
            can_bus = CachedCanBus(dataroot=cfg["nuscenes"]["dataroot"])
            rows, base, sequences = build(nusc, can_bus, cfg, out)
            del nusc, can_bus
            import gc
            gc.collect()
            if not rows:
                raise ValueError("No valid queries")
            manifest["build_complete"] = True
            manifest["artifacts"].update({name: digest(out / name) for name in BUILD_FILES})
            write_json(out / "query_run_manifest.json", manifest)
        else:
            rows, base, sequences = [json.loads((out / name).read_text()) for name in ("queries.json", "base_queries.json", "sequences.json")]
        transitions, pairs = analyze(rows, base, out)
        measured = load_journal(out / "query_profile_samples.jsonl")
        error = None
        if args.mode in {"all", "profile"}:
            try:
                expected = len(rows) * cfg["senna"]["profile_repeats"]
                if len(measured) < expected and not args.allow_shared_gpu:
                    if args.wait_for_idle:
                        wait_for_idle(cfg, args, out)
                    elif not gpu_is_idle(hardware_snapshot(), int(cfg["senna"]["device"].split(":")[-1])):
                        raise RuntimeError("Selected GPU has concurrent load; retry with --wait-for-idle")
                measured = profile(rows, cfg, out)
            except Exception as exc:
                logging.exception("Original Senna profiling incomplete")
                error = f"{type(exc).__name__}: {exc}"
                measured = load_journal(out / "query_profile_samples.jsonl")
        if not (out / "query_profile_samples.csv").exists():
            write_csv(out / "query_profile_samples.csv", measured, None if measured else
                      ["record_id", "scene_token", "sample_token", "sequence_id", "query_type", "repeat", "raw_answer", "ground_truth",
                       "generated_token_count", "llm_prefill_ms", "llm_decode_ms", "end_to_end_ms"])
        summary, sequence_rows = aggregate(rows, base, sequences, transitions, pairs, measured, cfg, out)
        plot(summary, pairs, sequence_rows, cfg, out)
        report(summary, cfg, out, error)
        if summary["status"] == "profile_complete":
            manifest["artifacts"].update({name: digest(out / name) for name in
                ("query_profile_samples.csv", "query_profile_samples.jsonl", "query_model_metadata.json", "query_output_equivalence.json")})
            write_json(out / "query_run_manifest.json", manifest)
        write_json(out / "query_invocation_status.json", {"mode": args.mode, "error": error, "summary_status": summary["status"]})
        print(json.dumps({k: summary[k] for k in ("status", "scenes", "sequence_count", "valid_query_events", "canonical_adjacent_pairs", "profile_measurements")}, indent=2))
        return 1 if error else 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
