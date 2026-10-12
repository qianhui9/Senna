"""Shared CLI, provenance, atomic output, and official devkit access."""
import argparse
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK")
LOG = logging.getLogger("motivation_exp1")


def parser(default_mode="all"):
    p = argparse.ArgumentParser(description="WorldStream E1, original Senna and nuScenes only")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--mode", choices=["all", "data", "index", "redundancy", "objects", "profile", "aggregate", "plot", "report"], default=default_mode)
    for name in ("checkpoint-path", "eval-json-path", "dataroot", "version", "device", "output-dir"):
        p.add_argument("--" + name)
    p.add_argument("--device-map", choices=["single", "auto"])
    p.add_argument("--precision", choices=["fp16"])
    for name in ("max-scenes", "max-pairs", "max-profile-samples", "warmup-iterations", "profile-repeats", "max-new-tokens", "seed"):
        p.add_argument("--" + name, type=int)
    p.add_argument("--temporal-gaps-seconds", type=float, nargs="+")
    p.add_argument("--patch-grid", type=int, nargs=2)
    p.add_argument("--analysis-resize", type=int, nargs=2)
    p.add_argument("--gap-tolerance-seconds", type=float)
    p.add_argument("--ssim-thresholds", type=float, nargs="+")
    p.add_argument("--bootstrap-iterations", type=int)
    p.add_argument("--validation-samples", type=int)
    p.add_argument("--detailed-samples", type=int)
    p.add_argument("--no-save-visualizations", action="store_true")
    p.add_argument("--profile-mode", choices=["timing", "detailed", "both"], default="timing")
    p.add_argument("--existing", choices=["skip", "overwrite", "resume"], default="skip")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p


def configuration(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    destinations = {
        "checkpoint_path": ("senna", "checkpoint_path"), "eval_json_path": ("senna", "eval_json_path"),
        "device": ("senna", "device"), "device_map": ("senna", "device_map"),
        "max_new_tokens": ("senna", "max_new_tokens"), "dataroot": ("nuscenes", "dataroot"),
        "version": ("nuscenes", "version"), "output_dir": ("output", "directory"),
        "precision": ("senna", "precision"), "validation_samples": ("senna", "validation_samples"), "detailed_samples": ("senna", "detailed_samples"),
        "max_pairs": ("experiment", "max_temporal_pairs"),
    }
    for key in ("max_scenes", "max_profile_samples", "warmup_iterations", "profile_repeats", "seed", "temporal_gaps_seconds", "patch_grid", "analysis_resize", "gap_tolerance_seconds", "ssim_thresholds", "bootstrap_iterations"):
        destinations[key] = ("experiment", key)
    for arg, (section, key) in destinations.items():
        value = getattr(args, arg, None)
        if value is not None:
            cfg[section][key] = value
    cfg["senna"]["checkpoint_path"] = cfg["senna"]["checkpoint_path"] or os.environ.get("SENNA_MODEL_PATH", "")
    cfg["nuscenes"]["dataroot"] = cfg["nuscenes"]["dataroot"] or os.environ.get("NUSCENES_ROOT", "")
    if args.no_save_visualizations:
        cfg["experiment"]["save_visualizations"] = False
    if args.smoke_test:
        for key, limit in (("max_scenes", 2), ("max_temporal_pairs", 36), ("max_profile_samples", 2), ("warmup_iterations", 1), ("profile_repeats", 1)):
            cfg["experiment"][key] = min(cfg["experiment"][key], limit)
    e = cfg["experiment"]
    for key in ("max_scenes", "max_temporal_pairs", "max_profile_samples", "profile_repeats", "bootstrap_iterations"):
        if e[key] < 1:
            raise ValueError(f"{key} must be positive")
    if e["warmup_iterations"] < 1:
        raise ValueError("At least one warmup iteration is required")
    if cfg["senna"]["precision"] != "fp16":
        raise ValueError("The local original evaluation uses fp16; other precision would change the baseline")
    if min(cfg["senna"]["max_new_tokens"], cfg["senna"]["validation_samples"], cfg["senna"]["detailed_samples"]) < 1:
        raise ValueError("Generation length, validation_samples and detailed_samples must be positive")
    if len(e["temporal_gaps_seconds"]) != len(set(e["temporal_gaps_seconds"])) or any(g <= 0 for g in e["temporal_gaps_seconds"]):
        raise ValueError("Temporal gaps must be distinct and positive")
    if any(t < 0 or t > 1 for t in e["ssim_thresholds"]) or e["gap_tolerance_seconds"] < 0:
        raise ValueError("Invalid similarity threshold or gap tolerance")
    w, h = e["analysis_resize"]
    rows, cols = e["patch_grid"]
    if min(rows, cols) < 1 or min(w // cols, h // rows) < 3:
        raise ValueError("Each analysis patch must be at least 3 x 3 pixels")
    for section, key in (("senna", "checkpoint_path"), ("senna", "eval_json_path"), ("nuscenes", "dataroot"), ("output", "directory")):
        value = cfg[section][key]
        if value:
            path = Path(os.path.expandvars(value)).expanduser()
            cfg[section][key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    return cfg


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else ["status"])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_nuscenes(cfg):
    from nuscenes.nuscenes import NuScenes
    LOG.info("Loading official nuScenes devkit (full metadata tables can require substantial RAM)")
    return NuScenes(dataroot=cfg["nuscenes"]["dataroot"], version=cfg["nuscenes"]["version"], verbose=False)


def selected_scenes(nusc, cfg):
    scenes = sorted(nusc.scene, key=lambda s: s["name"])
    # Prefer the existing evaluation workload, without selecting by similarity.
    eval_path = Path(cfg["senna"]["eval_json_path"])
    if eval_path.is_file():
        evaluation = json.loads(eval_path.read_text())
        scene_tokens = {s["scene_token"] for s in evaluation if s.get("scene_token")}
        if scene_tokens:
            scenes = [s for s in scenes if s["token"] in scene_tokens]
    random.Random(cfg["experiment"]["seed"]).shuffle(scenes)
    return sorted(scenes[:cfg["experiment"]["max_scenes"]], key=lambda s: s["name"])


def initialize(cfg, args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    random.seed(cfg["experiment"]["seed"])
    np.random.seed(cfg["experiment"]["seed"])
    out = Path(cfg["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    sources = ["data_tools/senna_qa_utils.py", "data_tools/senna_inputs.py", "llava/model/builder.py", "llava/senna/senna_llava_llama.py", "llava/senna/senna_llava_arch.py", "llava/model/multimodal_encoder/clip_encoder.py"]
    sources += [str(p.relative_to(ROOT)) for p in sorted(DEFAULT_CONFIG.parent.rglob("*.py"))]
    identity = {"config": cfg, "sources": {s: digest(ROOT / s) for s in sources}}
    identity["packages"] = {}
    for package in ("torch", "transformers", "numpy", "scipy", "Pillow", "nuscenes-devkit", "matplotlib", "PyYAML"):
        try:
            identity["packages"][package] = version(package)
        except PackageNotFoundError:
            identity["packages"][package] = "missing"
    for section, key in (("senna", "eval_json_path"), ("senna", "checkpoint_path")):
        p = Path(cfg[section][key])
        if key == "checkpoint_path":
            p = p / "config.json"
        if p.is_file():
            identity[key + "_sha256"] = digest(p)
    checkpoint = Path(cfg["senna"]["checkpoint_path"])
    if cfg["senna"]["checkpoint_path"] and checkpoint.is_dir():
        identity["model_files"] = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in checkpoint.iterdir() if p.is_file()}
        config_path = checkpoint / "config.json"
        if config_path.is_file():
            tower = Path(json.loads(config_path.read_text()).get("mm_vision_tower", ""))
            if tower.is_dir():
                identity["vision_files"] = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in tower.iterdir() if p.is_file()}
    table_root = Path(cfg["nuscenes"]["dataroot"]) / cfg["nuscenes"]["version"]
    identity["dataset_tables"] = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in table_root.glob("*.json")} if table_root.is_dir() else {}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    old = out / "run_metadata.json"
    if old.exists():
        changed = json.loads(old.read_text())["fingerprint"] != fingerprint
        if changed and args.existing != "overwrite":
            raise ValueError("Existing outputs have different configuration/code/data; use a new --output-dir or --existing overwrite")
        if changed and args.existing == "overwrite":
            # Never label stale outputs with new provenance. Clear experiment-
            # owned products only; user files and unrelated directories survive.
            owned = {"camera_manifest.csv", "temporal_pairs.csv", "temporal_similarity.csv", "patch_similarity.csv", "object_persistence.csv", "profiling_samples.csv", "matched_profile_pairs.csv", "scene_statistics.csv", "experiment_report.md", "model_metadata.json", "output_equivalence.json", "aggregate_summary.json", "profiling_summary.json", "manifest_summary.json", "manifest_issues.json", "similarity_issues.json", "object_issues.json", "matched_profile_issues.json", "figure_metadata.json", "invocation_status.json", "verification_results.json"}
            for p in out.iterdir():
                if p.is_file() and (p.name in owned or p.name.endswith("_status.json")):
                    p.unlink()
            for folder, patterns in (("figures", ("M1*.png", "M1*.pdf", "matched_similarity_latency.*")), ("profiler_trace", ("*.json", "*_operators.txt"))):
                for pattern in patterns:
                    for p in (out / folder).glob(pattern):
                        p.unlink()
            LOG.info("Configuration/code/data changed: cleared owned experiment outputs before overwrite")
    metadata = {**identity, "fingerprint": fingerprint, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
                "python": sys.version, "argv": sys.argv, "data_kind": "real_dataset", "camera_order": CAMERAS}
    write_json(old, metadata)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def completed(out, stage, args):
    return args.existing in ("skip", "resume") and (out / (stage + "_status.json")).exists() and json.loads((out / (stage + "_status.json")).read_text()).get("status") == "complete"


def mark(out, stage, status="complete", **kwargs):
    write_json(out / (stage + "_status.json"), {"status": status, **kwargs})


def dry_run(cfg, mode):
    checks = {}
    if mode in ("all", "data", "index", "objects"):
        root = Path(cfg["nuscenes"]["dataroot"]) / cfg["nuscenes"]["version"]
        checks["nuscenes_tables"] = {n: (root / (n + ".json")).is_file() for n in ("scene", "sample", "sample_data", "sensor", "calibrated_sensor", "sample_annotation")}
    if mode in ("all", "profile"):
        from data_tools.senna_inputs import check_model, validate_sample
        import torch
        try:
            check_model(cfg["senna"]["checkpoint_path"])
            records = json.loads(Path(cfg["senna"]["eval_json_path"]).read_text())
            records = [s for s in records if "SPEED plan" in s["conversations"][0]["value"]]
            for s in records[:cfg["experiment"]["max_profile_samples"]]:
                validate_sample(s)
            checks["senna_inputs"] = bool(records)
        except (OSError, ValueError, KeyError) as exc:
            checks["senna_inputs"] = False
            checks["senna_error"] = str(exc)
        checks["cuda_available"] = torch.cuda.is_available()
    checks["resolved_config"] = cfg
    return checks
