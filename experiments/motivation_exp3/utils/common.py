"""CLI, atomic files and strict provenance; reuse the E1/E2 writers."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/senna_exp3_matplotlib")

from experiments.motivation_exp1.utils.data_utils import ROOT, CAMERAS, digest, write_json

BASE = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("motivation_exp3")
STAGES = ["build-pairs", "extract-features", "map-patches", "object-features", "similarity",
          "layer-analysis", "temporal-analysis", "static-dynamic", "aggregate", "plot", "report"]


def parser(default_stage=None):
    p = argparse.ArgumentParser(description="E3: official-instance / original Senna representation diagnostics")
    p.add_argument("--config", default=str(BASE / "config.yaml"))
    p.add_argument("--stage", choices=STAGES, default=default_stage)
    for key in ["run-all", "dry-run", "smoke-test"]:
        p.add_argument("--"+key, action="store_true")
    p.add_argument("--existing", choices=["resume", "overwrite"], default="resume")
    for key in ["checkpoint-path", "device", "dataroot", "exp2-output-dir", "output-dir"]:
        p.add_argument("--"+key)
    for key in ["max-scenes", "max-object-pairs", "batch-size", "seed", "bootstrap-iterations"]:
        p.add_argument("--"+key, type=int)
    p.add_argument("--feature-layers", type=int, nargs="+")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=YAML")
    return p


def resolve(value):
    p = Path(os.path.expandvars(str(value))).expanduser()
    return str((ROOT/p).resolve() if not p.is_absolute() else p.resolve())


def configuration(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    locations = {"checkpoint_path": ("senna", "checkpoint_path"), "device": ("senna", "device"),
                 "batch_size": ("senna", "batch_size"), "feature_layers": ("senna", "feature_layers"),
                 "dataroot": ("data", "nuscenes_dataroot"), "exp2_output_dir": ("data", "exp2_output_dir"),
                 "max_scenes": ("data", "max_scenes"), "max_object_pairs": ("data", "max_object_pairs"),
                 "output_dir": ("output", "directory"), "seed": ("analysis", "seed"),
                 "bootstrap_iterations": ("analysis", "bootstrap_iterations")}
    for key, (section, field) in locations.items():
        if getattr(args, key, None) is not None:
            cfg[section][field] = getattr(args, key)
    for item in args.set:
        field, value = item.split("=", 1)
        section, key = field.split(".", 1)
        if section not in cfg or key not in cfg[section]:
            raise ValueError(f"Unknown configuration: {field}")
        cfg[section][key] = yaml.safe_load(value)
    e1 = ROOT / "outputs/motivation_exp1/resolved_config.yaml"
    fallback = yaml.safe_load(e1.read_text()) if e1.exists() else {}
    cfg["senna"]["checkpoint_path"] = (cfg["senna"]["checkpoint_path"] or os.getenv("SENNA_MODEL_PATH", "")
                                          or fallback.get("senna", {}).get("checkpoint_path", ""))
    cfg["data"]["nuscenes_dataroot"] = (cfg["data"]["nuscenes_dataroot"] or os.getenv("NUSCENES_ROOT", "")
                                          or fallback.get("nuscenes", {}).get("dataroot", ""))
    for s, k in [("senna", "checkpoint_path"), ("data", "nuscenes_dataroot"),
                 ("data", "exp2_output_dir"), ("output", "directory")]:
        if cfg[s][k]:
            cfg[s][k] = resolve(cfg[s][k])
    if args.smoke_test:
        cfg["data"]["max_scenes"] = min(2, cfg["data"]["max_scenes"])
        cfg["data"]["max_object_pairs"] = min(100, cfg["data"]["max_object_pairs"])
        cfg["output"]["visualization_samples"] = min(6, cfg["output"]["visualization_samples"])
        if args.output_dir is None and not any(x.startswith("output.directory=") for x in args.set):
            cfg["output"]["directory"] += "_smoke"
    validate(cfg)
    return cfg


def validate(cfg):
    s, d, a, o = [cfg[k] for k in ["senna", "data", "analysis", "output"]]
    if not s["use_original_preprocessing"] or not s["use_inference_mode"] or s["precision"] != "fp16":
        raise ValueError("This implementation requires original preprocessing, inference_mode and original-loader fp16")
    if not o["save_object_features"]:
        raise ValueError("Offline object descriptors must be saved for resumable analysis")
    if any(type(i) is not int for i in s["feature_layers"]) or len(set(s["feature_layers"])) != len(s["feature_layers"]):
        raise ValueError("feature_layers must be distinct integer hidden-state indices")
    if s["equivalence_atol"] < 0 or s["equivalence_rtol"] < 0:
        raise ValueError("Numerical tolerances must be nonnegative")
    for v in [d["max_scenes"], d["max_object_pairs"], d["min_valid_patch_count"], s["batch_size"],
              s["validation_images"], a["bootstrap_iterations"], d["min_bbox_width_px"], d["min_bbox_height_px"]]:
        if v < 1:
            raise ValueError("Sample, batch, validation, size and bootstrap limits must be positive")
    for key in ["min_projection_quality", "min_retained_area_ratio", "min_patch_coverage_ratio"]:
        if not 0 <= d[key] <= 1:
            raise ValueError(f"Invalid ratio: {key}")
    if (not a["time_gap_groups"] or any(x <= 0 for x in a["time_gap_groups"])
            or a["time_gap_tolerance_seconds"] <= 0 or a["static_speed_threshold_mps"] < 0
            or any(x < 0 for x in a["static_speed_sensitivity_mps"])
            or any(x < 1 for x in a["patch_count_sensitivity"])
            or min(a["negative_max_area_ratio"], a["negative_max_depth_ratio"]) < 1
            or len(a["fixed_position_grid"]) != 2 or min(a["fixed_position_grid"]) < 1):
        raise ValueError("Invalid analysis thresholds")
    if not a["compare_same_instance"]:
        raise ValueError("The primary same-instance comparison cannot be disabled")
    if o["visualization_samples"] < 0 or o["raw_feature_max_images"] < 0 or o["raw_feature_budget_mb"] < 0:
        raise ValueError("Storage and visualization limits must be nonnegative")
    out, inp = Path(o["directory"]), Path(d["exp2_output_dir"])
    protected = [ROOT / "experiments", inp, ROOT / "outputs/motivation_exp1"]
    if out == ROOT or any(out == p or out in p.parents or p in out.parents for p in protected):
        raise ValueError("Use a dedicated E3 output directory outside source and E1/E2 outputs")


def read(out, name):
    p = Path(out)/"data"/(name+".csv")
    if not p.exists():
        raise FileNotFoundError(f"Missing prerequisite {p}; run its stage first")
    # Gap labels must remain strings even when a file contains only numeric labels,
    # or when pandas chunks see 'other' in only part of a large CSV.
    return pd.read_csv(p, low_memory=False, dtype={"time_gap_group": str})


def save(out, name, frame):
    p = Path(out)/"data"/(name+".csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(p)


def json_read(p, default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default


def key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def initialize(cfg, overwrite=False):
    out = Path(cfg["output"]["directory"])
    inp = Path(cfg["data"]["exp2_output_dir"])
    cp = Path(cfg["senna"]["checkpoint_path"])
    # Files that determine geometry, identity, model weights or preprocessing invalidate resume.
    paths = list((inp/"data").glob("*.csv")) + list((inp/"metadata").glob("*.json"))
    paths += list(BASE.rglob("*.py")) + list((ROOT/"llava").rglob("*.py"))
    paths += [cp/"config.json", cp/"pytorch_model.bin.index.json"]
    paths += [ROOT/"experiments/motivation_exp1/utils/data_utils.py",
              ROOT/"experiments/motivation_exp1/utils/stats_utils.py",
              ROOT/"experiments/motivation_exp2/utils/statistics_utils.py"]
    model_cfg = json_read(cp/"config.json", {})
    vision = Path(model_cfg.get("mm_vision_tower", ""))
    paths += [vision/"config.json", vision/"preprocessor_config.json"]
    identity = {"config": cfg, "sha256": {str(p): digest(p) for p in paths if p.is_file() and "tests" not in p.parts},
                "weight_files": {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                                 for folder in [cp, vision] for p in folder.glob("*")
                                 if p.suffix in {".bin", ".safetensors"}},
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    previous = out/"metadata/run_metadata.json"
    old = json_read(previous, {})
    if old and (old["fingerprint"] != fingerprint or overwrite):
        if not overwrite:
            raise ValueError("Config/source/input/model changed. Use another --output-dir or --run-all --existing overwrite")
        archive = out/"previous_runs"/(old["fingerprint"][:16]+"_"+key(os.urandom(16).hex()))
        archive.mkdir(parents=True)
        for name in ["metadata", "data", "features", "figures", "figure_pdf", "visualizations", "statistics", "experiment_report.md"]:
            if (out/name).exists():
                (out/name).rename(archive/name)
    elif out.exists() and not old and any(out.iterdir()):
        raise ValueError("Output directory has no E3 provenance; choose a new directory")
    for folder in ["metadata", "data", "features/object_feature_shards", "figures", "figure_pdf", "visualizations", "statistics"]:
        (out/folder).mkdir(parents=True, exist_ok=True)
    write_json(previous, {**identity, "fingerprint": fingerprint, "data_kind": "real_nuscenes_original_senna"})
    (out/"metadata/experiment_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def standalone(stage):
    from experiments.motivation_exp3.run_exp3 import main
    raise SystemExit(main(default_stage=stage))
