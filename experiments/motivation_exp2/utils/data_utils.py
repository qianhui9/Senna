"""Reuse E1's official loader, camera protocol, atomic writers and selection."""
import argparse
import hashlib
import json
import logging
import os
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yaml

from experiments.motivation_exp1.utils.data_utils import (
    ROOT, CAMERAS, clean_json, digest, load_nuscenes, selected_scenes,
    write_csv, write_json,
)

LOG = logging.getLogger("motivation_exp2")
BASE = Path(__file__).resolve().parents[1]
STAGES = ["build-pairs", "project-boxes", "persistence", "displacement",
          "grid-overlap", "ego-motion", "aggregate", "plot", "report"]


def parser(default_stage=None):
    p = argparse.ArgumentParser(description="E2: offline nuScenes annotated keyframe analysis (CPU)")
    p.add_argument("--config", default=str(BASE / "config.yaml"))
    p.add_argument("--stage", choices=STAGES, default=default_stage)
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--existing", choices=["reuse", "overwrite"], default="reuse")
    for key in ("dataroot", "version", "output-dir", "scene-manifest", "frame-manifest"):
        p.add_argument("--" + key)
    for key in ("max-scenes", "seed", "bootstrap-iterations", "visualization-samples"):
        p.add_argument("--" + key, type=int)
    p.add_argument("--keyframe-gaps", type=int, nargs="+")
    p.add_argument("--cameras", nargs="+")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=YAML",
                   help="Override any configuration field, including lists and booleans")
    return p


def resolve_path(value):
    p = Path(os.path.expandvars(str(value))).expanduser()
    return str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())


def configuration(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    mapping = {"dataroot": ("nuscenes", "dataroot"), "version": ("nuscenes", "version"),
               "output_dir": ("output", "directory"), "max_scenes": ("dataset", "max_scenes"),
               "scene_manifest": ("dataset", "scene_manifest"), "frame_manifest": ("dataset", "frame_manifest"),
               "cameras": ("dataset", "cameras"), "keyframe_gaps": ("pairs", "keyframe_gaps"),
               "seed": ("analysis", "seed"), "bootstrap_iterations": ("analysis", "bootstrap_iterations"),
               "visualization_samples": ("output", "visualization_samples")}
    for arg, (section, key) in mapping.items():
        if getattr(args, arg, None) is not None:
            cfg[section][key] = getattr(args, arg)
    for override in args.set:
        key, value = override.split("=", 1)
        section, field = key.split(".", 1)
        if section not in cfg or field not in cfg[section]:
            raise ValueError(f"Unknown configuration field: {key}")
        cfg[section][field] = yaml.safe_load(value)
    e1_path = resolve_path(cfg["dataset"]["exp1_config"])
    e1 = yaml.safe_load(Path(e1_path).read_text())
    cfg["nuscenes"]["dataroot"] = cfg["nuscenes"]["dataroot"] or os.environ.get("NUSCENES_ROOT", "")
    if not cfg["nuscenes"]["dataroot"]:
        resolved_e1 = ROOT / e1["output"]["directory"] / "resolved_config.yaml"
        fallback = yaml.safe_load(resolved_e1.read_text()) if resolved_e1.is_file() else e1
        cfg["nuscenes"]["dataroot"] = fallback["nuscenes"]["dataroot"]
    if not cfg["nuscenes"]["dataroot"]:
        raise ValueError("Set --dataroot or NUSCENES_ROOT; E1 has no resolved dataroot")
    for section, field in [("nuscenes", "dataroot"), ("output", "directory"),
                           ("dataset", "exp1_config"), ("dataset", "scene_manifest"), ("dataset", "frame_manifest")]:
        if cfg[section][field]:
            cfg[section][field] = resolve_path(cfg[section][field])
    if args.smoke_test:
        cfg["dataset"]["max_scenes"] = min(2, cfg["dataset"]["max_scenes"])
        cfg["output"]["visualization_samples"] = min(6, cfg["output"]["visualization_samples"])
        if args.output_dir is None and not any(s.startswith("output.directory=") for s in args.set) and not cfg["output"]["directory"].endswith("_smoke"):
            cfg["output"]["directory"] += "_smoke"
    validate(cfg)
    return cfg


def validate(cfg):
    d, p, a, proj = cfg["dataset"], cfg["pairs"], cfg["analysis"], cfg["projection"]
    if p["allow_cross_scene"]:
        raise ValueError("Cross-scene pairs are forbidden by this experiment")
    if d["max_scenes"] < 1 or a["bootstrap_iterations"] < 1:
        raise ValueError("max_scenes and bootstrap_iterations must be positive")
    if not d["cameras"] or len(set(d["cameras"])) != len(d["cameras"]) or set(d["cameras"]) - set(CAMERAS):
        raise ValueError("cameras must be distinct official E1 camera channels")
    if not p["keyframe_gaps"] or len(set(p["keyframe_gaps"])) != len(p["keyframe_gaps"]) or any(type(x) is not int or x < 1 for x in p["keyframe_gaps"]):
        raise ValueError("keyframe_gaps must be distinct positive integers")
    if min(proj["near_plane_m"], proj["min_bbox_width_px"], proj["min_bbox_height_px"]) <= 0:
        raise ValueError("Projection depth and size thresholds must be positive")
    if any(not 0 <= x <= 1 for x in [proj["min_visible_area_ratio"], *proj["visible_area_sensitivity"], a["grid_min_coverage_ratio"]]):
        raise ValueError("Area ratios must lie in [0,1]")
    if a["static_speed_threshold_mps"] < 0 or any(x < 0 for x in a["static_speed_sensitivity_mps"]):
        raise ValueError("Speed thresholds must be nonnegative")
    if any(len(s) != 2 or any(type(x) is not int or x < 1 for x in s) for s in a["grid_shapes"]):
        raise ValueError("grid_shapes must contain positive [rows, columns]")
    if len({tuple(s) for s in a["grid_shapes"]}) != len(a["grid_shapes"]) or a["ego_grid_shape"] not in a["grid_shapes"]:
        raise ValueError("Grid shapes must be distinct and include ego_grid_shape")
    if not set(a["grid_rules"]) <= {"any", "coverage", "center"} or a["primary_grid_rule"] not in a["grid_rules"]:
        raise ValueError("Invalid grid rules")
    if cfg["output"]["visualization_samples"] < 0:
        raise ValueError("visualization_samples cannot be negative")
    for key in ["translation_bins_m", "yaw_bins_deg", "distance_bins_m"]:
        if len(a[key]) < 2 or any(y <= x for x, y in zip(a[key], a[key][1:])):
            raise ValueError(f"{key} must be strictly increasing")
    # Never write E2 output into E1 or into source directories.
    out = Path(cfg["output"]["directory"])
    e1 = yaml.safe_load(Path(d["exp1_config"]).read_text())
    e1_out = Path(resolve_path(e1["output"]["directory"]))
    if out == ROOT or out == e1_out or e1_out in out.parents or out in e1_out.parents or BASE == out or out in BASE.parents:
        raise ValueError("Choose a dedicated E2 output directory outside E1 outputs and source")


def select_scenes(nusc, cfg):
    path = cfg["dataset"]["scene_manifest"]
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Scene manifest missing: {p}; use --set dataset.scene_manifest=\"\" for E1 selection fallback")
        if p.suffix == ".csv":
            names = set(pd.read_csv(p)["scene_name"].dropna())
        else:
            content = json.loads(p.read_text())
            entries = content.get("scenes", content.get("scene_list", [])) if isinstance(content, dict) else content
            names = {r.get("name", r.get("scene_name")) if isinstance(r, dict) else r for r in entries}
        scenes = sorted([s for s in nusc.scene if s["name"] in names], key=lambda s: s["name"])
        if len(scenes) != len(names):
            raise ValueError("Scene manifest names do not all exist in configured nuScenes version")
        return scenes[:cfg["dataset"]["max_scenes"]]
    e1 = yaml.safe_load(Path(cfg["dataset"]["exp1_config"]).read_text())
    e1["senna"]["eval_json_path"] = resolve_path(e1["senna"]["eval_json_path"])
    e1["experiment"].update(max_scenes=cfg["dataset"]["max_scenes"], seed=cfg["analysis"]["seed"])
    return selected_scenes(nusc, e1)


def read_table(out, name):
    p = out / "data" / (name + ".csv")
    if not p.is_file():
        raise FileNotFoundError(f"Required E2 input missing: {p}. Run its prerequisite stage first.")
    return pd.read_csv(p)


def save_table(out, name, rows, fields=None):
    if isinstance(rows, pd.DataFrame):
        fields = list(rows.columns)
        rows = rows.to_dict("records")
    write_csv(out / "data" / (name + ".csv"), rows, fields)


def initialize(cfg, args):
    out = Path(cfg["output"]["directory"])
    identity = {"config": cfg, "source_hashes": {str(p.relative_to(ROOT)): digest(p)
                 for p in sorted(BASE.rglob("*.py")) if "tests" not in p.parts},
                "e1_source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in [
                    ROOT / "experiments/motivation_exp1/utils/data_utils.py",
                    ROOT / "experiments/motivation_exp1/utils/stats_utils.py"]},
                "packages": {p: version(p) for p in ["numpy", "pandas", "scipy", "nuscenes-devkit", "Pillow", "matplotlib", "PyYAML"]}}
    root = Path(cfg["nuscenes"]["dataroot"]) / cfg["nuscenes"]["version"]
    identity["dataset_tables"] = {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in root.glob("*.json")}
    identity["input_hashes"] = {key: digest(cfg["dataset"][key]) for key in ["scene_manifest", "frame_manifest", "exp1_config"] if cfg["dataset"][key] and Path(cfg["dataset"][key]).is_file()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    previous = out / "metadata" / "run_metadata.json"
    if previous.exists() and json.loads(previous.read_text())["fingerprint"] != fingerprint:
        if args.existing != "overwrite" or args.stage:
            raise ValueError("Existing outputs have different config/source/data. Choose a new --output-dir, or recompute all with --run-all --existing overwrite.")
        old = json.loads(previous.read_text())
        if old.get("data_kind") != "real_nuscenes_keyframes":
            raise ValueError("Refusing to replace an output directory without E2 provenance")
        archive = out / "previous_runs" / old["fingerprint"][:16]
        if archive.exists():
            raise ValueError(f"Previous-run archive already exists: {archive}. Use a new output directory.")
        archive.mkdir(parents=True)
        for name in ["metadata", "data", "figures", "figure_pdf", "visualizations", "statistics", "experiment_report.md"]:
            p = out / name
            if p.exists():
                p.rename(archive / name)
        LOG.info("Preserved previous E2 results at %s; recomputing all stages", archive)
    for folder in ["metadata", "data", "figures", "figure_pdf", "visualizations", "statistics"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    write_json(previous, {**identity, "fingerprint": fingerprint, "data_kind": "real_nuscenes_keyframes"})
    (out / "metadata" / "experiment_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def standalone(stage):
    from experiments.motivation_exp2.run_exp2 import main
    main(default_stage=stage)
