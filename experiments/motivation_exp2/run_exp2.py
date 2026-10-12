"""Unified CPU entry, provenance-checked stage reuse and independent artifacts."""
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.motivation_exp2"

from .utils.data_utils import BASE, LOG, STAGES, configuration, digest, initialize, load_nuscenes, parser, write_json

MODULES = dict(zip(STAGES, ["build_keyframe_pairs", "extract_projected_annotations", "analyze_physical_persistence",
            "analyze_image_displacement", "analyze_grid_overlap", "analyze_ego_motion", "aggregate_results", "plot_results", "generate_report"]))
PRODUCTS = {"build-pairs": ["data/keyframe_pairs.csv", "data/camera_frames.csv", "data/global_annotations.csv", "data/ego_pairs.csv"],
            "project-boxes": ["data/projected_objects.csv", "metadata/projection_regression.json"],
            "persistence": ["data/global_object_persistence.csv", "data/camera_object_persistence.csv"],
            "displacement": ["data/image_displacement.csv"], "grid-overlap": ["data/fixed_grid_overlap.csv"],
            "ego-motion": ["data/ego_motion_displacement.csv", "data/grid_ego_motion.csv"], "aggregate": ["statistics/summary.json"],
            "plot": ["metadata/figure_metadata.json"] + [f"{folder}/{name}.{suffix}" for folder,suffix in [("figures","png"),("figure_pdf","pdf")]
                     for name in ["M2a_physical_persistence","M2b_image_displacement","M2c_ego_motion_displacement","M2d_grid_overlap","M2e_qualitative_examples"]],
            "report": ["experiment_report.md"]}


def dry_run(cfg):
    required = ["scene", "sample", "sample_data", "sample_annotation", "ego_pose", "calibrated_sensor", "sensor", "instance", "category", "visibility", "attribute", "log", "map"]
    root = Path(cfg["nuscenes"]["dataroot"]) / cfg["nuscenes"]["version"]
    checks = {"tables": {name: (root / f"{name}.json").is_file() for name in required},
              "input_manifests": {name: not cfg["dataset"][name] or Path(cfg["dataset"][name]).is_file() for name in ["scene_manifest", "frame_manifest"]},
              "resolved_config": cfg, "requires_cuda": False, "requires_model": False}
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not all(checks["tables"].values()) or not all(checks["input_manifests"].values()):
        raise FileNotFoundError("Dry-run found missing nuScenes metadata or configured E1 manifests; inspect checks above")


def main(default_stage=None):
    args = parser(default_stage).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    cfg = configuration(args)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/senna_exp2_matplotlib")
    if args.dry_run:
        dry_run(cfg)
        return
    if args.run_all and args.stage:
        raise ValueError("Choose --run-all or --stage")
    out = initialize(cfg, args)
    stages = [args.stage] if args.stage else STAGES
    nusc = None
    for stage in stages:
        status = out / "metadata" / f"{stage}_status.json"
        saved = json.loads(status.read_text()) if status.exists() else {}
        intact = all((out/p).is_file() and saved.get("product_sha256", {}).get(p) == digest(out/p) for p in PRODUCTS[stage])
        if args.existing == "reuse" and saved.get("status") == "complete" and intact:
            LOG.info("Reuse completed stage: %s", stage)
            continue
        # Invalidate downstream status on any upstream rewrite, including missing/corrupt products.
        for downstream in STAGES[STAGES.index(stage):]:
            (out / "metadata" / f"{downstream}_status.json").unlink(missing_ok=True)
        started = time.monotonic()
        write_json(status, {"status": "running"})
        LOG.info("Starting stage: %s", stage)
        try:
            module = importlib.import_module(f"experiments.motivation_exp2.{MODULES[stage]}")
            if stage in ["build-pairs", "project-boxes"]:
                nusc = nusc or load_nuscenes(cfg)
                module.run(cfg, out, nusc)
            else:
                nusc = None
                module.run(cfg, out)
            write_json(status, {"status": "complete", "seconds": time.monotonic()-started, "products": PRODUCTS[stage],
                       "product_sha256": {p: digest(out/p) for p in PRODUCTS[stage]}})
        except Exception as exc:
            write_json(status, {"status": "failed", "error": str(exc), "seconds": time.monotonic()-started})
            raise
        LOG.info("Completed %s in %.1f s", stage, time.monotonic()-started)
    LOG.info("Results: %s", out)


if __name__ == "__main__":
    main()
