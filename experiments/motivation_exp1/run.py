"""Unified experiment CLI. Run from repository root with python -m."""
import gc
import importlib
import json
from pathlib import Path

from .utils.data_utils import LOG, completed, configuration, dry_run, initialize, load_nuscenes, mark, parser, write_json

MODULES = {"index": "build_nuscenes_manifest", "redundancy": "analyze_temporal_redundancy", "objects": "analyze_object_persistence", "profile": "profile_senna", "aggregate": "aggregate_results", "plot": "plot_results", "report": "generate_report"}


def main(default_mode="all"):
    args = parser(default_mode).parse_args()
    cfg = configuration(args)
    if args.dry_run:
        checks = dry_run(cfg, args.mode)
        print(json.dumps(checks, indent=2, ensure_ascii=False))
        bad = checks.get("senna_inputs") is False or checks.get("cuda_available") is False or not all(checks.get("nuscenes_tables", {}).values())
        return 1 if bad else 0
    out = initialize(cfg, args)
    stages = ["index", "objects", "redundancy", "profile", "aggregate", "plot", "report"] if args.mode == "all" else ["index", "objects", "redundancy", "aggregate", "plot", "report"] if args.mode == "data" else [args.mode]
    failed, nusc = [], None
    for stage in stages:
        try:
            if stage == "plot" and not cfg["experiment"]["save_visualizations"]:
                mark(out, "plot", "disabled", reason="save_visualizations=false")
                continue
            module = importlib.import_module("." + MODULES[stage], __package__)
            if stage in ("index", "objects"):
                if not completed(out, stage, args):
                    if nusc is None:
                        nusc = load_nuscenes(cfg)
                    module.run(cfg, args, out, nusc)
            else:
                if nusc is not None:
                    del nusc
                    nusc = None
                    gc.collect()
                module.run(cfg, args, out)
            if stage in ("aggregate", "plot", "report"):
                mark(out, stage)
        except Exception as exc:
            LOG.exception("Stage %s failed", stage)
            mark(out, stage, "failed", reason=f"{type(exc).__name__}: {exc}")
            failed.append(stage)
            # Data failures invalidate downstream data analysis. A GPU failure
            # still permits an honest partial report and unavailable figures.
            if stage in ("index", "objects", "redundancy"):
                if args.mode in ("all", "data"):
                    continue
                raise
            if args.mode not in ("all", "data"):
                raise
    write_json(out / "invocation_status.json", {"failed_stages": failed, "mode": args.mode})
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
