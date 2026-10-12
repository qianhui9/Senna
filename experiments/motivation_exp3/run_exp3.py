"""Unified entry point; partial stages require their real prerequisite artifacts."""
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
import json
import importlib
from experiments.motivation_exp3.utils.common import (
    STAGES, parser, configuration, initialize, write_json, digest, LOG, json_read,
)

MODULES = dict(zip(STAGES, ["build_feature_pairs", "extract_senna_features", "map_bbox_to_patches", "extract_object_features",
    "analyze_feature_similarity", "analyze_layer_stability", "analyze_temporal_drift", "analyze_static_dynamic",
    "aggregate_results", "plot_results", "generate_report"]))
ARTIFACTS = {
    "build-pairs": ["data/feature_object_pairs.csv", "data/object_observations.csv", "metadata/pair_selection.json", "metadata/scene_list.json"],
    "extract-features": ["metadata/extraction_status.json", "metadata/model_metadata.json", "metadata/image_processor_config.json", "metadata/output_equivalence.json"],
    "map-patches": ["data/object_patch_mapping.csv", "metadata/mapping_summary.json"],
    "object-features": ["data/object_feature_index.csv", "metadata/object_feature_summary.json"],
    "similarity": ["data/feature_similarity.csv", "data/control_matches.csv", "data/fixed_position_similarity.csv", "data/global_similarity.csv", "metadata/similarity_summary.json"],
    "layer-analysis": ["data/layer_stability.csv", "metadata/layer_cohort.json"],
    "temporal-analysis": ["data/temporal_stability.csv", "data/observation_condition_stability.csv", "statistics/condition_correlations.json", "data/displacement_stability.csv"],
    "static-dynamic": ["data/static_dynamic_comparison.csv"],
    "aggregate": ["statistics/summary.json", "data/aggregate_statistics.csv", "data/identity_margins.csv", "data/identity_margin_statistics.csv", "data/patch_sensitivity.csv", "data/area_sensitivity.csv", "data/fixed_position_stability.csv", "data/global_stability.csv"],
    "plot": ["metadata/figure_manifest.json"], "report": ["experiment_report.md"],
}
DEPENDENCIES = {"extract-features": ["build-pairs"], "map-patches": ["build-pairs"],
    "object-features": ["extract-features", "map-patches"], "similarity": ["object-features"],
    "layer-analysis": ["similarity"], "temporal-analysis": ["similarity"], "static-dynamic": ["similarity"],
    "aggregate": ["layer-analysis", "temporal-analysis", "static-dynamic"],
    "plot": ["aggregate"], "report": ["plot"]}


def artifacts_for(stage, out):
    paths = [out/p for p in ARTIFACTS[stage]]
    if stage == "extract-features":
        status = json_read(out/"metadata/extraction_status.json", {})
        paths += [out/"features/object_feature_shards"/(token+".npz") for token in status.get("shards", {})]
    if stage == "plot":
        paths += [out/p for p in json_read(out/"metadata/figure_manifest.json", {}).get("artifacts", [])]
    return paths


def completed(stage, out, state, memo=None):
    memo = {} if memo is None else memo
    if stage in memo:
        return memo[stage]
    expected = state.get(stage, {}).get("artifacts", {})
    memo[stage] = (bool(expected) and all(Path(p).is_file() and digest(p) == sha for p, sha in expected.items())
                   and all(completed(dep, out, state, memo) for dep in DEPENDENCIES.get(stage, [])))
    return memo[stage]


def main(default_stage=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    args = parser(default_stage).parse_args()
    try:
        cfg = configuration(args)
        from experiments.motivation_exp3.extract_senna_features import inspect_model
        metadata, processor = inspect_model(cfg)
        if args.dry_run:
            inp = Path(cfg["data"]["exp2_output_dir"])/"data"
            required = ["projected_objects.csv", "camera_frames.csv", "keyframe_pairs.csv"]
            if not all((inp/p).is_file() for p in required):
                raise FileNotFoundError(f"Missing E2 input files in {inp}")
            if not any((inp/p).exists() for p in ["image_displacement.csv", "ego_motion_displacement.csv"]):
                LOG.info("E2 displacement will be regenerated with its original compute_rows")
            print(json.dumps({"dry_run": True, "config": cfg, "model_inspection": metadata,
                              "processor": processor.to_dict(), "stages": STAGES,
                              "gpu_not_required_for_dry_run": True}, indent=2))
            return 0
        if args.existing == "overwrite" and args.stage:
            raise ValueError("Overwrite requires --run-all; stage changes cannot invalidate unrelated results")
        out = initialize(cfg, overwrite=args.existing == "overwrite")
        stages = [args.stage] if args.stage else STAGES
        state_path = out/"metadata/stages.json"
        state = json_read(state_path, {})
        verification = {}
        for stage in stages:
            missing = [dep for dep in DEPENDENCIES.get(stage, []) if not completed(dep, out, state, verification)]
            if missing:
                raise RuntimeError(f"Stage {stage} requires verified stages {missing}; run them first or use --run-all")
            if completed(stage, out, state, verification):
                LOG.info("Verified completed stage: %s", stage)
                continue
            LOG.info("Running stage: %s", stage)
            module = importlib.import_module("experiments.motivation_exp3."+MODULES[stage])
            module.run(cfg, out)
            paths = artifacts_for(stage, out)
            if any(not p.is_file() for p in paths):
                raise RuntimeError(f"Stage {stage} did not produce all required artifacts")
            # Recomputing a prerequisite invalidates downstream stage markers.
            for dependent in STAGES[STAGES.index(stage)+1:]:
                state.pop(dependent, None)
            state[stage] = {"status": "complete", "artifacts": {str(p): digest(p) for p in paths}}
            write_json(state_path, state)
            verification.clear()
        LOG.info("Completed requested stages. Output: %s", out)
        return 0
    except Exception as exc:
        LOG.exception("Experiment stopped: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
