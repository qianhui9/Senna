"""Automatically report measured evidence, missingness and research boundaries."""
import json
import numpy as np
import pandas as pd

from .utils.common import read, json_read, standalone
from .utils.statistics import grouped


def md_table(df, columns, limit=40):
    x = df[columns].head(limit).copy()
    def fmt(v):
        if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
            return "unavailable"
        if isinstance(v, (float, np.floating)):
            return f"{v:.5g}"
        return str(v).replace("|", "/").replace("\n", " ")
    rows = ["| "+" | ".join(columns)+" |", "| "+" | ".join(["---"]*len(columns))+" |"]
    rows += ["| "+" | ".join(fmt(v) for v in row)+" |" for row in x.itertuples(index=False, name=None)]
    if len(df) > limit:
        rows.append(f"\nShowing {limit}/{len(df)} rows; full values are in the accompanying CSV/JSON.")
    return "\n".join(rows)


def run(cfg, out):
    summary = json_read(out/"statistics/summary.json")
    if not summary:
        raise FileNotFoundError("Run aggregate before report; no results will be invented")
    model = json_read(out/"metadata/model_metadata.json")
    provenance = json_read(out/"metadata/run_metadata.json")
    processor = json_read(out/"metadata/image_processor_config.json")
    selection = summary["selection"]
    layer = "layer_"+str(model["selected_layer"])
    dist = pd.DataFrame(summary["distributions"])
    main = dist[(dist.layer_id == layer) & (dist.aggregation == "area")]
    checks = json_read(out/"metadata/output_equivalence.json", [])
    data = read(out, "feature_similarity")
    same = data[(data.layer_id == layer) & (data.aggregation == "area") & (data.comparison_group == "B1_same_instance") & (data.numeric_status == "valid")]
    temporal = grouped(same, ["time_gap_group"], cfg)
    layers = read(out, "layer_stability")
    motion = read(out, "static_dynamic_comparison")
    motion = motion[(motion.layer_id == layer) & (motion.aggregation == "area") & (motion.metric == "cosine_distance") & (motion.static_speed_threshold_mps == cfg["analysis"]["static_speed_threshold_mps"])]
    patches = read(out, "patch_sensitivity")
    patches = patches[(patches.layer_id == layer) & (patches.metric == "cosine_distance")]
    all_repeats = [r for check in checks for r in check.get("repeated_object_descriptors", []) if r["layer_id"] == layer and r["numeric_status"] == "valid"]
    repeat_distances = np.array([r["cosine_distance"] for r in all_repeats])
    max_repeat = float(repeat_distances.max()) if len(repeat_distances) else None
    evidence = "No valid selected-layer same-instance descriptors; the hypothesis remains unassessed."
    if len(same):
        evidence = (f"In the selected cohort, {len(same)} valid same-instance pairs at {layer} have cosine-distance "
                    f"median {same.cosine_distance.median():.5g}, P95 {same.cosine_distance.quantile(.95):.5g}, "
                    f"and relative-L2 median {same.relative_l2_error.median():.5g}.")
        if max_repeat is not None:
            fraction = float((same.cosine_distance > max_repeat+1e-6).mean())
            evidence += (f" The maximum repeated-input object-descriptor cosine distance is {max_repeat:.5g} "
                         f"({len(all_repeats)} controls); {fraction:.1%} of same-instance pairs exceed that empirical numerical envelope plus 1e-6.")
            if fraction > 0:
                evidence += " These measurements support non-invariance of the sampled bbox-pooled intermediate representations despite official identity persistence. They do not establish a universal or task-relevant error threshold."
            else:
                evidence += " These measurements do not show variation beyond the repeated-input numerical envelope."
        else:
            evidence += " Repeated-input descriptor measurements are unavailable, so variation beyond numerical noise is not established."
    conditions = json_read(out/"statistics/condition_correlations.json", [])
    corrs = pd.DataFrame([r for r in conditions if r["layer_id"] == layer and r["aggregation"] == "area"])
    shapes = pd.DataFrame([{ "layer": k, "shape": str(v["shape"]), "dtype": v["dtype"], "spatial_mapping": v["spatial_mapping"]} for k,v in model["runtime_shapes"].items()])
    count_table = pd.DataFrame(summary["mapping"]["counts_by_method_and_status"])
    figure = json_read(out/"metadata/figure_manifest.json", {})
    margins = pd.DataFrame(summary["identity_margins"])
    margins = margins[(margins.layer_id == layer) & (margins.aggregation == "area")]
    rows = ["# Experiment 3: Physical Identity vs. Representation Stability", "",
        "## Research question and scope", "",
        "Test whether official physical identity implies invariant intermediate visual representations across continuous camera observations. Q1–Q5 cover same-ID variation, observation conditions, CLIP layers, approximate motion strata and matched different-ID baselines. All images undergo original full-image visual forwards. This is a motivation observation, with no WorldStream implementation or task-accuracy evaluation.",
        "", "## Reproducibility and original model", "",
        f"- Senna Git commit: `{provenance['git_commit']}`; source/input hashes and weight file identities: `metadata/run_metadata.json`.",
        f"- Checkpoint: `{model['checkpoint']}`; loaded class `{model['loaded_model_class']}`.",
        f"- Tower: `{model['vision_tower_class']}`, `{model['vision_tower_name']}`; {model['vision_config']['num_hidden_layers']} transformer blocks, hidden dimension {model['vision_config']['hidden_size']}, patch size {model['vision_config']['patch_size']}, image size {model['vision_config']['image_size']}.",
        f"- Actual device/precision: `{model['actual_device']}` / `{model['actual_vision_dtype']}`; GPU `{model['gpu']}`.",
        f"- Selected feature `{model['selected_feature']}`, configured layer {model['model_config']['mm_vision_select_layer']}, resolved hidden-state index {model['selected_layer']}.",
        "- Camera order: FRONT, FRONT LEFT, FRONT RIGHT, BACK LEFT, BACK RIGHT, BACK. Each camera is encoded independently before original per-view token compression.",
        "", "```text", model["projector_structure"], model["img_adapter_structure"], "```", "",
        model["adapter_note"], "Projector descriptors are collected before the img_adapter. The adapter output is summarized globally in `data/global_stability.csv`; it never receives bounding-box patch indices.",
        "", "## Original image preprocessing and exact spatial mapping", "",
        f"Model aspect-ratio strategy: `{model['model_config'].get('image_aspect_ratio')}`. Original `llava.mm_utils.process_images` performs mean-color square padding, processor resize, center crop, rescale and normalization. Processor configuration:",
        "", "```json", json.dumps(processor, indent=2), "```", "",
        "Every image records its original dimensions, integer padding/crop offsets and rounded resize dimensions in its feature shard and `object_patch_mapping.csv`. Runtime input dimensions are compared with the recorded transform. Boxes are clipped to the actual input region; completely removed, low-retention and insufficient-patch objects are excluded and counted.",
        f"Area pooling requires intersection/patch area ≥ {cfg['data']['min_patch_coverage_ratio']}, at least {cfg['data']['min_valid_patch_count']} patches, retained bbox area ≥ {cfg['data']['min_retained_area_ratio']}. Row-major patch indices exclude CLS. Mean pooling uses the same qualifying patches; center pooling uses contained centers. Grid {model['grid_shape']} is derived from the actual vision configuration and validated against tensor token counts.",
        "", "## Layer indexing and actual output shapes", "", model["layer_zero_note"], "",
        md_table(shapes, ["layer", "shape", "dtype", "spatial_mapping"]), "",
        "Configured early/middle/late/terminal hidden states and the resolved Senna-selected state are extracted with `output_hidden_states=True` through transparent forward hooks. Hooks return None. All-layer comparisons use an identical valid comparison cohort, separately by aggregation method and control group.",
        "", "## Forward correctness and numerical control", "",
        f"{len(checks)} batches compared original and instrumented Vision Tower + compressed visual outputs; all passed: {all(c['passed'] for c in checks)}. Absolute tolerance {cfg['senna']['equivalence_atol']}, relative tolerance {cfg['senna']['equivalence_rtol']}. Full shape/dtype/max-error checks and repeated object-descriptor metrics: `metadata/output_equivalence.json`.",
        f"Repeated-input selected-layer descriptor controls: n={len(all_repeats)}, maximum cosine distance={max_repeat}. Aggregation and distances use FP32; saved descriptors use FP16. Nonfinite and near-zero norms are explicitly labeled, not replaced with zero.",
        "", "## Official identity pairs, selection and missingness", "",
        f"E2 matched records: {selection['e2_matched_rows']}; selected-scene candidates {selection['within_selected_scenes']}; quality-filtered {selection['quality_filtered']} ({selection['quality_filter_fraction']:.2%}); eligible {selection['eligible_pairs']}; selected {selection['selected_pairs']}; sampling omissions {selection['sampling_omitted']}; unique physical instances {summary['unique_instances']}.",
        f"Actual scenes {summary['scenes']}, cameras {summary['cameras']}, camera gaps {summary['actual_gap_seconds']}. Scene list: `metadata/scene_list.json`.",
        selection["sampling"], "Per-scene, camera, category, gap and motion counts before/after selection and all quality exclusions are saved in `metadata/pair_selection.json`. Sample selection is not representative of the whole nuScenes population.",
        "Identity is checked against E2 projected official annotations at both endpoints with exact scene/camera/instance/sample-data tokens, category, bbox and actual image existence. Similarity time uses camera exposures; approximate object speed uses annotation-center displacement over sample timestamp gaps.",
        "", md_table(count_table, ["aggregation", "mapping_status", "n"]),
        "", "## Same identity and matched comparison baselines", "",
        "B1 compares same official instance over time. B2 changes identity within the exact category; B3 changes category. Both use the same anchor, scene, camera and exact target frame, with target area and reliable depth ratios bounded by predeclared constraints. Unmatched anchors are explicitly recorded, and controls can be reused across anchors. B4 compares predeclared fixed spatial patch centers in two frames, excluding padded-content centers; summary first averages sampled patches per frame pair.",
        md_table(main, ["comparison_group", "metric", "n", "mean", "p50", "std", "p25", "p75", "p90", "p95", "scene_equal_weight_mean", "ci95_low", "ci95_high"]),
        "", "Matched same-minus-different similarity margins use the same anchors, not subtraction of unrelated population means:", "",
        md_table(margins, ["control_group", "n", "mean", "p50", "ci95_low", "ci95_high"]),
        "", "Matched-cohort overlap and descriptive AUC (no fitted classifier):", "",
        md_table(pd.DataFrame(summary["matched_overlap_auc"]), ["layer_id", "aggregation", "control_group", "n", "histogram_overlap", "descriptive_auc"]),
        "", "## Actual time-gap distributions", "",
        md_table(temporal, ["time_gap_group", "metric", "n", "mean", "p50", "p95", "ci95_low", "ci95_high"]),
        "Different gaps also involve different viewpoints, projected scales and scene content; this does not isolate a pure temporal-decay effect.",
        "", "## Image displacement, ego motion and observation conditions", "",
        "Normalized displacement is E2 bbox-center Euclidean migration divided by original image diagonal. Displacement, translation, rotation, yaw, depth, area, scale ratio and geometric projection quality are retained. `observation_condition_stability.csv` stratifies category, motion and time; `condition_correlations.json` reports descriptive observation Spearman and scene-median bootstrap intervals.",
        md_table(corrs, ["category_name", "motion_group", "time_gap_group", "variable", "n", "scenes", "observation_spearman", "scene_median_spearman"]),
        "These associations do not identify causal effects of position embeddings, ego motion or elapsed time. Geometric image-clipping ratios are not occlusion rates; official visibility labels are not precise masks.",
        "", "## Across-layer stability", "",
        md_table(layers[(layers.aggregation == "area") & (layers.metric == "cosine_distance")], ["layer_id", "comparison_group", "n", "p50", "mean", "ci95_low", "ci95_high"]),
        "", "## Approximately static and moving objects", "",
        f"Main threshold ≤ {cfg['analysis']['static_speed_threshold_mps']} m/s; sensitivity thresholds {cfg['analysis']['static_speed_sensitivity_mps']}. Nonfinite speed is Unknown. NuScenes annotation-center velocities may contain noise and do not identify unannotated roads/buildings as static instances.",
        md_table(motion, ["category_name", "motion_group", "time_gap_group", "n", "mean", "p50", "ci95_low", "ci95_high"]),
        "Motion comparisons are within category and actual gap groups; remaining scale/depth/viewpoint and annotation differences prevent a causal interpretation.",
        "", "## Object-scale and patch-aggregation sensitivity", "",
        md_table(patches, ["aggregation", "minimum_patch_count", "n", "mean", "p50", "ci95_low", "ci95_high"]),
        "Full area-bin summaries: `data/area_sensitivity.csv`; pooling/patch-count summaries: `data/patch_sensitivity.csv`. Minimum-patch sensitivity only tightens the primary filter; samples rejected by the configured primary patch minimum are not silently reintroduced.",
        "", "## Statistical uncertainty and limitations", "", summary["uncertainty"],
        "Repeated objects, shared frames, cross-camera observations and reused negatives are correlated. The number of scenes limits inference. Axis-aligned projected 3D-box rectangles include background and do not establish unobstructed visibility. CLIP patch states depend on global self-attention. Offline pooled descriptors are not native object tokens or pure object semantics. Conclusions apply to the sampled checkpoint, precision and cohort.",
        "", "## Figures and qualitative examples", "",
        "Qualitative selection uses lower/upper feature-distance quartiles and large/small displacement quartiles, choosing cases nearest each within-condition median; remaining examples are seeded references. Missing conditions are reported rather than manufactured.",
        f"Unavailable qualitative conditions: {figure.get('unavailable_example_conditions', [])}."]
    rows += [f"\n![{p}]({p})" for p in figure.get("artifacts", []) if p.startswith("figures/")]
    rows += ["", "## Reproduction", "", "```bash", "/home/qh/miniconda3/envs/Senna/bin/python experiments/motivation_exp3/run_exp3.py \\",
             "  --config experiments/motivation_exp3/config.yaml --run-all", "```", "",
             "Resolved configuration: `metadata/experiment_config.yaml`; use its paths/limits to reproduce this particular cohort. `--smoke-test` uses a separate output directory, at most 2 scenes and 100 object pairs. Valid offline feature shards are reused only to resume this offline experiment; every distinct image represented in the experiment was originally fully encoded.",
             "", "## A. Observation Supported by Evidence", "", evidence, "",
             "Matched identity margins and overlapping baseline distributions above characterize identity discrimination separately from high absolute cosine similarity. No monotone layer/time/motion effect is presumed.",
             "", "## B. Claims Not Established", "",
             "This experiment does not establish that historical visual features cannot be cached; that a similarity decline implies driving-accuracy loss; or that any feature-error cutoff is an optimal reuse threshold. It does not establish that World Alignment resolves variation, that WorldStream State Validity works, that reuse reduces FLOPs by a particular amount, that incremental execution accelerates a GPU, or that a Deadline Scheduler is necessary. No features were replaced, no historical inference computations were reused, and no model was trained or fine-tuned."]
    (out/"experiment_report.md").write_text("\n\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    standalone("report")
