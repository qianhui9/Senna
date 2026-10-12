"""Distributions and strata with scene-level uncertainty; all inputs are CSV."""
import pandas as pd

from .analyze_image_displacement import compute_rows
from .analyze_physical_persistence import annotation_sets, persistence_rows
from .utils.data_utils import LOG, read_table, save_table, standalone, write_json
from .utils.statistics_utils import correlation_summary, grouped_summary, metric_summary


def run(cfg, out):
    tables = {name: read_table(out, name) for name in ["keyframe_pairs", "global_object_persistence",
              "camera_object_persistence", "projected_objects", "image_displacement", "fixed_grid_overlap",
              "ego_motion_displacement", "ego_pairs", "grid_ego_motion"]}
    glob, cam, disp, grid, motion = [tables[k] for k in ["global_object_persistence", "camera_object_persistence", "image_displacement", "fixed_grid_overlap", "ego_motion_displacement"]]
    a = cfg["analysis"]
    metrics = ["center_displacement_px", "normalized_displacement", "bbox_iou", "bbox_scale_ratio"]
    summary = {"unit_of_replication": "scene; objects observed across cameras are not independent physical replicates",
               "bootstrap_iterations": a["bootstrap_iterations"], "seed": a["seed"],
               "counts": {"keyframe_pairs": len(tables["keyframe_pairs"]), "projection_attempts": len(tables["projected_objects"]),
                  "valid_projections": int((tables["projected_objects"].projection_status == "valid").sum()),
                  "global_persistent_pair_instances": int(glob[glob.category_name == "ALL"].persistent_objects.sum()),
                  "matched_camera_pair_instances": len(disp), "unique_matched_physical_instances": disp.instance_token.nunique(),
                  "matched_scenes": disp.scene_token.nunique(), "grid_diagnostic_rows": len(grid)},
               "projection_status_counts": tables["projected_objects"].projection_status.value_counts().to_dict()}
    summary["global_persistence"] = grouped_summary(glob[glob.category_name == "ALL"], ["gap_steps"], ["persistent_ratio"], cfg)
    summary["camera_persistence"] = grouped_summary(cam[cam.category_name == "ALL"], ["gap_steps", "camera_channel"], ["persistent_ratio"], cfg)
    summary["persistence_by_category"] = grouped_summary(pd.concat([glob, cam]), ["gap_steps", "camera_channel", "category_name"], ["persistent_ratio"], cfg)
    ego_fields = ["pair_id", "camera_channel", "ego_translation_m", "ego_yaw_change_deg"]
    from .analyze_ego_motion import add_motion_bins
    pc = add_motion_bins(cam.merge(tables["ego_pairs"][ego_fields], on=["pair_id", "camera_channel"], validate="many_to_one"), cfg)
    # Global persistence uses an explicit reference-camera pose; it is never six duplicate global counts.
    reference = cfg["dataset"]["cameras"][0]
    eg = tables["ego_pairs"][tables["ego_pairs"].camera_channel == reference].drop(columns="camera_channel")
    pg = glob.merge(eg[["pair_id", "ego_translation_m", "ego_yaw_change_deg"]], on="pair_id", validate="many_to_one")
    pg["motion_reference_camera"] = reference
    pg = add_motion_bins(pg, cfg)
    save_table(out, "persistence_with_motion", pd.concat([pg, pc], ignore_index=True))
    summary["persistence_by_motion"] = grouped_summary(pd.concat([pg[pg.category_name == "ALL"], pc[pc.category_name == "ALL"]]),
                ["gap_steps", "camera_channel", "translation_bin", "yaw_bin"], ["persistent_ratio"], cfg)
    summary["displacement"] = grouped_summary(disp, ["gap_steps"], metrics, cfg)
    summary["displacement_by_camera"] = grouped_summary(disp, ["gap_steps", "camera_channel"], metrics, cfg)
    summary["displacement_by_category"] = grouped_summary(disp, ["gap_steps", "category_name"], metrics, cfg)
    summary["displacement_by_scene"] = grouped_summary(disp, ["scene_token", "scene_name", "gap_steps", "camera_channel"], metrics, cfg)
    summary["static_dynamic"] = grouped_summary(motion, ["gap_steps", "motion_group"], metrics+["object_speed_mps"], cfg)
    summary["grid"] = grouped_summary(grid, ["gap_steps", "grid_rows", "grid_cols", "grid_rule"], ["grid_overlap", "old_grid_retention"], cfg)
    summary["grid_by_camera"] = grouped_summary(grid[grid.grid_rule == a["primary_grid_rule"]], ["gap_steps", "camera_channel", "grid_rows", "grid_cols"], ["grid_overlap", "old_grid_retention"], cfg)
    summary["grid_by_motion_group"] = grouped_summary(grid[grid.grid_rule == a["primary_grid_rule"]], ["gap_steps", "motion_group", "grid_rows", "grid_cols"], ["grid_overlap", "old_grid_retention"], cfg)
    summary["grid_by_category"] = grouped_summary(grid[grid.grid_rule == a["primary_grid_rule"]], ["gap_steps", "category_name", "grid_rows", "grid_cols"], ["grid_overlap", "old_grid_retention"], cfg)
    summary["grid_by_scene"] = grouped_summary(grid[grid.grid_rule == a["primary_grid_rule"]], ["scene_token", "gap_steps", "grid_rows", "grid_cols"], ["grid_overlap", "old_grid_retention"], cfg)
    summary["actual_time_gaps"] = grouped_summary(tables["ego_pairs"], ["gap_steps", "camera_channel"], ["delta_t_seconds", "camera_delta_t_seconds"], cfg)
    summary["correlations"] = []
    for keys, group in motion.groupby(["gap_steps", "motion_group"]):
        for variable in ["ego_translation_m", "ego_rotation_deg", "ego_yaw_change_deg"]:
            summary["correlations"].append({"gap_steps": keys[0], "motion_group": keys[1], "x": variable,
                    "y": "normalized_displacement", **correlation_summary(group, variable, "normalized_displacement", cfg)})
    summary["stratified_correlations"] = []
    for keys, group in motion.groupby(["gap_steps", "camera_channel", "motion_group", "distance_bin"]):
        summary["stratified_correlations"].append(dict(zip(["gap_steps", "camera_channel", "motion_group", "distance_bin"], keys)) |
                     {"x": "ego_translation_m", **correlation_summary(group, "ego_translation_m", "normalized_displacement", cfg)})
    summary["motion_bins"] = {variable: grouped_summary(motion, ["gap_steps", "camera_channel", "motion_group", "distance_bin", variable],
                           ["normalized_displacement", "bbox_iou", "grid_overlap"], cfg) for variable in ["translation_bin", "yaw_bin"]}
    scene_rows = []
    for scene, group in motion.groupby("scene_name"):
        for variable in ["ego_translation_m", "ego_rotation_deg"]:
            scene_rows.append({"scene_name": scene, "variable": variable, **correlation_summary(group, variable, "normalized_displacement", cfg, False)})
    save_table(out, "scene_correlations", pd.DataFrame(scene_rows))
    # Projection-quality sensitivity includes objects rejected only by geometric area ratio.
    LOG.info("Computing projection and object-speed threshold sensitivity")
    projected = tables["projected_objects"]
    candidates = projected[projected.projection_status.isin(["valid", "low_in_image_area_ratio"])]
    sensitivity = []
    for threshold in cfg["projection"]["visible_area_sensitivity"]:
        valid = candidates[candidates.geometric_in_image_area_ratio >= threshold]
        d, _ = compute_rows(tables["keyframe_pairs"], valid, cfg)
        p = pd.DataFrame(persistence_rows(tables["keyframe_pairs"], annotation_sets(valid, True), cfg["dataset"]["cameras"]))
        p = p[p.category_name == "ALL"]
        for gap in sorted(tables["keyframe_pairs"].gap_steps.unique()):
            dg, pgap = d[d.gap_steps == gap], p[p.gap_steps == gap]
            sensitivity.append({"geometric_area_threshold": threshold, "gap_steps": gap, "valid_projection_count": len(valid),
                     "matched_observations": len(dg), "displacement": metric_summary(dg, "normalized_displacement", cfg),
                     "camera_persistence": metric_summary(pgap, "persistent_ratio", cfg)})
    summary["projection_sensitivity"] = sensitivity
    summary["static_speed_sensitivity"] = []
    for threshold in a["static_speed_sensitivity_mps"]:
        temp = motion.copy()
        temp["motion_group"] = temp.object_speed_mps.map(lambda s: "approximately_static" if s <= threshold else "moving")
        stats = grouped_summary(temp, ["gap_steps", "motion_group"], ["normalized_displacement", "bbox_iou", "grid_overlap"], cfg)
        summary["static_speed_sensitivity"].extend({"threshold_mps": threshold, **g} for g in stats)
    # Object-size and distance changes remain possible confounders, even in static strata.
    summary["analysis_limits"] = ["Bounding rectangles are geometric extents, not segmented object pixels or exact visibility",
            "Actual camera poses with sample-level keyframe boxes; exposure time offsets are retained",
            "No causal ego-only decomposition; distance, object motion, clipping and annotation error can contribute",
            "Static annotated objects do not represent roads/buildings or all static background",
            "Cross-camera observations and repeated pairs are dependent; all mean CIs resample scenes",
            "Only configured scenes are analysed; not a random representative sample of all driving conditions"]
    write_json(out / "statistics" / "summary.json", summary)
    flattened = []
    for family in ["global_persistence", "camera_persistence", "displacement", "static_dynamic", "grid", "displacement_by_scene"]:
        for group in summary[family]:
            for metric, stats in group["metrics"].items():
                ci = stats["ci95"] or [None, None]
                flattened.append({"family": family, **{k: v for k, v in group.items() if k != "metrics"}, "metric": metric,
                    **{k: stats[k] for k in ["n", "scenes", "observation_weighted_mean", "scene_equal_weight_mean", "p50", "p75", "p90", "p95"]},
                    "ci95_low": ci[0], "ci95_high": ci[1]})
    save_table(out, "aggregate_statistics", pd.DataFrame(flattened))


if __name__ == "__main__":
    standalone("aggregate")
