"""Actual camera gaps and observation-condition strata; correlations are descriptive."""
import numpy as np
import pandas as pd
from .utils.common import read, save, standalone, write_json
from .utils.statistics import grouped, correlation_summary
from .analyze_layer_stability import common_layer_cohort


def run(cfg, out):
    x = common_layer_cohort(read(out, "feature_similarity"))
    x = x[x.comparison_group == "B1_same_instance"].copy()
    save(out, "temporal_stability", grouped(x, ["layer_id", "aggregation", "time_gap_group", "category_name", "camera_channel"], cfg))
    specs = {"normalized_displacement": [0, .03, .1, .25, np.inf], "ego_translation_m": [0, 1, 3, 10, np.inf],
             "ego_rotation_deg": [0, 1, 3, 10, np.inf], "distance_t0_m": [0, 10, 30, 60, np.inf],
             "bbox_scale_ratio": [0, .5, .8, 1.25, 2, np.inf], "bbox_area_t0": [0, 2000, 10000, 50000, np.inf],
             "projection_quality_t0": [0, .5, .8, 1.000001], "retained_area_ratio_t0": [0, .5, .8, 1.000001]}
    conditions, correlations = [], []
    for variable, bins in specs.items():
        x["condition_bin"] = pd.cut(x[variable], bins, include_lowest=True, right=False).astype(str)
        stats = grouped(x, ["layer_id", "aggregation", "time_gap_group", "category_name", "motion_group", "condition_bin"], cfg)
        stats["variable"] = variable
        conditions.append(stats)
    for (layer, method, category, motion, gap), group in x.groupby(["layer_id", "aggregation", "category_name", "motion_group", "time_gap_group"], observed=True):
        for variable in ["normalized_displacement", "ego_translation_m", "ego_rotation_deg", "ego_yaw_change_deg", "bbox_scale_ratio"]:
            correlations.append({"layer_id": layer, "aggregation": method, "category_name": category, "motion_group": motion,
                "time_gap_group": gap, "variable": variable, **correlation_summary(group[["scene_token", variable, "cosine_distance"]], variable, "cosine_distance", cfg)})
    save(out, "observation_condition_stability", pd.concat(conditions, ignore_index=True))
    write_json(out/"statistics/condition_correlations.json", correlations)
    # Compact unconditional displacement summary for plotting, with separate motion groups.
    x["displacement_bin"] = pd.cut(x.normalized_displacement, specs["normalized_displacement"], right=False).astype(str)
    save(out, "displacement_stability", grouped(x, ["layer_id", "aggregation", "motion_group", "displacement_bin"], cfg))


if __name__ == "__main__":
    standalone("temporal-analysis")
