"""Within-category motion comparison and annotation-speed threshold sensitivity."""
import numpy as np
import pandas as pd
from .utils.common import read, save, standalone
from .utils.statistics import grouped
from .analyze_layer_stability import common_layer_cohort


def run(cfg, out):
    x = common_layer_cohort(read(out, "feature_similarity"))
    x = x[x.comparison_group == "B1_same_instance"].copy()
    rows = []
    for threshold in sorted(set(cfg["analysis"]["static_speed_sensitivity_mps"]+[cfg["analysis"]["static_speed_threshold_mps"]])):
        x["motion_group"] = np.where(~np.isfinite(x.object_speed_mps), "unknown", np.where(
            x.object_speed_mps <= threshold, "approximately_static", "moving"))
        table = grouped(x, ["layer_id", "aggregation", "category_name", "motion_group", "time_gap_group"], cfg)
        table["static_speed_threshold_mps"] = threshold
        rows.append(table)
    save(out, "static_dynamic_comparison", pd.concat(rows, ignore_index=True))


if __name__ == "__main__":
    standalone("static-dynamic")
