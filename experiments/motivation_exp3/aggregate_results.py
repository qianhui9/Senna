"""Matched identity margins, overlap, sensitivity and scene bootstrap."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .utils.common import read, save, write_json, json_read, standalone
from .utils.statistics import grouped
from .analyze_layer_stability import common_layer_cohort


def matched_margins(data):
    keys = ["object_pair_id", "layer_id", "aggregation"]
    a = data[data.comparison_group == "B1_same_instance"]
    rows = []
    for group in ["B2_different_instance_same_category", "B3_different_category"]:
        b = data[data.comparison_group == group]
        match = a.merge(b[keys+["cosine_similarity"]], on=keys, suffixes=("", "_control"), validate="one_to_one")
        match["delta_similarity"] = match.cosine_similarity-match.cosine_similarity_control
        match["control_group"] = group
        rows.append(match)
    return pd.concat(rows, ignore_index=True)


def overlap_and_auc(same, different):
    a, b = np.asarray(same, float), np.asarray(different, float)
    if not len(a) or not len(b):
        return {"histogram_overlap": None, "descriptive_auc": None, "n": 0}
    bins = np.linspace(-1, 1, 81)
    ha, hb = np.histogram(a, bins)[0]/len(a), np.histogram(b, bins)[0]/len(b)
    ranks = rankdata(np.r_[a, b])
    auc = (ranks[:len(a)].sum()-len(a)*(len(a)+1)/2)/(len(a)*len(b))
    return {"histogram_overlap": float(np.minimum(ha, hb).sum()), "descriptive_auc": float(auc), "n": len(a),
            "note": "Matched anchor cohort; overlap uses 80 fixed bins [-1,1]; AUC descriptive only, no classifier or IID significance"}


def run(cfg, out):
    raw = read(out, "feature_similarity")
    data = common_layer_cohort(raw)
    table = grouped(data, ["layer_id", "aggregation", "comparison_group", "time_gap_group", "category_name"], cfg)
    save(out, "aggregate_statistics", table)
    margins = matched_margins(data)
    save(out, "identity_margins", margins)
    margin_summary = grouped(margins, ["layer_id", "aggregation", "control_group", "time_gap_group", "category_name", "camera_channel"], cfg, ["delta_similarity"])
    save(out, "identity_margin_statistics", margin_summary)
    pooled_margins = grouped(margins, ["layer_id", "aggregation", "control_group"], cfg, ["delta_similarity"])
    sensitivity = []
    primary = data[data.comparison_group == "B1_same_instance"].copy()
    for minimum in cfg["analysis"]["patch_count_sensitivity"]:
        frame = grouped(primary[primary.minimum_patch_count >= minimum], ["layer_id", "aggregation"], cfg)
        frame["minimum_patch_count"] = minimum
        frame["available_primary_rows"] = len(primary)
        sensitivity.append(frame)
    save(out, "patch_sensitivity", pd.concat(sensitivity, ignore_index=True))
    primary["area_bin"] = pd.cut(np.minimum(primary.bbox_area_t0, primary.bbox_area_t1), [0, 2000, 10000, 50000, np.inf], right=False).astype(str)
    save(out, "area_sensitivity", grouped(primary, ["layer_id", "aggregation", "area_bin"], cfg))
    fixed = read(out, "fixed_position_similarity")
    fixed = fixed[fixed.numeric_status == "valid"]
    # Give every frame pair equal weight within a scene, not every sampled patch.
    fixed_pair = fixed.groupby(["pair_id", "scene_token", "scene_name", "camera_channel", "layer_id"], as_index=False)[["cosine_similarity", "cosine_distance", "relative_l2_error"]].mean()
    fixed_stats = grouped(fixed_pair, ["layer_id"], cfg)
    save(out, "fixed_position_stability", fixed_stats)
    global_data = read(out, "global_similarity")
    global_stats = grouped(global_data[global_data.numeric_status == "valid"], ["layer_id"], cfg)
    save(out, "global_stability", global_stats)
    overlaps = []
    for (layer, method, control), group in margins.groupby(["layer_id", "aggregation", "control_group"], observed=True):
        overlaps.append({"layer_id": layer, "aggregation": method, "control_group": control,
                         **overlap_and_auc(group.cosine_similarity, group.cosine_similarity_control)})
    selected = read(out, "feature_object_pairs")
    summary = {"data_kind": "real_nuscenes_original_senna", "selected_pairs": len(selected),
               "unique_instances": selected.instance_token.nunique(), "scenes": selected.scene_token.nunique(),
               "cameras": sorted(selected.camera_channel.unique()), "actual_gap_seconds": {
                   "min": selected.delta_t_seconds.min(), "max": selected.delta_t_seconds.max(),
                   "median": selected.delta_t_seconds.median()},
               "distributions": grouped(data, ["layer_id", "aggregation", "comparison_group"], cfg).to_dict("records"),
               "identity_margins": pooled_margins.to_dict("records"), "matched_overlap_auc": overlaps,
               "fixed_position": fixed_stats.to_dict("records"), "global": global_stats.to_dict("records"),
               "selection": json_read(out/"metadata/pair_selection.json"), "mapping": json_read(out/"metadata/mapping_summary.json"),
               "numeric": json_read(out/"metadata/similarity_summary.json"),
               "uncertainty": "Bootstrap scene-equal means, not independent objects/frames. A single-scene CI is unavailable. Quantiles and AUC are descriptive. Condition associations are not causal.",
               "hypothesis_status": "Use observed distances together with repeated-input control and matched negative baselines; no universal invariance/caching threshold"}
    write_json(out/"statistics/summary.json", summary)


if __name__ == "__main__":
    standalone("aggregate")
