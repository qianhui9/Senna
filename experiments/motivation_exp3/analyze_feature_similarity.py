"""Four predeclared comparison groups, exact-frame matched negatives and FP32 metrics."""
from functools import lru_cache
import json
import numpy as np
import pandas as pd

from .extract_senna_features import load_shard
from .utils.common import read, save, write_json, standalone
from .utils.statistics import temporal_group


def feature_metrics(a, b, epsilon=1e-8):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    invalid = {"cosine_similarity": np.nan, "cosine_distance": np.nan, "relative_l2_error": np.nan}
    if a.shape != b.shape:
        raise ValueError("Feature shapes differ")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {**invalid, "numeric_status": "nonfinite"}
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if min(na, nb) <= epsilon:
        return {**invalid, "numeric_status": "near_zero_norm"}
    cosine = float(np.clip(np.dot(a, b)/(na*nb), -1, 1))
    return {"cosine_similarity": cosine, "cosine_distance": 1-cosine,
            "relative_l2_error": float(np.linalg.norm(b-a)/(nb+epsilon)), "numeric_status": "valid"}


def choose_negative(anchor, target, candidates, same_category, cfg, rng):
    """Control scene/camera/time exactly; constrain target area and reliable depth."""
    cand = candidates[(candidates.instance_token != anchor["instance_token"]) &
                      ((candidates.category_name == anchor["category_name"]) if same_category else
                       (candidates.category_name != anchor["category_name"]))].copy()
    cand = cand[(cand.scene_token == anchor["scene_token"]) & (cand.camera_channel == anchor["camera_channel"])]
    if not len(cand):
        return None
    ratios = cand.bbox_area/target["bbox_area"]
    max_area = cfg["analysis"]["negative_max_area_ratio"]
    cand = cand[(ratios >= 1/max_area) & (ratios <= max_area)].copy()
    depth = target.get("camera_distance_m", np.nan)
    if np.isfinite(depth) and depth > 0:
        ratios = cand.camera_distance_m/depth
        max_depth = cfg["analysis"]["negative_max_depth_ratio"]
        cand = cand[(ratios >= 1/max_depth) & (ratios <= max_depth)].copy()
    else:
        cand = cand[~np.isfinite(cand.camera_distance_m)]
    if not len(cand):
        return None
    cand = cand.sort_values("observation_id")
    # Seeded random draw among controls passing predeclared geometry constraints.
    return cand.iloc[int(rng.integers(len(cand)))].to_dict()


def run(cfg, out):
    pairs, index, mapping = [read(out, n) for n in ["feature_object_pairs", "object_feature_index", "object_patch_mapping"]]
    obs = index.set_index("observation_id").to_dict("index")
    maps = {(r["observation_id"], r["aggregation"]): r for r in mapping.to_dict("records")}
    eligible = mapping[(mapping.aggregation == "area") & mapping.mapping_valid].observation_id
    candidates = index[index.observation_id.isin(eligible)]
    candidates_by_frame = {t: g for t, g in candidates.groupby("sample_data_token")}
    rng = np.random.default_rng(cfg["analysis"]["seed"])
    rows, controls, missing = [], [], []

    @lru_cache(maxsize=8)
    def shard(token):
        return load_shard(out/"features/object_feature_shards"/(token+".npz"))

    def vector(o, layer, method):
        _, arrays = shard(o["sample_data_token"])
        return arrays[f"{layer}__{method}"][int(o["shard_row"])]

    def append(pair, a, b, group, layer, method):
        ma = maps[(a["observation_id"], method)]
        mb = maps[(b["observation_id"], method)]
        metric = feature_metrics(vector(a, layer, method), vector(b, layer, method))
        if not ma["mapping_valid"] or not mb["mapping_valid"]:
            metric["numeric_status"] = "invalid_patch_mapping"
        row = {**pair, "comparison_group": group, "layer_id": layer, "aggregation": method,
               "comparison_instance_token_t1": b["instance_token"], "comparison_category_t1": b["category_name"],
               "comparison_bbox_t1": b["bbox"],
               "observation_id_t1": b["observation_id"], "patch_count_t0": ma["patch_count"], "patch_count_t1": mb["patch_count"],
               "bbox_area_t0": a["bbox_area"], "bbox_area_t1": b["bbox_area"],
               "comparison_depth_t1_m": b["camera_distance_m"], "retained_area_ratio_t0": ma["retained_area_ratio"],
               "retained_area_ratio_t1": mb["retained_area_ratio"], **metric}
        rows.append(row)

    for pair in pairs.to_dict("records"):
        a, b = [obs.get(pair[f"observation_id_t{t}"]) for t in [0, 1]]
        if a is None or b is None:
            missing.append({"object_pair_id": pair["object_pair_id"], "reason": "missing_object_descriptor"})
            continue
        for t, o in enumerate([a, b]):
            o["observation_id"] = pair[f"observation_id_t{t}"]
            if o["instance_token"] != pair["instance_token"] or o["scene_token"] != pair["scene_token"] or o["camera_channel"] != pair["camera_channel"]:
                raise ValueError("Descriptor identity mismatch")
        layers = sorted(set(json.loads(a["layers"])) & set(json.loads(b["layers"])))
        targets = [("B1_same_instance", b)]
        if cfg["analysis"]["compare_different_instance"] and all(maps[(o["observation_id"], "area")]["mapping_valid"] for o in [a, b]):
            pool = candidates_by_frame.get(b["sample_data_token"], candidates.iloc[:0])
            for group, samecat in [("B2_different_instance_same_category", True), ("B3_different_category", False)]:
                target = choose_negative(a, b, pool, samecat, cfg, rng)
                controls.append({"object_pair_id": pair["object_pair_id"], "comparison_group": group,
                                 "matched": target is not None, "target_observation_id": target["observation_id"] if target else None,
                                 "scene_camera_target_frame_exact": True,
                                 "depth_control": "ratio" if np.isfinite(b["camera_distance_m"]) else "both_missing",
                                 "reason": "matched" if target else "no_control_within_geometry_constraints"})
                if target:
                    targets.append((group, target))
        for group, target in targets:
            for layer in layers:
                for method in ["area", "mean", "center"]:
                    append(pair, a, target, group, layer, method)
    result = pd.DataFrame(rows)
    if not len(result):
        raise ValueError("No object descriptors available")
    result["time_gap_group"] = result.delta_t_seconds.map(lambda x: temporal_group(x, cfg))
    result["minimum_patch_count"] = result[["patch_count_t0", "patch_count_t1"]].min(axis=1)
    result["bbox_scale_ratio"] = result.bbox_area_t1/result.bbox_area_t0
    save(out, "feature_similarity", result)
    save(out, "control_matches", pd.DataFrame(controls, columns=["object_pair_id", "comparison_group", "matched", "target_observation_id", "scene_camera_target_frame_exact", "depth_control", "reason"]))
    fixed_rows, global_rows = [], []
    for pair in pairs.drop_duplicates(["sample_data_token_t0", "sample_data_token_t1"]).to_dict("records"):
        m0, x0 = shard(pair["sample_data_token_t0"])
        m1, x1 = shard(pair["sample_data_token_t1"])
        if m0["fixed_indices"] != m1["fixed_indices"]:
            raise ValueError("Fixed patch indices changed across frames")
        layers = sorted(set(m0["layers"]) & set(m1["layers"]))
        base = {k: pair[k] for k in ["pair_id", "scene_token", "scene_name", "camera_channel", "sample_data_token_t0", "sample_data_token_t1", "timestamp_t0", "timestamp_t1", "delta_t_seconds"]}
        if cfg["analysis"]["compare_fixed_position"]:
            for layer in layers:
                for j, patch in enumerate(m0["fixed_indices"]):
                    if m0["fixed_valid"][j] and m1["fixed_valid"][j]:
                        fixed_rows.append({**base, "comparison_group": "B4_fixed_image_position", "layer_id": layer,
                                           "patch_index": patch, **feature_metrics(x0[f"{layer}__fixed"][j], x1[f"{layer}__fixed"][j])})
        for layer in layers+["adapter_global"]:
            name = f"{layer}__global" if layer != "adapter_global" else layer
            global_rows.append({**base, "layer_id": layer, **feature_metrics(x0[name], x1[name])})
    save(out, "fixed_position_similarity", pd.DataFrame(fixed_rows, columns=list(base)+["comparison_group", "layer_id", "patch_index", "cosine_similarity", "cosine_distance", "relative_l2_error", "numeric_status"]))
    save(out, "global_similarity", pd.DataFrame(global_rows))
    write_json(out/"metadata/similarity_summary.json", {"rows": len(result), "numeric_status_counts": result.numeric_status.value_counts().to_dict(),
        "groups": result.groupby(["comparison_group", "aggregation", "numeric_status"]).size().reset_index(name="n").to_dict("records"),
        "missing_pairs": missing, "controls_matched": sum(c["matched"] for c in controls), "controls_attempted": len(controls),
        "fixed_position_rows": len(fixed_rows), "fixed_position_selection": "Predeclared uniform patch-center sample; excludes padded-content centers. This is an image-coordinate diagnostic.",
        "negative_control": "Same scene, camera and exact target frame; different official instance; target area/depth ratios bounded; controls may be reused across anchors",
        "metrics_precision": "float32", "relative_l2_denominator": "norm(t1) + 1e-8"})


if __name__ == "__main__":
    standalone("similarity")
