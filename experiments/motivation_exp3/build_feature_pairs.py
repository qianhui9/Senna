"""Reuse E2's matches; independently verify both official projected annotations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.motivation_exp2.analyze_image_displacement import BBOX_FIELDS, compute_rows
from .utils.common import CAMERAS, read, save, key, write_json, standalone


def stratified_sample(df, limit, seed):
    """Round-robin shuffled condition strata; save selection distributions."""
    if len(df) <= limit:
        return df.copy()
    x = df.copy()
    x["_area"] = pd.cut(np.minimum(x.bbox_area_t0, x.bbox_area_t1), [0, 2000, 10000, 50000, np.inf]).astype(str)
    x["_disp"] = pd.cut(x.normalized_displacement, [-np.inf, .03, .1, .25, np.inf]).astype(str)
    x["_ego"] = pd.cut(x.ego_translation_m, [-np.inf, 1, 3, 10, np.inf]).astype(str)
    fields = ["scene_name", "camera_channel", "category_name", "gap_steps", "motion_group", "_area", "_disp", "_ego"]
    rng = np.random.default_rng(seed)
    buckets = [rng.permutation(g.index).tolist() for _, g in x.groupby(fields, dropna=False, observed=True, sort=True)]
    order = rng.permutation(len(buckets))
    selected = []
    for depth in range(max(map(len, buckets))):
        for i in order:
            if depth < len(buckets[i]):
                selected.append(buckets[i][depth])
                if len(selected) == limit:
                    return df.loc[selected].copy()
    return df.loc[selected].copy()


def build_pairs(matches, projected, frames, keyframes, cfg):
    """No appearance matching; fail on contradictory official identity metadata."""
    ids = ["scene_token", "sample_token", "camera_channel", "instance_token"]
    if projected.duplicated(ids).any() or frames.duplicated(["scene_token", "sample_token", "camera_channel"]).any():
        raise ValueError("Duplicate E2 annotation/frame identities")
    proj = {tuple(r[k] for k in ids): r for r in projected.to_dict("records")}
    frm = {tuple(r[k] for k in ids[:3]): r for r in frames.to_dict("records")}
    pairs = {r["pair_id"]: r for r in keyframes.to_dict("records")}
    records, excluded = [], []
    seen = set()
    for row in matches.to_dict("records"):
        identity = (row["scene_token"], row["camera_channel"], row["instance_token"],
                    row["sample_data_token_t0"], row["sample_data_token_t1"])
        reason = None
        pair = pairs.get(row["pair_id"])
        if not pair or pair["scene_token"] != row["scene_token"]:
            reason = "cross_scene_or_missing_pair"
        elif any(pair[f"sample_token_t{t}"] != row[f"sample_token_t{t}"] for t in [0, 1]):
            reason = "sample_pair_mismatch"
        elif row["camera_channel"] not in CAMERAS or not isinstance(row["instance_token"], str) or not row["instance_token"]:
            reason = "invalid_camera_or_official_id"
        endpoints = []
        for t in [0, 1]:
            lookup = (row["scene_token"], row[f"sample_token_t{t}"], row["camera_channel"])
            obj, frame = proj.get(lookup+(row["instance_token"],)), frm.get(lookup)
            if obj is None or frame is None or obj["projection_status"] != "valid":
                reason = reason or "missing_valid_projection"
                continue
            if any(obj[k] != frame[k] for k in ["scene_name", "sample_data_token"]) or obj["sample_data_token"] != row[f"sample_data_token_t{t}"] or obj["category_name"] != row["category_name"]:
                raise ValueError("Contradictory official E2 identity/category/frame metadata")
            for field in ["camera_timestamp", "image_width", "image_height"]:
                if field in obj and field in frame and obj[field] != frame[field]:
                    raise ValueError(f"Projected annotation/frame metadata contradicts {field}")
            box = np.array([obj[k] for k in BBOX_FIELDS], float)
            if not np.allclose(box, [row[f"{k}_t{t}"] for k in BBOX_FIELDS], atol=1e-6, rtol=1e-8, equal_nan=True):
                raise ValueError("E2 matched bounding box differs from official projected row")
            if (not np.isfinite(box).all() or box[2]-box[0] < cfg["data"]["min_bbox_width_px"]
                    or box[3]-box[1] < cfg["data"]["min_bbox_height_px"]):
                reason = reason or "invalid_or_small_bbox"
            if not np.isfinite(obj["geometric_in_image_area_ratio"]) or obj["geometric_in_image_area_ratio"] < cfg["data"]["min_projection_quality"]:
                reason = reason or "low_projection_quality"
            image = Path(frame["image_path"])
            if not image.is_absolute() and cfg["data"]["nuscenes_dataroot"]:
                image = Path(cfg["data"]["nuscenes_dataroot"])/image
            if not image.is_file():
                reason = reason or "missing_image"
            endpoints.append((obj, frame, box, str(image.resolve())))
        if len(endpoints) == 2:
            dt = (endpoints[1][1]["camera_timestamp"]-endpoints[0][1]["camera_timestamp"])/1e6
            if not np.isfinite(dt) or dt <= 0 or pair is None or pair["timestamp_t1"] <= pair["timestamp_t0"]:
                reason = reason or "invalid_time_order"
            elif not np.isclose(dt, row["camera_delta_t_seconds"], atol=1e-6):
                raise ValueError("Camera timestamp gap contradicts E2 displacement")
        if identity in seen:
            reason = reason or "duplicate_pair"
        if reason:
            excluded.append({"pair_id": row["pair_id"], "instance_token": row["instance_token"], "reason": reason})
            continue
        seen.add(identity)
        record = {**row, "object_pair_id": key(*identity), "annotation_delta_t_seconds": row["delta_t_seconds"],
                  "delta_t_seconds": dt, "image_displacement": row["normalized_displacement"]}
        for optional in ["object_speed_mps", "ego_translation_m", "ego_rotation_deg", "ego_yaw_change_deg", "distance_t0_m", "distance_t1_m"]:
            record.setdefault(optional, np.nan)
        for t, (obj, frame, box, image) in enumerate(endpoints):
            record.update({f"timestamp_t{t}": int(frame["camera_timestamp"]),
                           f"sample_timestamp_t{t}": int(frame["sample_timestamp"]), f"image_path_t{t}": image,
                           f"bbox_t{t}": json.dumps(box.tolist()), f"observation_id_t{t}": key(obj["sample_data_token"], obj["instance_token"])})
        records.append(record)
    return pd.DataFrame(records), excluded


def run(cfg, out):
    inp = Path(cfg["data"]["exp2_output_dir"])
    projected, frames, pairs = [read(inp, n) for n in ["projected_objects", "camera_frames", "keyframe_pairs"]]
    if (inp/"data/ego_motion_displacement.csv").exists():
        matches = read(inp, "ego_motion_displacement")
    elif (inp/"data/image_displacement.csv").exists():
        matches = read(inp, "image_displacement")
    else:
        matches, _ = compute_rows(pairs, projected[projected.projection_status == "valid"],
                                  {"dataset": {"cameras": list(frames.camera_channel.unique())}, "analysis": cfg["analysis"]})
    if "ego_translation_m" not in matches and (inp/"data/ego_pairs.csv").exists():
        ego = read(inp, "ego_pairs")
        matches = matches.merge(ego[["pair_id", "camera_channel"]+[c for c in ego if c.startswith("ego_")]],
                                on=["pair_id", "camera_channel"], how="left", validate="many_to_one")
    for col in ["ego_translation_m", "ego_rotation_deg", "ego_yaw_change_deg", "object_speed_mps"]:
        if col not in matches:
            matches[col] = np.nan
    scenes = sorted(matches.scene_name.unique())[:cfg["data"]["max_scenes"]]
    subset = matches[matches.scene_name.isin(scenes)]
    valid, excluded = build_pairs(subset, projected, frames, pairs, cfg)
    if not len(valid):
        raise ValueError("No valid real official-instance pairs; see E2 data/thresholds")
    valid["motion_group"] = np.where(valid.object_speed_mps.isna(), "unknown", np.where(
        valid.object_speed_mps <= cfg["analysis"]["static_speed_threshold_mps"], "approximately_static", "moving"))
    selected = stratified_sample(valid, cfg["data"]["max_object_pairs"], cfg["analysis"]["seed"])
    selected = selected.sort_values(["scene_name", "camera_channel", "timestamp_t0", "instance_token"]).reset_index(drop=True)
    save(out, "feature_object_pairs", selected)
    # Include all valid E2 objects in the chosen frames, including nonpersistent negatives.
    tokens = set(selected.sample_data_token_t0) | set(selected.sample_data_token_t1)
    objects = projected[(projected.projection_status == "valid") & projected.sample_data_token.isin(tokens)].copy()
    objects = objects[(objects.box_width >= cfg["data"]["min_bbox_width_px"]) &
                      (objects.box_height >= cfg["data"]["min_bbox_height_px"]) &
                      (objects.geometric_in_image_area_ratio >= cfg["data"]["min_projection_quality"])]
    fcols = ["sample_data_token", "image_path"]
    objects = objects.merge(frames[fcols], on="sample_data_token", validate="many_to_one")
    objects["bbox"] = objects[BBOX_FIELDS].apply(lambda r: json.dumps(r.tolist()), axis=1)
    objects["observation_id"] = [key(sd, inst) for sd, inst in zip(objects.sample_data_token, objects.instance_token)]
    objects["image_path"] = objects.image_path.map(lambda x: str((Path(cfg["data"]["nuscenes_dataroot"])/x).resolve()) if not Path(x).is_absolute() else x)
    save(out, "object_observations", objects)
    dimensions = ["scene_name", "camera_channel", "category_name", "gap_steps", "motion_group"]
    for frame in [valid, selected]:
        frame["area_bin"] = pd.cut(np.minimum(frame.bbox_area_t0, frame.bbox_area_t1), [0, 2000, 10000, 50000, np.inf]).astype(str)
        frame["displacement_bin"] = pd.cut(frame.normalized_displacement, [-np.inf, .03, .1, .25, np.inf]).astype(str)
        frame["ego_translation_bin"] = pd.cut(frame.ego_translation_m, [-np.inf, 1, 3, 10, np.inf]).astype(str)
    dimensions += ["area_bin", "displacement_bin", "ego_translation_bin"]
    distributions = {c: {"eligible": valid.groupby(c, dropna=False).size().to_dict(),
                         "selected": selected.groupby(c, dropna=False).size().to_dict()} for c in dimensions}
    write_json(out/"metadata/pair_selection.json", {"e2_matched_rows": len(matches), "within_selected_scenes": len(subset),
        "eligible_pairs": len(valid), "selected_pairs": len(selected), "quality_filtered": len(excluded),
        "quality_filter_fraction": len(excluded)/len(subset), "sampling_omitted": len(valid)-len(selected),
        "unique_instances": selected.instance_token.nunique(), "object_observations_including_controls": len(objects),
        "sampling": "seeded round robin over scene/camera/category/gap/motion/area/displacement/ego strata; descriptive selected cohort, not population estimate",
        "distributions": distributions, "exclusions": excluded,
        "speed_timebase": "sample annotation-center displacement / sample timestamp gap",
        "similarity_timebase": "actual camera exposure timestamps"})
    write_json(out/"metadata/scene_list.json", selected[["scene_name", "scene_token"]].drop_duplicates().to_dict("records"))


if __name__ == "__main__":
    standalone("build-pairs")
