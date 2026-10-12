"""Match only same scene/camera/official instance ID with two valid projections."""
import numpy as np
import pandas as pd

from .utils.annotation_utils import instance_map
from .utils.data_utils import read_table, save_table, standalone, write_json
from .utils.geometry_utils import displacement

BBOX_FIELDS = ["box_xmin", "box_ymin", "box_xmax", "box_ymax"]
DISPLACEMENT_FIELDS = ["pair_id", "scene_token", "scene_name", "sample_token_t0", "sample_token_t1", "gap_steps", "delta_t_seconds",
    "camera_channel", "instance_token", "category_name", "sample_data_token_t0", "sample_data_token_t1", "annotation_token_t0", "annotation_token_t1",
    "image_width", "image_height", "camera_delta_t_seconds", "center_displacement_px", "normalized_displacement", "displacement_x_px",
    "displacement_y_px", "displacement_width_fraction", "displacement_height_fraction", "bbox_iou", "bbox_area_t0", "bbox_area_t1",
    "bbox_scale_ratio", "object_speed_mps", "motion_group", "distance_t0_m", "distance_t1_m", "projection_quality_t0", "projection_quality_t1",
    "annotation_camera_time_offset_ms_t0", "annotation_camera_time_offset_ms_t1"] + [f"{k}_t{t}" for t in [0, 1] for k in BBOX_FIELDS]


def compute_rows(pairs, projected, cfg):
    index = {key: instance_map(group.to_dict("records")) for key, group in projected.groupby(["sample_token", "camera_channel"])}
    rows, issues = [], []
    for pair in pairs.to_dict("records"):
        for camera in cfg["dataset"]["cameras"]:
            a, b = [index.get((pair[k], camera), {}) for k in ["sample_token_t0", "sample_token_t1"]]
            for instance in sorted(a.keys() & b.keys()):
                x, y = a[instance], b[instance]
                if x["scene_token"] != y["scene_token"] or x["scene_token"] != pair["scene_token"]:
                    raise ValueError("Cross-scene projection match")
                if x["category_name"] != y["category_name"]:
                    raise ValueError(f"Instance category changed: {instance}")
                if (x["image_width"], x["image_height"]) != (y["image_width"], y["image_height"]):
                    issues.append({"kind": "pair_image_dimension_mismatch", "pair_id": pair["pair_id"], "camera_channel": camera, "instance_token": instance})
                    continue
                camera_dt = (y["camera_timestamp"]-x["camera_timestamp"])/1e6
                if camera_dt <= 0:
                    raise ValueError("Camera timestamps are not increasing")
                dt = pair["delta_t_seconds"]
                center0, center1 = [np.array([r[k] for k in ["global_x", "global_y", "global_z"]]) for r in [x, y]]
                speed = np.linalg.norm(center1-center0)/dt
                metrics = displacement([x[k] for k in BBOX_FIELDS], [y[k] for k in BBOX_FIELDS], x["image_width"], x["image_height"])
                record = {**{k: pair[k] for k in ["pair_id", "scene_token", "scene_name", "sample_token_t0", "sample_token_t1", "gap_steps", "delta_t_seconds"]},
                          "camera_channel": camera, "instance_token": instance, "category_name": x["category_name"],
                          "image_width": x["image_width"], "image_height": x["image_height"], "camera_delta_t_seconds": camera_dt,
                          **metrics, "bbox_area_t0": x["bbox_area"], "bbox_area_t1": y["bbox_area"],
                          "bbox_scale_ratio": y["bbox_area"]/x["bbox_area"], "object_speed_mps": speed,
                          "motion_group": "approximately_static" if speed <= cfg["analysis"]["static_speed_threshold_mps"] else "moving",
                          "distance_t0_m": x["camera_distance_m"], "distance_t1_m": y["camera_distance_m"],
                          "projection_quality_t0": x["geometric_in_image_area_ratio"], "projection_quality_t1": y["geometric_in_image_area_ratio"]}
                for t, row in enumerate([x, y]):
                    for k in BBOX_FIELDS + ["annotation_token", "sample_data_token", "annotation_camera_time_offset_ms"]:
                        record[f"{k}_t{t}"] = row[k]
                rows.append(record)
    return pd.DataFrame(rows, columns=DISPLACEMENT_FIELDS), issues


def run(cfg, out):
    projected = read_table(out, "projected_objects")
    valid = projected[projected.projection_status == "valid"]
    result, issues = compute_rows(read_table(out, "keyframe_pairs"), valid, cfg)
    save_table(out, "image_displacement", result)
    write_json(out / "metadata" / "displacement_summary.json", {"matched_camera_observations": len(result),
               "unique_instances": result.instance_token.nunique(), "issues": issues,
               "normalization": "Euclidean center displacement divided by actual image diagonal",
               "object_speed_timebase": "sample timestamps (annotation centers), not camera timestamps"})


if __name__ == "__main__":
    standalone("displacement")
