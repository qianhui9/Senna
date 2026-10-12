"""Official devkit transforms at actual camera poses, then offline rectangles."""
import numpy as np
import pandas as pd
from PIL import Image
from nuscenes.utils.geometry_utils import BoxVisibility, view_points

from .utils.data_utils import LOG, load_nuscenes, read_table, save_table, standalone, write_json
from .utils.geometry_utils import project_box, world_to_camera


def run(cfg, out, nusc=None):
    nusc = nusc or load_nuscenes(cfg)
    frames = read_table(out, "camera_frames")
    annotations = read_table(out, "global_annotations").set_index("annotation_token").to_dict("index")
    rows, regressions = [], []
    for i, frame in enumerate(frames.to_dict("records")):
        token = frame["sample_data_token"]
        path, boxes, intrinsic = nusc.get_sample_data(token, box_vis_level=BoxVisibility.NONE)
        sample = nusc.get("sample", frame["sample_token"])
        if {b.token for b in boxes} != set(sample["anns"]):
            raise ValueError(f"Official Box token / annotation token correspondence failed: {token}")
        image_error = None
        if frame["image_exists"]:
            try:
                with Image.open(path) as image:
                    if image.size != (frame["image_width"], frame["image_height"]):
                        image_error = "image_dimension_mismatch"
            except OSError:
                image_error = "unreadable_image"
        else:
            image_error = "missing_image"
        for box in boxes:
            annotation = annotations[box.token]
            projected = project_box(box.corners(), intrinsic, frame["image_width"], frame["image_height"], cfg["projection"])
            if image_error:
                projected["projection_status"] = image_error
            rows.append({**{k: frame[k] for k in ["scene_name", "scene_token", "sample_token", "sample_data_token", "camera_channel",
                         "timestamp_us", "sample_timestamp", "camera_timestamp", "annotation_camera_time_offset_ms",
                         "image_width", "image_height"]}, **annotation, "annotation_token": box.token,
                         "camera_distance_m": float(np.linalg.norm(box.center)), **projected})
            # One independent transform comparison per scene/camera, persisted as real regression evidence.
            key = (frame["scene_token"], frame["camera_channel"])
            if projected["projection_status"] == "valid" and key not in {tuple(r["key"]) for r in regressions}:
                sd = nusc.get("sample_data", token)
                pose = nusc.get("ego_pose", sd["ego_pose_token"])
                calibration = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                custom = world_to_camera(nusc.get_box(box.token).corners(), pose, calibration)
                error = float(np.abs(custom-box.corners()).max())
                if error > 1e-8:
                    raise ValueError(f"World→Ego→Camera disagrees with devkit: {error} m")
                pixels = view_points(box.corners(), intrinsic, normalize=True)
                manual_pixels = np.asarray(intrinsic) @ custom
                manual_pixels /= manual_pixels[2:3]
                pixel_error = float(np.abs(manual_pixels-pixels).max())
                if pixel_error > 1e-6:
                    raise ValueError("Independent projection differs from official view_points")
                regressions.append({"key": key, "sample_data_token": token, "annotation_token": box.token,
                                    "instance_token": annotation["instance_token"], "transform_max_error_m": error,
                                    "pixel_max_error_px": pixel_error,
                                    "official_corners_camera": box.corners().tolist(), "official_pixels": pixels.tolist(),
                                    "image_path": path, "box": [projected[k] for k in ["box_xmin", "box_ymin", "box_xmax", "box_ymax"]]})
        if i % 100 == 0:
            LOG.info("Projected %d/%d camera keyframes", i+1, len(frames))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Selected scenes have no annotated objects to project")
    save_table(out, "projected_objects", df)
    write_json(out / "metadata" / "projection_summary.json", {"attempts": len(df), "status_counts": df.projection_status.value_counts().to_dict(),
               "valid": int((df.projection_status == "valid").sum()), "geometric_clipped": int(df.geometric_clipped.sum()),
               "near_plane_policy": "explicitly exclude boxes crossing z=near_plane_m",
               "area_ratio": "image-clipped / raw projected rectangle area, not occlusion visibility",
               "annotation_timing": "Keyframe sample annotations without interpolation, transformed using actual camera ego pose"})
    write_json(out / "metadata" / "projection_regression.json", regressions)


if __name__ == "__main__":
    standalone("project-boxes")
