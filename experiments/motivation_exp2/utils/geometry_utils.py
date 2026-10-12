"""Column-vector transforms: T_global_ego maps ego coordinates to global."""
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points


def world_to_camera(points, pose, calibration):
    """T_camera_world = inverse(T_ego_camera) @ inverse(T_world_ego)."""
    q_ego = Quaternion(pose["rotation"]).normalised
    q_cam = Quaternion(calibration["rotation"]).normalised
    ego = q_ego.rotation_matrix.T @ (np.asarray(points) - np.asarray(pose["translation"])[:, None])
    return q_cam.rotation_matrix.T @ (ego - np.asarray(calibration["translation"])[:, None])


def relative_ego_motion(pose0, pose1, delta_t):
    if not np.isfinite(delta_t) or delta_t <= 0:
        raise ValueError("Ego-motion requires a positive actual camera timestamp gap")
    q0, q1 = (Quaternion(p["rotation"]).normalised for p in [pose0, pose1])
    delta = np.asarray(pose1["translation"]) - np.asarray(pose0["translation"])
    local = q0.rotation_matrix.T @ delta
    relative = q0.inverse * q1
    angle = 2 * np.arccos(np.clip(abs(relative.w), 0, 1))
    yaw0, yaw1 = q0.yaw_pitch_roll[0], q1.yaw_pitch_roll[0]
    yaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))
    distance = float(np.linalg.norm(delta))
    return {"ego_translation_m": distance, "ego_translation_x_m": local[0],
            "ego_translation_y_m": local[1], "ego_translation_z_m": local[2],
            "ego_rotation_deg": np.degrees(angle), "ego_yaw_change_deg": np.degrees(yaw),
            "ego_speed_mps": distance / delta_t, "ego_yaw_rate_deg_s": np.degrees(yaw) / delta_t,
            "camera_delta_t_seconds": delta_t}


def project_box(corners, intrinsic, width, height, cfg):
    """Project official camera-frame corners. Exclude near-plane crossings.

    The clipped/raw rectangle area measures image-boundary clipping, not occlusion.
    Pixel origin is top left; x is horizontal, y vertical. Boxes are half-open.
    """
    c = np.asarray(corners, dtype=float)
    fields = ["box_xmin", "box_ymin", "box_xmax", "box_ymax", "box_center_x", "box_center_y",
              "box_width", "box_height", "bbox_area", "raw_box_xmin", "raw_box_ymin",
              "raw_box_xmax", "raw_box_ymax", "raw_bbox_area", "geometric_in_image_area_ratio"]
    result = {k: np.nan for k in fields}
    result.update(projection_status="nonfinite_geometry", geometric_clipped=False,
                  near_plane_crossing=False, min_depth_m=np.nan, max_depth_m=np.nan)
    if c.shape != (3, 8) or not np.isfinite(c).all() or not np.isfinite(intrinsic).all():
        return result
    result.update(min_depth_m=float(c[2].min()), max_depth_m=float(c[2].max()))
    if width <= 0 or height <= 0:
        result["projection_status"] = "invalid_image_dimensions"
        return result
    if c[2].max() <= cfg["near_plane_m"]:
        result["projection_status"] = "behind_or_inside_near_plane"
        return result
    if c[2].min() <= cfg["near_plane_m"]:
        result.update(projection_status="near_plane_crossing_excluded", near_plane_crossing=True)
        return result
    pixels = view_points(c, np.asarray(intrinsic), normalize=True)[:2]
    raw = np.array([pixels[0].min(), pixels[1].min(), pixels[0].max(), pixels[1].max()])
    if not np.isfinite(raw).all():
        return result
    clipped = np.clip(raw, [0, 0, 0, 0], [width, height, width, height])
    x0, y0, x1, y1 = clipped
    w, h = max(0, x1 - x0), max(0, y1 - y0)
    raw_area = (raw[2] - raw[0]) * (raw[3] - raw[1])
    ratio = w * h / raw_area if raw_area > 0 else np.nan
    result.update(dict(zip(["raw_box_xmin", "raw_box_ymin", "raw_box_xmax", "raw_box_ymax"], raw)))
    result.update(box_xmin=x0, box_ymin=y0, box_xmax=x1, box_ymax=y1,
                  box_center_x=(x0 + x1) / 2, box_center_y=(y0 + y1) / 2,
                  box_width=w, box_height=h, bbox_area=w*h, raw_bbox_area=raw_area,
                  geometric_in_image_area_ratio=ratio, geometric_clipped=bool(np.any(raw != clipped)))
    status = "valid"
    if raw_area <= 0:
        status = "degenerate_projection"
    elif w*h == 0:
        status = "outside_image"
    elif w < cfg["min_bbox_width_px"] or h < cfg["min_bbox_height_px"]:
        status = "too_small"
    elif ratio < cfg["min_visible_area_ratio"]:
        status = "low_in_image_area_ratio"
    result["projection_status"] = status
    return result


def bbox_iou(b0, b1):
    a, b = np.asarray(b0), np.asarray(b1)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    areas = [np.maximum(0, x[2:] - x[:2]).prod() for x in [a, b]]
    union = sum(areas) - intersection
    return float(intersection / union) if union > 0 else np.nan


def displacement(b0, b1, width, height):
    c0, c1 = (np.array([(b[0]+b[2])/2, (b[1]+b[3])/2]) for b in [b0, b1])
    dx, dy = c1-c0
    distance = float(np.hypot(dx, dy))
    return {"center_displacement_px": distance, "normalized_displacement": distance / np.hypot(width, height),
            "displacement_x_px": dx, "displacement_y_px": dy,
            "displacement_width_fraction": abs(dx)/width, "displacement_height_fraction": abs(dy)/height,
            "bbox_iou": bbox_iou(b0, b1)}


def grid_cells(box, width, height, shape, rule="any", min_coverage=0.25):
    rows, cols = shape
    x0, y0, x1, y1 = np.clip(box, [0]*4, [width, height, width, height])
    if x1 <= x0 or y1 <= y0:
        return set()
    cw, ch = width / cols, height / rows
    cells = set()
    for r in range(max(0, int(y0 // ch)), min(rows, int(np.ceil(y1/ch)))):
        for c in range(max(0, int(x0 // cw)), min(cols, int(np.ceil(x1/cw)))):
            intersection = max(0, min(x1, (c+1)*cw)-max(x0, c*cw)) * max(0, min(y1, (r+1)*ch)-max(y0, r*ch))
            if rule == "any":
                keep = intersection > 0
            elif rule == "coverage":
                keep = intersection > 0 and intersection / (cw*ch) >= min_coverage
            elif rule == "center":
                keep = x0 <= (c+.5)*cw < x1 and y0 <= (r+.5)*ch < y1
            else:
                raise ValueError(f"Unknown grid rule: {rule}")
            if keep:
                cells.add((r, c))
    return cells


def grid_overlap(g0, g1):
    intersection, union = len(g0 & g1), len(g0 | g1)
    return {"grid_overlap": intersection/union if union else np.nan,
            "old_grid_retention": intersection/len(g0) if g0 else np.nan,
            "grid_cells_t0": len(g0), "grid_cells_t1": len(g1), "grid_intersection": intersection,
            "completely_migrated": not intersection if g0 and g1 else None}
