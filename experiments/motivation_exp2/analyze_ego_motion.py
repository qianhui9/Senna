"""Merge per-camera actual pose deltas with same-instance diagnostic metrics."""
import numpy as np
import pandas as pd

from .utils.data_utils import read_table, save_table, standalone


def add_motion_bins(df, cfg):
    result = df.copy()
    a = cfg["analysis"]
    result["translation_bin"] = pd.cut(result.ego_translation_m, a["translation_bins_m"], right=False).astype(str)
    result["yaw_bin"] = pd.cut(result.ego_yaw_change_deg.abs(), a["yaw_bins_deg"], right=False).astype(str)
    if "distance_t0_m" in result:
        result["distance_bin"] = pd.cut(result.distance_t0_m, a["distance_bins_m"], right=False).astype(str)
    return result


def run(cfg, out):
    disp, ego, grid = [read_table(out, n) for n in ["image_displacement", "ego_pairs", "fixed_grid_overlap"]]
    keys = ["pair_id", "camera_channel"]
    motion_fields = [c for c in ego if c.startswith("ego_")]
    result = disp.merge(ego[keys+motion_fields], on=keys, how="left", validate="many_to_one")
    if len(result) and result.ego_translation_m.isna().any():
        raise ValueError("A displacement record has no matching camera Ego Pose pair")
    shape = cfg["analysis"]["ego_grid_shape"]
    primary = grid[(grid.grid_rule == cfg["analysis"]["primary_grid_rule"]) & (grid.grid_rows == shape[0]) & (grid.grid_cols == shape[1])]
    result = result.merge(primary[keys+["instance_token", "grid_overlap", "old_grid_retention"]],
                          on=keys+["instance_token"], how="left", validate="one_to_one")
    result["object_displacement_px"] = result.center_displacement_px
    save_table(out, "ego_motion_displacement", add_motion_bins(result, cfg))
    # Keep each grid variant for M2(d); never combine these as independent objects.
    grid_motion = grid.merge(ego[keys+motion_fields], on=keys, how="left", validate="many_to_one")
    save_table(out, "grid_ego_motion", add_motion_bins(grid_motion, cfg))


if __name__ == "__main__":
    standalone("ego-motion")
