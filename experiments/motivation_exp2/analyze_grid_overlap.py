"""Diagnostic fixed rectangles; these grids are not CLIP tokens or cache hits."""
import pandas as pd

from .analyze_image_displacement import BBOX_FIELDS
from .utils.data_utils import read_table, save_table, standalone
from .utils.geometry_utils import grid_cells, grid_overlap

GRID_FIELDS = ["pair_id", "scene_token", "scene_name", "camera_channel", "instance_token", "category_name", "gap_steps", "delta_t_seconds",
               "motion_group", "grid_rows", "grid_cols", "grid_rule", "grid_overlap", "old_grid_retention", "grid_cells_t0", "grid_cells_t1",
               "grid_intersection", "completely_migrated", "grid_displacement_x", "grid_displacement_y"]


def run(cfg, out):
    data = read_table(out, "image_displacement")
    rows = []
    a = cfg["analysis"]
    # Repeated objects/pairs reuse only diagnostic cell sets within this offline stage.
    cells = {}
    for row in data.to_dict("records"):
        b0, b1 = [[row[f"{k}_t{t}"] for k in BBOX_FIELDS] for t in [0, 1]]
        width, height = row["image_width"], row["image_height"]
        for shape in a["grid_shapes"]:
            for rule in a["grid_rules"]:
                g = []
                for t, box in enumerate([b0, b1]):
                    key = (row[f"sample_data_token_t{t}"], row["instance_token"], tuple(shape), rule)
                    if key not in cells:
                        cells[key] = grid_cells(box, width, height, shape, rule, a["grid_min_coverage_ratio"])
                    g.append(cells[key])
                rows.append({**{k: row[k] for k in ["pair_id", "scene_token", "scene_name", "camera_channel", "instance_token", "category_name", "gap_steps", "delta_t_seconds", "motion_group"]},
                             "grid_rows": shape[0], "grid_cols": shape[1], "grid_rule": rule, **grid_overlap(*g),
                             "grid_displacement_x": row["displacement_x_px"]/(width/shape[1]),
                             "grid_displacement_y": row["displacement_y_px"]/(height/shape[0])})
    save_table(out, "fixed_grid_overlap", pd.DataFrame(rows, columns=GRID_FIELDS))


if __name__ == "__main__":
    standalone("grid-overlap")
