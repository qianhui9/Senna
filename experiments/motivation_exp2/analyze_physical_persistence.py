"""Keep global annotated persistence separate from projected-in-view sets."""
from collections import defaultdict

import numpy as np

from .utils.annotation_utils import persistence_counts
from .utils.data_utils import read_table, save_table, standalone


def annotation_sets(df, camera=False):
    groups = defaultdict(lambda: defaultdict(set))
    for row in df.to_dict("records"):
        key = (row["sample_token"], row["camera_channel"]) if camera else row["sample_token"]
        groups[key]["ALL"].add(row["instance_token"])
        groups[key][row["category_name"]].add(row["instance_token"])
    return groups


def persistence_rows(pairs, sets, cameras=None):
    rows = []
    for pair in pairs.to_dict("records"):
        for camera in cameras or [None]:
            akey = (pair["sample_token_t0"], camera) if camera else pair["sample_token_t0"]
            bkey = (pair["sample_token_t1"], camera) if camera else pair["sample_token_t1"]
            a, b = sets[akey], sets[bkey]
            for category in sorted(set(a) | set(b) | {"ALL"}):
                rows.append({**pair, "camera_channel": camera or "GLOBAL", "category_name": category,
                             **persistence_counts(a.get(category, set()), b.get(category, set()))})
    return rows


def run(cfg, out):
    pairs = read_table(out, "keyframe_pairs")
    global_sets = annotation_sets(read_table(out, "global_annotations"))
    projected = read_table(out, "projected_objects")
    camera_sets = annotation_sets(projected[projected.projection_status == "valid"], camera=True)
    save_table(out, "global_object_persistence", persistence_rows(pairs, global_sets))
    save_table(out, "camera_object_persistence", persistence_rows(pairs, camera_sets, cfg["dataset"]["cameras"]))


if __name__ == "__main__":
    standalone("persistence")
