"""Join real keyframe camera pairs to GLOBAL annotation changes by exact tokens."""
from .utils.data_utils import write_csv
from .utils.stats_utils import scene_bootstrap


def match_objects(similarity, manifest, objects):
    frames = {r["sample_data_token"]: r for r in manifest}
    annotations = {(r["scene_token"], r["previous_sample_token"], r["current_sample_token"]): r for r in objects}
    matched = []
    for pair in similarity:
        previous = frames.get(pair["previous_sample_data_token"])
        current = frames.get(pair["current_sample_data_token"])
        if not previous or not current:
            continue
        if any(str(r["is_key_frame"]).lower() != "true" for r in (previous, current)):
            continue
        if any(r["scene_token"] != pair["scene_token"] or r["camera_channel"] != pair["camera_channel"] for r in (previous, current)):
            continue
        if any(r["sample_token"] != pair[field] for r, field in ((previous, "previous_sample_token"), (current, "current_sample_token"))):
            continue
        annotation = annotations.get((pair["scene_token"], previous["sample_token"], current["sample_token"]))
        if annotation is None:
            continue
        changed = int(annotation["new_instance_count"]) + int(annotation["disappeared_instance_count"]) > 0
        matched.append({**pair, **{k: annotation[k] for k in ("persistence_ratio", "instance_jaccard", "new_instance_count", "disappeared_instance_count", "persistent_instance_count")},
                        "annotation_previous_timestamp_us": annotation["previous_timestamp_us"],
                        "annotation_current_timestamp_us": annotation["current_timestamp_us"],
                        "global_instance_set_changed": int(changed),
                        "annotation_scope": "GLOBAL_3D_ANNOTATIONS; camera visibility and persistent-object motion not evaluated"})
    return matched


def joint_summary(rows, thresholds, cfg, out):
    write_csv(out / "matched_object_pairs.csv", rows)
    groups = []
    for camera in sorted({r["camera_channel"] for r in rows}):
        for threshold in thresholds:
            selected = [r for r in rows if r["camera_channel"] == camera and float(r["ssim"]) >= threshold]
            unique = {(r["scene_token"], r["previous_sample_token"], r["current_sample_token"]) for r in selected}
            groups.append({"camera_channel": camera, "ssim_threshold": threshold, "camera_pairs": len(selected),
                           "unique_global_transitions": len(unique),
                           "changed_global_transitions": sum(r["global_instance_set_changed"] for r in selected),
                           **scene_bootstrap(selected, "global_instance_set_changed", cfg["experiment"]["seed"], cfg["experiment"]["bootstrap_iterations"])})
    return {"status": "available" if rows else "unavailable", "matched_camera_pairs": len(rows),
            "unique_global_transitions": len({(r["scene_token"], r["previous_sample_token"], r["current_sample_token"]) for r in rows}),
            "conditional_changes": groups,
            "interpretation": "High per-camera image SSIM co-occurring with a GLOBAL annotated instance-set change. This does not establish that a changed object is visible in that camera. Motion of persistent identities is not measured; unchanged identity sets do not imply unchanged object states."}
