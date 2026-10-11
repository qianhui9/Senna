"""Global annotated instance persistence at keyframes only."""
from .utils.data_utils import completed, load_nuscenes, mark, selected_scenes, write_csv, write_json

FIELDS = ["scene_name", "scene_token", "camera_channel", "previous_sample_token", "current_sample_token", "previous_timestamp_us", "current_timestamp_us", "actual_delta_t_ms", "persistent_instance_count", "previous_instance_count", "current_instance_count", "persistence_ratio", "instance_jaccard", "new_instance_count", "disappeared_instance_count", "scope"]


def counts(previous, current):
    common = previous & current
    return {"persistent_instance_count": len(common), "previous_instance_count": len(previous), "current_instance_count": len(current), "persistence_ratio": len(common) / len(previous) if previous else "", "instance_jaccard": len(common) / len(previous | current) if previous | current else "", "new_instance_count": len(current - previous), "disappeared_instance_count": len(previous - current)}


def run(cfg, args, out, nusc=None):
    if completed(out, "objects", args):
        return
    nusc = nusc or load_nuscenes(cfg)
    rows, issues = [], []
    if not getattr(nusc, "sample_annotation", []):
        write_csv(out / "object_persistence.csv", [], FIELDS)
        mark(out, "objects", "unavailable", reason="No sample_annotation table records")
        return
    for scene in selected_scenes(nusc, cfg):
        token, previous, previous_instances, seen = scene["first_sample_token"], None, None, set()
        while token:
            if token in seen:
                issues.append({"scene_name": scene["name"], "token": token, "reason": "sample_cycle"})
                break
            seen.add(token)
            try:
                current = nusc.get("sample", token)
                if current["scene_token"] != scene["token"]:
                    raise ValueError("sample scene boundary")
                instances = {nusc.get("sample_annotation", a)["instance_token"] for a in current["anns"]}
                if previous is not None:
                    delta = (int(current["timestamp"]) - int(previous["timestamp"])) / 1000
                    if delta <= 0:
                        raise ValueError("non-increasing keyframe timestamps")
                    rows.append({"scene_name": scene["name"], "scene_token": scene["token"], "camera_channel": "GLOBAL_3D_ANNOTATIONS", "previous_sample_token": previous["token"], "current_sample_token": token, "previous_timestamp_us": previous["timestamp"], "current_timestamp_us": current["timestamp"], "actual_delta_t_ms": delta, **counts(previous_instances, instances), "scope": "global annotated instances; camera visibility not evaluated"})
                previous, previous_instances = current, instances
                token = current["next"]
            except (KeyError, ValueError) as exc:
                issues.append({"scene_name": scene["name"], "token": token, "reason": str(exc)})
                break
    write_csv(out / "object_persistence.csv", rows, FIELDS)
    write_json(out / "object_issues.json", issues)
    mark(out, "objects", "complete" if rows and not issues else "partial" if rows else "unavailable", pairs=len(rows), issues=len(issues), definition="intersection / previous instance count; empty previous set is undefined, not zero")


if __name__ == "__main__":
    from .run import main
    raise SystemExit(main("objects"))
