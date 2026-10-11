"""Observed distributions and scene bootstrap, without predetermined claims."""
from collections import defaultdict
import json

from .profile_senna import profile_summary
from .utils.data_utils import read_csv, write_csv, write_json
from .utils.stats_utils import distribution, scene_bootstrap


def summarize_groups(records, keys, metrics, cfg):
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        entry = dict(zip(keys, key))
        entry["records"] = len(rows)
        entry["metrics"] = {}
        for metric in metrics:
            entry["metrics"][metric] = {**distribution([r.get(metric) for r in rows]), **scene_bootstrap(rows, metric, cfg["experiment"]["seed"], cfg["experiment"]["bootstrap_iterations"])}
        result.append(entry)
    return result


def run(cfg, args, out):
    similarity = [r for r in read_csv(out / "temporal_similarity.csv") if r.get("status") == "ok"]
    objects = read_csv(out / "object_persistence.csv")
    profile = read_csv(out / "profiling_samples.csv")
    timing = [r for r in profile if r["measurement_mode"] == "timing"]
    manifest = read_csv(out / "camera_manifest.csv")
    sim_metrics = ["actual_delta_t_ms", "ssim", "psnr_db", "normalized_pixel_difference"] + [f"high_similarity_ratio_{t:.2f}" for t in cfg["experiment"]["ssim_thresholds"]]
    sim = summarize_groups(similarity, ["camera_channel", "gap_group"], sim_metrics, cfg)
    obj = summarize_groups(objects, ["camera_channel"], ["persistence_ratio", "instance_jaccard", "actual_delta_t_ms", "persistent_instance_count", "new_instance_count", "disappeared_instance_count"], cfg)
    summary = profile_summary(profile)
    profile_scene_stats = {m: scene_bootstrap(timing, m, cfg["experiment"]["seed"], cfg["experiment"]["bootstrap_iterations"]) for m in summary["metrics"]}
    shape_groups = summarize_groups(timing, ["input_shape", "vision_tokens_per_image", "visual_tokens_total"], ["vision_tower_ms", "vision_tower_calls", "projector_calls", "img_adapter_calls"], cfg)
    statuses = {p.stem.removesuffix("_status"): json.loads(p.read_text()) for p in sorted(out.glob("*_status.json")) if p.name not in ("invocation_status.json",)}
    issues = {}
    for name in ("manifest_issues", "similarity_issues", "object_issues", "matched_profile_issues"):
        path = out / (name + ".json")
        issues[name] = json.loads(path.read_text()) if path.exists() else []
    result = {"counts": {"indexed_scenes": len({r["scene_token"] for r in manifest}), "indexed_images": len(manifest), "present_images": sum(r["file_exists"].lower() == "true" for r in manifest), "non_key_frames": sum(r["is_key_frame"].lower() == "false" for r in manifest), "selected_pairs": len(read_csv(out / "temporal_pairs.csv")), "valid_pairs": len(similarity), "failed_pairs": len(read_csv(out / "temporal_similarity.csv")) - len(similarity), "global_annotation_pairs": len(objects), "profile_samples": summary["unique_samples"], "profile_repeats": len(timing), "matched_front_pairs": sum(r.get("camera_channel") == "CAM_FRONT" for r in read_csv(out / "matched_profile_pairs.csv"))},
              "similarity": sim, "objects": obj, "profiling": summary, "profile_scene_statistics": profile_scene_stats, "input_shape_groups": shape_groups, "statuses": statuses, "issues": issues,
              "statistical_unit": "Unweighted mean of within-scene means; 95% percentile bootstrap over scene means. CI undefined with <2 scenes. Latency percentiles describe recorded requests/repeats, not independent scene samples.",
              "sources": {"redundancy": "nuScenes per-camera sample_data keyframes and sweeps", "profiling": "existing Senna planning evaluation JSON and original local inference helper", "matched": "exact six keyframe image paths + sample token + scene, verified against camera_manifest; matched_profile_pairs.csv only"}}
    write_json(out / "aggregate_summary.json", result)
    flat = []
    for group in sim:
        for metric, stats in group["metrics"].items():
            flat.append({"camera_channel": group["camera_channel"], "gap_group": group["gap_group"], "metric": metric, "records": stats["n"], "scenes": stats["scenes"], "pooled_mean": stats["mean"], "scene_equal_weight_mean": stats["scene_equal_weight_mean"], "ci95_lower": stats["ci95"][0] if stats["ci95"] else "", "ci95_upper": stats["ci95"][1] if stats["ci95"] else "", "p50": stats["p50"], "p95": stats["p95"], "p99": stats["p99"]})
    write_csv(out / "scene_statistics.csv", flat)
    if profile:
        write_json(out / "profiling_summary.json", summary)


if __name__ == "__main__":
    from .run import main
    raise SystemExit(main("aggregate"))
