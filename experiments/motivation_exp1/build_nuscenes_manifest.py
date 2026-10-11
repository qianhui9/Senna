"""Follow official sample_data camera sweep links, without assuming synchrony."""
from bisect import bisect_left
from collections import defaultdict
import hashlib
from pathlib import Path

from .utils.data_utils import CAMERAS, LOG, completed, load_nuscenes, mark, selected_scenes, write_csv, write_json

FRAME_FIELDS = ["scene_name", "scene_token", "camera_channel", "sample_data_token", "sample_token", "timestamp_us", "image_path", "is_key_frame", "ego_pose_token", "calibrated_sensor_token", "prev_sample_data_token", "next_sample_data_token", "file_exists"]
PAIR_FIELDS = ["pair_id", "scene_name", "scene_token", "camera_channel", "previous_sample_data_token", "current_sample_data_token", "previous_sample_token", "current_sample_token", "previous_image_path", "current_image_path", "previous_timestamp_us", "current_timestamp_us", "actual_delta_t_ms", "gap_group", "target_delta_t_ms"]


def collect_frames(nusc, scenes, cfg):
    rows, issues = [], []
    low, high = cfg["experiment"]["abnormal_interval_seconds"]
    for scene in scenes:
        first = nusc.get("sample", scene["first_sample_token"])
        for camera in CAMERAS:
            token = first["data"].get(camera)
            seen, previous_time, previous_token = set(), None, ""
            while token:
                if token in seen:
                    issues.append({"scene_name": scene["name"], "camera_channel": camera, "token": token, "reason": "cycle"})
                    break
                seen.add(token)
                try:
                    frame = nusc.get("sample_data", token)
                    sample = nusc.get("sample", frame["sample_token"])
                    if sample["scene_token"] != scene["token"]:
                        issues.append({"scene_name": scene["name"], "camera_channel": camera, "token": token, "reason": "scene_boundary"})
                        break
                    calibrated = nusc.get("calibrated_sensor", frame["calibrated_sensor_token"])
                    sensor = nusc.get("sensor", calibrated["sensor_token"])
                    if sensor["channel"] != camera:
                        raise ValueError("camera channel mismatch")
                    timestamp = int(frame["timestamp"])
                    if previous_time is not None and not low <= (timestamp - previous_time) / 1e6 <= high:
                        issues.append({"scene_name": scene["name"], "camera_channel": camera, "token": token, "reason": "abnormal_interval", "delta_t_ms": (timestamp - previous_time) / 1000})
                    if previous_token and frame["prev"] != previous_token:
                        issues.append({"scene_name": scene["name"], "camera_channel": camera, "token": token, "reason": "nonreciprocal_link"})
                    path = (Path(cfg["nuscenes"]["dataroot"]) / frame["filename"]).resolve()
                    exists = path.is_file()
                    if not exists:
                        issues.append({"scene_name": scene["name"], "camera_channel": camera, "token": token, "reason": "missing_image", "path": str(path)})
                    rows.append(dict(zip(FRAME_FIELDS, [scene["name"], scene["token"], camera, token, frame["sample_token"], timestamp, str(path), frame["is_key_frame"], frame["ego_pose_token"], frame["calibrated_sensor_token"], frame["prev"], frame["next"], exists])))
                    previous_time, previous_token = timestamp, token
                    token = frame["next"]
                except (KeyError, ValueError, TypeError) as exc:
                    issues.append({"scene_name": scene["name"], "camera_channel": camera, "token": token, "reason": "invalid_record", "detail": str(exc)})
                    break
    rows.sort(key=lambda r: (r["scene_name"], r["camera_channel"], r["timestamp_us"], r["sample_data_token"]))
    return rows, issues


def make_pairs(frames, cfg):
    groups = defaultdict(list)
    for frame in frames:
        groups[frame["scene_token"], frame["camera_channel"]].append(frame)
    candidates = []
    gaps = cfg["experiment"]["temporal_gaps_seconds"]
    tolerance = cfg["experiment"]["gap_tolerance_seconds"] * 1e6
    for group in sorted(groups):
        seq = sorted(groups[group], key=lambda r: (int(r["timestamp_us"]), r["sample_data_token"]))
        times = [int(r["timestamp_us"]) for r in seq]
        for gap_index, gap in enumerate(gaps):
            name = "adjacent" if gap_index == 0 else f"{gap:g}s"
            for i, previous in enumerate(seq[:-1]):
                if gap_index == 0:
                    j = i + 1
                    # Missing frames never create fictitious adjacent observations.
                    if previous["next_sample_data_token"] != seq[j]["sample_data_token"]:
                        continue
                else:
                    k = bisect_left(times, times[i] + gap * 1e6, lo=i + 1)
                    choices = [j for j in (k - 1, k) if i < j < len(seq)]
                    if not choices:
                        continue
                    j = min(choices, key=lambda j: (abs(times[j] - times[i] - gap * 1e6), times[j]))
                    if abs(times[j] - times[i] - gap * 1e6) > tolerance:
                        continue
                current = seq[j]
                delta = times[j] - times[i]
                if delta <= 0 or not all(str(x["file_exists"]).lower() == "true" for x in (previous, current)):
                    continue
                pair_id = hashlib.sha256(f"{previous['sample_data_token']}:{current['sample_data_token']}:{name}".encode()).hexdigest()[:24]
                candidates.append(dict(zip(PAIR_FIELDS, [pair_id, previous["scene_name"], group[0], group[1], previous["sample_data_token"], current["sample_data_token"], previous["sample_token"], current["sample_token"], previous["image_path"], current["image_path"], times[i], times[j], delta / 1000, name, gap * 1000])))
    # Balanced round-robin over scene/camera/gap, with seeded hash ordering within
    # each stratum. Content and measured similarity never influence selection.
    strata = defaultdict(list)
    seed = cfg["experiment"]["seed"]
    for row in candidates:
        strata[row["scene_token"], row["camera_channel"], row["gap_group"]].append(row)
    for rows in strata.values():
        rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['pair_id']}".encode()).digest())
    keys = sorted(strata, key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).digest())
    selected, offset = [], 0
    while len(selected) < cfg["experiment"]["max_temporal_pairs"]:
        available = False
        for key in keys:
            if offset < len(strata[key]):
                available = True
                selected.append(strata[key][offset])
                if len(selected) == cfg["experiment"]["max_temporal_pairs"]:
                    break
        if not available:
            break
        offset += 1
    return sorted(selected, key=lambda r: (r["scene_name"], r["camera_channel"], r["gap_group"], r["previous_timestamp_us"])), len(candidates)


def run(cfg, args, out, nusc=None):
    if completed(out, "index", args):
        LOG.info("Skipping completed index")
        return
    nusc = nusc or load_nuscenes(cfg)
    scenes = selected_scenes(nusc, cfg)
    frames, issues = collect_frames(nusc, scenes, cfg)
    pairs, candidate_count = make_pairs(frames, cfg)
    write_csv(out / "camera_manifest.csv", frames, FRAME_FIELDS)
    write_csv(out / "temporal_pairs.csv", pairs, PAIR_FIELDS)
    write_json(out / "manifest_issues.json", issues)
    write_json(out / "manifest_summary.json", {"scenes": [{"name": s["name"], "token": s["token"], "description": s.get("description", "")} for s in scenes], "frames": len(frames), "missing_images": sum(not r["file_exists"] for r in frames), "non_key_frames": sum(not r["is_key_frame"] for r in frames), "candidate_pairs": candidate_count, "selected_pairs": len(pairs), "issues": len(issues), "selection": "seeded scene selection; scene/camera/gap round-robin with seeded content-independent pair hashes; first gap denotes adjacent sample_data links"})
    if not pairs:
        raise ValueError("No valid temporal pairs; inspect manifest_issues.json and installed sweeps")
    mark(out, "index", frames=len(frames), pairs=len(pairs))
    LOG.info("Indexed %d frames (%d non-key frames), selected %d pairs", len(frames), sum(not r["is_key_frame"] for r in frames), len(pairs))


if __name__ == "__main__":
    from .run import main
    raise SystemExit(main("index"))
