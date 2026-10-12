"""Separate annotated sample.next pairs from E1's camera sweep pairs."""
import json
from pathlib import Path

import pandas as pd

from .utils.annotation_utils import instance_map
from .utils.data_utils import LOG, load_nuscenes, save_table, select_scenes, standalone, write_json
from .utils.geometry_utils import relative_ego_motion

PAIR_FIELDS = ["pair_id", "scene_token", "scene_name", "sample_token_t0", "sample_token_t1",
               "timestamp_t0", "timestamp_t1", "delta_t_seconds", "gap_steps", "index_t0", "index_t1"]
ANNOTATION_FIELDS = ["scene_token", "scene_name", "sample_token", "annotation_token", "instance_token",
                     "category_name", "visibility_token", "global_x", "global_y", "global_z"]


def sample_sequence(nusc, scene):
    samples, seen = [], set()
    token = scene["first_sample_token"]
    previous = ""
    while token:
        if token in seen:
            raise ValueError(f"Cycle in sample.next: {scene['name']} {token}")
        seen.add(token)
        sample = nusc.get("sample", token)
        if sample["scene_token"] != scene["token"] or sample["prev"] != previous:
            raise ValueError(f"Cross-scene or broken sample links: {token}")
        if samples and sample["timestamp"] <= samples[-1]["timestamp"]:
            raise ValueError(f"Non-increasing sample timestamps: {token}")
        samples.append(sample)
        previous, token = token, sample["next"]
    if len(samples) != scene["nbr_samples"] or samples[-1]["token"] != scene["last_sample_token"]:
        raise ValueError(f"Scene sample count/end mismatch: {scene['name']}")
    return samples


def keyframe_pairs(samples, scene, gaps):
    pairs = []
    for gap in sorted(gaps):
        for i in range(len(samples)-gap):
            a, b = samples[i], samples[i+gap]
            if a["scene_token"] != b["scene_token"] or a["scene_token"] != scene["token"]:
                raise ValueError("Cross-scene pair forbidden")
            dt = (b["timestamp"]-a["timestamp"])/1e6
            if dt <= 0:
                raise ValueError("Invalid pair time order")
            pairs.append(dict(pair_id=f"{a['token']}:{b['token']}", scene_token=scene["token"],
                              scene_name=scene["name"], sample_token_t0=a["token"], sample_token_t1=b["token"],
                              timestamp_t0=a["timestamp"], timestamp_t1=b["timestamp"], delta_t_seconds=dt,
                              gap_steps=gap, index_t0=i, index_t1=i+gap))
    return pairs


def pose_from_frame(row):
    return {"translation": [row["pose_x"], row["pose_y"], row["pose_z"]],
            "rotation": [row["pose_qw"], row["pose_qx"], row["pose_qy"], row["pose_qz"]]}


def run(cfg, out, nusc=None):
    nusc = nusc or load_nuscenes(cfg)
    scenes = select_scenes(nusc, cfg)
    if not scenes:
        raise ValueError("No selected scenes")
    e1_frames = {}
    if cfg["dataset"]["frame_manifest"]:
        manifest = Path(cfg["dataset"]["frame_manifest"])
        if not manifest.is_file():
            raise FileNotFoundError(f"E1 frame manifest missing: {manifest}")
        df = pd.read_csv(manifest)
        e1_frames = df.set_index("sample_data_token").to_dict("index")
    pairs, frames, annotations, issues = [], [], [], []
    reused = 0
    for scene in scenes:
        sequence = sample_sequence(nusc, scene)
        pairs.extend(keyframe_pairs(sequence, scene, cfg["pairs"]["keyframe_gaps"]))
        for sample in sequence:
            anns = [nusc.get("sample_annotation", t) for t in sample["anns"]]
            instance_map(anns)
            for ann in anns:
                if ann["sample_token"] != sample["token"]:
                    raise ValueError("Annotation belongs to another sample")
                category = ann.get("category_name")
                if not category:
                    inst = nusc.get("instance", ann["instance_token"])
                    category = nusc.get("category", inst["category_token"])["name"]
                annotations.append(dict(scene_token=scene["token"], scene_name=scene["name"],
                                        sample_token=sample["token"], annotation_token=ann["token"],
                                        instance_token=ann["instance_token"], category_name=category,
                                        visibility_token=ann["visibility_token"],
                                        **dict(zip(["global_x", "global_y", "global_z"], ann["translation"]))))
            for camera in cfg["dataset"]["cameras"]:
                token = sample["data"].get(camera)
                if not token:
                    raise ValueError(f"Missing keyframe camera {camera} in sample {sample['token']}")
                sd = nusc.get("sample_data", token)
                if not sd["is_key_frame"] or sd["sample_token"] != sample["token"] or sd["channel"] != camera:
                    raise ValueError(f"Not the expected annotated camera keyframe: {token}")
                pose = nusc.get("ego_pose", sd["ego_pose_token"])
                path = str(Path(nusc.get_sample_data_path(token)).resolve())
                if token in e1_frames:
                    e1 = e1_frames[token]
                    if (int(e1["timestamp_us"]) != sd["timestamp"] or e1["camera_channel"] != camera
                            or e1["scene_token"] != scene["token"] or e1["sample_token"] != sample["token"]
                            or e1["ego_pose_token"] != sd["ego_pose_token"]
                            or e1["calibrated_sensor_token"] != sd["calibrated_sensor_token"]):
                        raise ValueError(f"E1 manifest disagrees with configured devkit: {token}")
                    reused += 1
                exists = Path(path).is_file()
                if not exists:
                    issues.append({"kind": "missing_image", "sample_data_token": token, "path": path})
                frames.append(dict(scene_token=scene["token"], scene_name=scene["name"], sample_token=sample["token"],
                                   sample_data_token=token, camera_channel=camera, sample_timestamp=sample["timestamp"],
                                   camera_timestamp=sd["timestamp"], timestamp_us=sd["timestamp"],
                                   annotation_camera_time_offset_ms=(sd["timestamp"]-sample["timestamp"])/1000,
                                   image_path=path, image_exists=exists, image_width=sd["width"], image_height=sd["height"],
                                   ego_pose_token=sd["ego_pose_token"], calibrated_sensor_token=sd["calibrated_sensor_token"],
                                   **dict(zip(["pose_x", "pose_y", "pose_z"], pose["translation"])),
                                   **dict(zip(["pose_qw", "pose_qx", "pose_qy", "pose_qz"], pose["rotation"])) ))
        LOG.info("Indexed %s: %d annotated keyframes", scene["name"], len(sequence))
    if not pairs:
        raise ValueError("Selected scenes contain no pairs for requested gaps")
    frame_index = {(f["sample_token"], f["camera_channel"]): f for f in frames}
    ego_pairs = []
    for pair in pairs:
        if not .35 <= pair["delta_t_seconds"]/pair["gap_steps"] <= .65:
            issues.append({"kind": "unusual_sample_interval", "pair_id": pair["pair_id"],
                           "gap_steps": pair["gap_steps"], "delta_t_seconds": pair["delta_t_seconds"]})
        for camera in cfg["dataset"]["cameras"]:
            a, b = [frame_index[(pair[k], camera)] for k in ["sample_token_t0", "sample_token_t1"]]
            dt = (b["camera_timestamp"]-a["camera_timestamp"])/1e6
            motion = relative_ego_motion(pose_from_frame(a), pose_from_frame(b), dt)
            if abs(dt-pair["delta_t_seconds"]) > .1:
                issues.append({"kind": "camera_sample_gap_difference", "pair_id": pair["pair_id"], "camera_channel": camera, "difference_seconds": dt-pair["delta_t_seconds"]})
            ego_pairs.append({**pair, "camera_channel": camera, "sample_data_token_t0": a["sample_data_token"],
                              "sample_data_token_t1": b["sample_data_token"], "camera_timestamp_t0": a["camera_timestamp"],
                              "camera_timestamp_t1": b["camera_timestamp"], **motion})
    save_table(out, "keyframe_pairs", pairs, PAIR_FIELDS)
    save_table(out, "camera_frames", frames)
    save_table(out, "global_annotations", annotations, ANNOTATION_FIELDS)
    save_table(out, "ego_pairs", ego_pairs)
    write_json(out / "metadata" / "scene_list.json", [{k: s[k] for k in ["name", "token", "description"]} for s in scenes])
    write_json(out / "metadata" / "index_summary.json", {"scenes": len(scenes), "keyframes": len({f['sample_token'] for f in frames}),
               "pairs": len(pairs), "camera_frames": len(frames), "global_annotations": len(annotations),
               "e1_keyframe_rows_verified": reused, "issues": issues, "strategy": "All within-scene sample.next gaps; no sweeps; actual sample and per-camera timestamps"})


if __name__ == "__main__":
    standalone("build-pairs")
