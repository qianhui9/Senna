"""Official-annotation queries. Future data never enter request construction."""
from collections import Counter
import hashlib
import json
import math
import random

import numpy as np
from pyquaternion import Quaternion

from data_tools.senna_inputs import (
    CAMERAS, IMAGE_PROMPT, PROTOCOL_VERSION, current_speed, future_positions,
    navigation_command, planning_question, sample_images, sensor_position,
)
from experiments.motivation_exp1.utils.data_utils import write_csv, write_json

QUERY_TYPES = ("category", "existence", "position", "motion_prediction", "meta_action_planning")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def lidar_coordinates(nusc, sample, global_points):
    _, pose, calibration = sensor_position(nusc, sample)
    xyz = np.asarray(global_points, dtype=float) - np.asarray(pose["translation"])
    xyz = xyz @ Quaternion(pose["rotation"]).rotation_matrix
    xyz -= np.asarray(calibration["translation"])
    return xyz @ Quaternion(calibration["rotation"]).rotation_matrix


def in_scope(position, selection, distance):
    x, y = position[:2]
    return (y > selection["forward_min_m"] and math.hypot(x, y) < distance
            and abs(math.degrees(math.atan2(x, y))) <= selection["front_half_angle_degrees"])


def position_label(position, selection):
    index = int(np.searchsorted(selection["position_lateral_edges_m"], position[0], side="right"))
    return selection["position_labels"][index]


def scene_samples(nusc, scene):
    result, seen, token = [], set(), scene["first_sample_token"]
    previous = ""
    while token:
        if token in seen:
            raise ValueError("Cycle in sample.next")
        seen.add(token)
        sample = nusc.get("sample", token)
        if sample["scene_token"] != scene["token"] or sample["prev"] != previous:
            raise ValueError("Broken sample links or cross-scene sequence")
        if result and sample["timestamp"] <= result[-1]["timestamp"]:
            raise ValueError("Non-increasing sample timestamps")
        result.append(sample)
        previous, token = token, sample["next"]
    if previous != scene["last_sample_token"] or len(result) != scene["nbr_samples"]:
        raise ValueError("Incomplete scene sample chain")
    return result


def past_speed(nusc, annotation, sample):
    """Backward difference only: nusc.box_velocity uses future data and is avoided."""
    if not annotation.get("prev"):
        raise ValueError("current_object_velocity_unavailable")
    previous = nusc.get("sample_annotation", annotation["prev"])
    prev_sample = nusc.get("sample", previous["sample_token"])
    dt = (sample["timestamp"] - prev_sample["timestamp"]) / 1e6
    if (dt <= 0 or previous["instance_token"] != annotation["instance_token"]
            or prev_sample["scene_token"] != sample["scene_token"]
            or previous["sample_token"] != sample["prev"]):
        raise ValueError("invalid_previous_object_annotation")
    return float(np.linalg.norm(np.asarray(annotation["translation"])[:2]
                                - np.asarray(previous["translation"])[:2]) / dt)


def future_object(nusc, annotation, sample, selection):
    """Require every next keyframe; use original object-local/Senna axis convention."""
    points, tokens, timestamps = [], [], []
    previous, previous_sample = annotation, sample
    for _ in range(selection["future_steps"]):
        if not previous.get("next") or not previous_sample["next"]:
            raise ValueError("incomplete_future_object_annotations")
        following = nusc.get("sample_annotation", previous["next"])
        following_sample = nusc.get("sample", following["sample_token"])
        dt = (following_sample["timestamp"] - previous_sample["timestamp"]) / 1e6
        if (following["instance_token"] != annotation["instance_token"]
                or following_sample["scene_token"] != sample["scene_token"]
                or following["sample_token"] != previous_sample["next"]
                or following["prev"] != previous["token"]
                or abs(dt - selection["expected_interval_s"]) > selection["interval_tolerance_s"]):
            raise ValueError("invalid_future_object_link_or_interval")
        points.append(following["translation"])
        tokens.append(following["token"])
        timestamps.append(following_sample["timestamp"])
        previous, previous_sample = following, following_sample
    # Match get_mot_pred_qa's gt_agent_fut_trajs_vcs transformation exactly.
    _, _, calibrated = sensor_position(nusc, sample)
    relative = (np.asarray(points) - np.asarray(annotation["translation"])) @ Quaternion(annotation["rotation"]).rotation_matrix
    relative = (relative - np.asarray(calibrated["translation"])) @ Quaternion(calibrated["rotation"]).rotation_matrix
    return relative[:, :2], {"future_annotation_tokens": tokens, "future_timestamps": timestamps,
                             "future_global_positions": points, "future_object_local_positions": relative[:, :2].tolist()}


def motion_question(position, speed):
    x, y = position[:2]
    longitudinal = f"{int(abs(y))} meters {'ahead' if y >= 0 else 'behind'}"
    lateral = f"{int(abs(x))} meters to the {'right' if x >= 0 else 'left'}"
    return ("You are driving, I will now provide you with the location and velocity information of dynamic objects in the front view image. "
            "Please predict their future driving behaviors, which can be divided into SPEED decisions and PATH decisions. "
            "SPEED includes KEEP, ACCELERATE, DECELERATE, and STOP, while PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, and LEFT_TURN. "
            "I will now provide you with the position and velocity information of the dynamic objects: \n"
            f"Object 1: vehicle, {longitudinal}, {lateral}, speed of {int(speed)} m/s.\n"
            "Please predict the future driving behaviors of these objects based on the front view image. "
            "Answer only Object 1: SPEED, PATH.")


def annotations_for(nusc, sample):
    """Do not silently turn corrupt/missing annotation metadata into a negative."""
    result = []
    for token in sample["anns"]:
        a = nusc.get("sample_annotation", token)
        if (a["sample_token"] != sample["token"] or not a["instance_token"]
                or not a["category_name"] or len(a["translation"]) != 3
                or not np.isfinite(a["translation"]).all()):
            raise ValueError("incomplete_current_annotations")
        result.append(a)
    return result


def make_queries(nusc, can_bus, sample, cfg):
    from data_tools.senna_qa_utils import get_obj_acc_or_dec, get_obj_turn_or_lane_change
    from eval_tools.prepare_senna_plan_eval import SPEED_PLAN, PATH_PLAN
    s = cfg["selection"]
    records, exclusions = {}, {}
    try:
        images = sample_images(nusc, sample)
    except (FileNotFoundError, KeyError) as exc:
        return {}, {t: f"missing_six_view_images: {exc}" for t in cfg["experiment"]["query_types"]}
    try:
        annotations = annotations_for(nusc, sample)
        positions = lidar_coordinates(nusc, sample, [a["translation"] for a in annotations]) if annotations else []
        objects = list(zip(annotations, positions))
    except (ValueError, KeyError) as exc:
        return {}, {t: f"incomplete_current_annotations: {exc}" for t in cfg["experiment"]["query_types"]}
    candidates = [(a, p) for a, p in objects
                  if a["category_name"].startswith(s["vehicle_prefix"])
                  and a["category_name"] not in s["excluded_vehicle_categories"]
                  and in_scope(p, s, s["max_distance_m"])
                  and a["num_lidar_pts"] + a["num_radar_pts"] >= s["min_sensor_points"]]
    candidates.sort(key=lambda ap: (float(np.linalg.norm(ap[1][:2])), ap[0]["instance_token"]))
    target = candidates[0] if candidates else None
    protocol = {"version": PROTOCOL_VERSION, "camera_order": list(CAMERAS), "legacy_camera_order": False,
                "legacy_navigation": False, "navigation_source": "none", "future_label_in_prompt": False}

    def add(kind, text, truth, instances, attributes, signature, facts, offline=None):
        records[kind] = {
            "scene_token": sample["scene_token"], "sample_token": sample["token"],
            "prev_sample_token": sample["prev"], "next_sample_token": sample["next"],
            "timestamp": sample["timestamp"], "query_id": kind + "_v1", "query_type": kind,
            "query_text": text, "ground_truth": truth,
            "dependency_instance_tokens": sorted(instances), "dependency_attributes": attributes,
            "dependency_signature": signature, "state_facts": facts,
            "image_paths": images, "input_protocol": dict(protocol), "offline_ground_truth": offline or {},
        }

    selection_text = (f"the nearest vehicle (excluding {', '.join(s['excluded_vehicle_categories'])}) ahead within {s['max_distance_m']:g} meters "
                      f"and {s['front_half_angle_degrees']:g} degrees of the forward axis, beyond {s['forward_min_m']:g} meters forward")
    for kind in ("category", "position", "motion_prediction"):
        if not target:
            exclusions[kind] = "no_deterministic_vehicle_target"
    if target:
        a, p = target
        instance, category = a["instance_token"], a["category_name"]
        attributes = {"annotation_token": a["token"], "instance_token": instance, "category": category,
                      "position_lidar_m": p.tolist(), "position_label": position_label(p, s),
                      "selection_rule": "nearest_in_scope_vehicle_then_instance_token",
                      "sensor_points": a["num_lidar_pts"] + a["num_radar_pts"]}
        # Selector identity is necessary: a still-present old vehicle may cease to be nearest.
        selector = {"selected_vehicle": instance}
        add("category", f"What is the category of {selection_text}? Answer only its nuScenes category name.",
            category, [instance], attributes, [instance, category],
            {**selector, f"category:{instance}": category})
        label = position_label(p, s)
        add("position", f"Where is {selection_text} relative to you? "
            f"Answer only one of {', '.join(s['position_labels'])}; lateral boundaries in meters: {s['position_lateral_edges_m']} (x right).",
            label, [instance], attributes, [instance, label],
            {**selector, f"position:{instance}": label})
        try:
            speed = past_speed(nusc, a, sample)
            trajectory, offline = future_object(nusc, a, sample, s)
            truth = SPEED_PLAN[get_obj_acc_or_dec(trajectory)] + ", " + PATH_PLAN[get_obj_turn_or_lane_change(trajectory)]
            state = int(np.searchsorted(s["current_speed_bins_mps"], speed, side="right"))
            motion_attributes = {**attributes, "current_speed_mps": speed, "current_motion_state_bin": state,
                                 "prediction_label": truth, "velocity_source": "previous_annotation_only"}
            add("motion_prediction", motion_question(p, speed), "Object 1: " + truth, [instance], motion_attributes,
                [instance, state, truth], {**selector, f"motion_bin:{instance}": state, f"oracle_prediction:{instance}": truth}, offline)
        except (ValueError, KeyError) as exc:
            exclusions["motion_prediction"] = str(exc)
    # All official instances in the scope belong in the existence dependency set, including low-point annotations.
    members = [(a, p) for a, p in objects if any(a["category_name"].startswith(prefix) for prefix in s["existence_prefixes"])
               and in_scope(p, s, s["existence_distance_m"])]
    instances = sorted(a["instance_token"] for a, _ in members)
    scope = {"prefixes": s["existence_prefixes"], "radius_m": s["existence_distance_m"],
             "forward_min_m": s["forward_min_m"], "half_angle_degrees": s["front_half_angle_degrees"]}
    text = (f"Do you see any vulnerable road users within {s['existence_distance_m']:g} meters ahead of you, "
            "such as cyclists, motorcyclists, or pedestrians? "
            f"Consider the {s['front_half_angle_degrees']:g}-degree half-angle forward sector, beyond {s['forward_min_m']:g} meters forward, "
            f"and these nuScenes category prefixes: {s['existence_prefixes']}. Answer only YES or NO.")
    add("existence", text, "YES" if instances else "NO", instances,
        {"scope": scope, "members": [{"instance_token": a["instance_token"], "category": a["category_name"],
                                      "position_lidar_m": p.tolist()} for a, p in members],
         "annotation_coverage": "official_sample_annotation_table"}, [scope, instances],
        {"existence_scope:" + canonical(scope): instances})
    try:
        # current_speed's existing next-pose fallback at a scene start would leak a future observation.
        if not sample["prev"]:
            scene_name = nusc.get("scene", sample["scene_token"])["name"]
            can_bus.get_messages(scene_name, "pose")  # fail/filter rather than forward-pose fallback
        speed = current_speed(nusc, can_bus, sample)
        planning = future_positions(nusc, sample, steps=6, require_all=True)
        if planning is None:
            raise ValueError("incomplete_future_ego_trajectory")
        navigation = navigation_command(future_positions(nusc, sample, steps=12))
        trajectory = planning[1:, :2] - planning[0, :2]
        truth = SPEED_PLAN[get_obj_acc_or_dec(trajectory)] + ", " + PATH_PLAN[get_obj_turn_or_lane_change(trajectory)]
        speed_bin = int(np.searchsorted(s["current_speed_bins_mps"], speed, side="right"))
        attrs = {"navigation": navigation, "current_speed_mps": speed, "current_speed_prompt_mps": int(speed),
                 "driving_state_bin": speed_bin, "planning_label": truth,
                 "navigation_source": "offline_future_route_6s"}
        add("meta_action_planning", planning_question(speed, navigation), truth, [], attrs,
            [navigation, int(speed), speed_bin, truth],
            {"navigation": navigation, "ego_speed_prompt": int(speed), "driving_state_bin": speed_bin, "oracle_plan": truth},
            {"future_ego_lidar_positions": planning.tolist(), "label_source": "official_ego_pose_3s"})
        records["meta_action_planning"]["input_protocol"]["navigation_source"] = "offline_future_route_6s"
        records["meta_action_planning"]["input_protocol"]["navigation_is_offline_future_route_condition"] = True
    except (ValueError, KeyError, OSError) as exc:
        exclusions["meta_action_planning"] = str(exc)
    return records, exclusions


def build(nusc, can_bus, cfg, out):
    from nuscenes.utils import splits
    names = set(getattr(splits, cfg["nuscenes"]["split"]))
    scenes = sorted([s for s in nusc.scene if s["name"] in names], key=lambda s: s["name"])
    random.Random(cfg["experiment"]["seed"]).shuffle(scenes)
    scenes = sorted(scenes[:cfg["experiment"]["max_scenes"]], key=lambda s: s["name"])
    rows, excluded, sequences, base = [], [], [], []
    maximum = max(cfg["experiment"]["sequence_lengths"])
    for scene in scenes:
        chain = scene_samples(nusc, scene)
        # Start at keyframe 1, allowing current velocity from frame 0. No outcome-based window selection.
        selected = chain[1:maximum + 1]
        cache = {}
        for sample in selected:
            queries, reasons = make_queries(nusc, can_bus, sample, cfg)
            cache[sample["token"]] = queries
            base.extend(q for t, q in queries.items() if t in cfg["experiment"]["query_types"])
            for kind, reason in reasons.items():
                if kind in cfg["experiment"]["query_types"]:
                    excluded.append({"scene_token": scene["token"], "sample_token": sample["token"], "query_type": kind, "reason": reason})
        for length in cfg["experiment"]["sequence_lengths"]:
            if len(selected) < length:
                excluded.append({"scene_token": scene["token"], "sample_token": "", "query_type": "all", "reason": f"scene_shorter_than_sequence_{length}"})
                continue
            workloads = []
            if "fixed" in cfg["experiment"]["workloads"]:
                workloads += [("fixed", t) for t in cfg["experiment"]["query_types"]]
            if "mixed" in cfg["experiment"]["workloads"]:
                workloads.append(("mixed", "alternating"))
            for workload, kind in workloads:
                sequence_id = f"{scene['name']}_{length}_{workload}_{kind}"
                valid = 0
                for index, sample in enumerate(selected[:length]):
                    actual = kind if workload == "fixed" else cfg["experiment"]["query_types"][index % len(cfg["experiment"]["query_types"]) ]
                    if actual not in cache[sample["token"]]:
                        continue  # frame_index remains intact; analysis never bridges exclusions
                    q = dict(cache[sample["token"]][actual])
                    record_id = hashlib.sha256(f"{sequence_id}:{index}:{actual}".encode()).hexdigest()[:24]
                    q.update(record_id=record_id, sequence_id=sequence_id, sequence_length=length,
                             workload=workload, frame_index=index, scene_name=scene["name"])
                    rows.append(q)
                    valid += 1
                sequences.append({"sequence_id": sequence_id, "scene_token": scene["token"], "scene_name": scene["name"],
                                  "sequence_length": length, "workload": workload, "query_type": kind,
                                  "valid_queries": valid, "excluded_queries": length - valid})
    write_json(out / "queries.json", rows)
    write_json(out / "base_queries.json", base)
    write_json(out / "sequences.json", sequences)
    write_csv(out / "excluded_queries.csv", excluded, ["scene_token", "sample_token", "query_type", "reason"])
    nested = {"dependency_instance_tokens", "dependency_attributes", "dependency_signature", "state_facts", "image_paths", "input_protocol", "offline_ground_truth"}
    write_csv(out / "query_traces.csv", [{k: canonical(v) if k in nested else v for k, v in r.items()} for r in rows])
    write_csv(out / "query_dependencies.csv", [{k: canonical(r[k]) if isinstance(r[k], (dict, list)) else r[k]
              for k in ("record_id", "scene_token", "sample_token", "query_type", "dependency_instance_tokens", "dependency_signature", "dependency_attributes", "state_facts", "offline_ground_truth")} for r in rows])
    write_json(out / "query_build_summary.json", {"scenes": scenes, "sequence_count": len(sequences), "valid_query_events": len(rows),
               "unique_sample_queries": len(base), "excluded_by_reason": dict(Counter(r["reason"] for r in excluded)),
               "selection": cfg["selection"], "frame_window": "first keyframes 1..max_length, zero-based; no GT-dependent selection"})
    return rows, base, sequences


def model_sample(query):
    """Explicit allowlist: no annotation tokens, future labels or trajectories in inputs."""
    return {"token": query["sample_token"], "images": query["image_paths"],
            "conversations": [{"from": "human", "value": IMAGE_PROMPT + query["query_text"]}]}
