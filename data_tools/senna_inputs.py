"""Shared nuScenes conventions for Senna inference and data preparation."""

import math
import re
from pathlib import Path

import numpy as np
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

PROTOCOL_VERSION = "nuscenes_senna_v2"
CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK")
LEGACY_CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
VIEWS = ("FRONT", "FRONT LEFT", "FRONT RIGHT", "BACK LEFT", "BACK RIGHT", "BACK")
IMAGE_PROMPT = "".join(f"<{view} VIEW>:\n<image>\n" for view in VIEWS)
SPEED_PLANS = ("KEEP", "ACCELERATE", "DECELERATE", "STOP")
PATH_PLANS = ("RIGHT_TURN", "RIGHT_CHANGE", "LEFT_TURN", "LEFT_CHANGE", "STRAIGHT")
PLAN_PATTERN = re.compile(r"\s*(" + "|".join(SPEED_PLANS) + r")\s*,\s*(" + "|".join(PATH_PLANS) + r")\s*\.?\s*")


def parse_plan(answer):
    """Reject ambiguous actions instead of awarding substring matches."""
    match = PLAN_PATTERN.fullmatch(answer)
    return match.groups() if match else (None, None)


def planning_question(speed, navigation):
    return (
        f"Your current speed is {int(speed)} m/s, the navigation command is '{navigation}', "
        "based on the understanding of the driving scene and the navigation information, "
        "what is your plan for the next three seconds? Please answer your SPEED plan and "
        "your PATH plan. SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, PATH "
        "includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. For example, "
        "a correct answer format is like 'KEEP, LEFT_CHANGE'."
    )


def sensor_position(nusc, sample):
    lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pose = nusc.get("ego_pose", lidar["ego_pose_token"])
    calibrated = nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])
    global_from_ego = transform_matrix(pose["translation"], Quaternion(pose["rotation"]))
    ego_from_sensor = transform_matrix(calibrated["translation"], Quaternion(calibrated["rotation"]))
    return (global_from_ego @ ego_from_sensor)[:3, 3], pose, calibrated


def future_positions(nusc, sample, steps, require_all=False):
    """Sensor positions in the current LiDAR frame: x right, y forward."""
    positions = np.zeros((steps + 1, 3), dtype=np.float64)
    current = sample
    for index in range(steps + 1):
        positions[index] = sensor_position(nusc, current)[0]
        if index == steps:
            break
        if not current["next"]:
            if require_all:
                return None
            positions[index + 1:] = positions[index]
            break
        current = nusc.get("sample", current["next"])
    _, pose, calibrated = sensor_position(nusc, sample)
    positions -= np.asarray(pose["translation"])
    positions = (Quaternion(pose["rotation"]).inverse.rotation_matrix @ positions.T).T
    positions -= np.asarray(calibrated["translation"])
    return (Quaternion(calibrated["rotation"]).inverse.rotation_matrix @ positions.T).T


def navigation_command(positions, legacy=False):
    """Use longitudinal and signed lateral distances in the LiDAR frame."""
    lateral_right, forward = positions[-1, :2]
    if legacy:
        forward, lateral_left = lateral_right, forward
    else:
        lateral_left = -lateral_right
    if lateral_left >= 10.0:
        return "go straight and turn left" if forward >= 20.0 else "turn left"
    if lateral_left <= -10.0:
        return "go straight and turn right" if forward >= 20.0 else "turn right"
    return "go straight"


def current_speed(nusc, can_bus, sample):
    """Nearest CAN bus pose, matching the upstream locate_message function."""
    scene_name = nusc.get("scene", sample["scene_token"])["name"]
    try:
        messages = can_bus.get_messages(scene_name, "pose")
        times = [message["utime"] for message in messages]
        index = int(np.searchsorted(times, sample["timestamp"]))
        if index == len(times) or (index > 0 and sample["timestamp"] - times[index - 1] < times[index] - sample["timestamp"]):
            index -= 1
        return float(messages[index]["vel"][0])
    except (OSError, ValueError, KeyError, IndexError):
        neighbor_token = sample["prev"] or sample["next"]
        if not neighbor_token:
            raise ValueError(f"Cannot infer speed for isolated sample {sample['token']}")
        neighbor = nusc.get("sample", neighbor_token)
        _, pose, _ = sensor_position(nusc, sample)
        _, neighbor_pose, _ = sensor_position(nusc, neighbor)
        seconds = abs(sample["timestamp"] - neighbor["timestamp"]) / 1_000_000
        if seconds <= 0:
            raise ValueError("Non-increasing nuScenes timestamps")
        return math.dist(pose["translation"][:2], neighbor_pose["translation"][:2]) / seconds


def sample_images(nusc, sample, legacy=False):
    cameras = LEGACY_CAMERAS if legacy else CAMERAS
    images = [nusc.get_sample_data_path(sample["data"][camera]) for camera in cameras]
    missing = [image for image in images if not Path(image).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing camera images: {missing}")
    return images


def validate_sample(sample, allow_legacy=False):
    """Preflight inputs before loading the GPU checkpoint."""
    question = sample["conversations"][0]["value"]
    if question.count("<image>") != 6 or len(sample["images"]) != 6:
        raise ValueError(f"{sample['token']}: expected six images and placeholders")
    if not question.startswith(IMAGE_PROMPT):
        raise ValueError(f"{sample['token']}: unsupported view prompt order")
    if parse_plan(sample["conversations"][1]["value"]) == (None, None):
        raise ValueError(f"{sample['token']}: invalid ground-truth action")
    if not allow_legacy:
        protocol = sample.get("input_protocol", {})
        if (protocol.get("version") != PROTOCOL_VERSION or protocol.get("legacy_navigation") or protocol.get("legacy_camera_order")):
            raise ValueError("Legacy/unversioned JSON. Generate *_corrected.json first; "
                             "use --allow-legacy-inputs only for a controlled baseline.")
        actual = tuple(Path(image).parent.name for image in sample["images"])
        if actual != CAMERAS:
            raise ValueError(f"{sample['token']}: camera order {actual} != {CAMERAS}")
    for image in sample["images"]:
        if not Path(image).is_file():
            raise FileNotFoundError(image)


def check_model(model_path):
    model_path = Path(model_path)
    for name in ("config.json", "pytorch_model.bin.index.json", "tokenizer.model", "tokenizer_config.json"):
        if not (model_path / name).is_file():
            raise FileNotFoundError(model_path / name)
    sizes = {"pytorch_model-00001-of-00002.bin": 9_976_631_486,
             "pytorch_model-00002-of-00002.bin": 4_149_654_020}
    for name, size in sizes.items():
        path = model_path / name
        with path.open("rb") as file:
            header = file.read(2)
        if path.stat().st_size != size or header != b"PK":
            raise ValueError(f"Incomplete or invalid checkpoint shard: {path}")
