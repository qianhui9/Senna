"""Build planning-only nuScenes evaluation inputs with explicit conventions."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm

from data_tools.senna_inputs import (
    CAMERAS, LEGACY_CAMERAS, IMAGE_PROMPT, PROTOCOL_VERSION,
    current_speed, future_positions, navigation_command, planning_question, sample_images,
)
from data_tools.senna_qa_utils import get_obj_acc_or_dec, get_obj_turn_or_lane_change

SPEED_PLAN = {"const": "KEEP", "accelerate": "ACCELERATE", "decelerate": "DECELERATE", "stop": "STOP"}
PATH_PLAN = {"right turn": "RIGHT_TURN", "right lane change": "RIGHT_CHANGE",
             "left turn": "LEFT_TURN", "left lane change": "LEFT_CHANGE", "straight": "STRAIGHT"}


class CachedCanBus(NuScenesCanBus):
    """Avoid rereading a scene's CAN file for every frame; handle known missing scenes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.message_cache = {}

    def get_messages(self, scene_name, message_name, *args, **kwargs):
        if int(scene_name.rsplit("-", 1)[-1]) in self.can_blacklist:
            # The SDK raises a generic Exception here. Convert only this known case
            # to the existing ego-pose speed fallback, not arbitrary SDK errors.
            raise OSError(f"No CAN bus data for {scene_name}")
        key = (scene_name, message_name)
        if key not in self.message_cache:
            self.message_cache[key] = super().get_messages(scene_name, message_name, *args, **kwargs)
        return self.message_cache[key]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="/data/qh/nuscenes/v1.0-trainval")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output", default="data/senna_nusc_val_plan_corrected.json")
    parser.add_argument("--legacy-navigation", action="store_true", help="For input ablations only")
    parser.add_argument("--legacy-camera-order", action="store_true", help="For input ablations only")
    parser.add_argument("--per-class", type=int, help="Deterministic subset per joint GT class for smoke tests")
    return parser.parse_args()


def build_records(args):
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    can_bus = CachedCanBus(dataroot=args.dataroot)
    scene_names = set(getattr(splits, args.split))
    scene_tokens = {scene["token"] for scene in nusc.scene if scene["name"] in scene_names}
    samples = [sample for sample in nusc.sample if sample["scene_token"] in scene_tokens]
    records = []
    for sample in tqdm(samples, desc=f"Building nuScenes {args.split} planning set"):
        planning_positions = future_positions(nusc, sample, steps=6, require_all=True)
        if planning_positions is None:
            continue
        navigation = navigation_command(future_positions(nusc, sample, steps=12), legacy=args.legacy_navigation)
        speed = current_speed(nusc, can_bus, sample)
        # Keep the original GT label algorithm; input fixes must not change targets.
        trajectory = planning_positions[1:, :2] - planning_positions[0, :2]
        answer = SPEED_PLAN[get_obj_acc_or_dec(trajectory)] + ", " + PATH_PLAN[get_obj_turn_or_lane_change(trajectory)]
        images = sample_images(nusc, sample, legacy=args.legacy_camera_order)
        records.append({
            "token": sample["token"], "image": images[0], "images": images,
            "prev": sample["prev"], "next": sample["next"], "scene_token": sample["scene_token"],
            "speed_mps": speed, "navigation": navigation,
            "input_protocol": {
                "version": PROTOCOL_VERSION,
                "camera_order": list(LEGACY_CAMERAS if args.legacy_camera_order else CAMERAS),
                "navigation_source": "offline_future_route_6s",
                "legacy_navigation": args.legacy_navigation,
                "legacy_camera_order": args.legacy_camera_order,
            },
            "conversations": [
                {"from": "human", "value": IMAGE_PROMPT + planning_question(speed, navigation)},
                {"from": "gpt", "value": answer + "\n"},
            ],
        })
    if args.per_class is not None:
        if args.per_class <= 0:
            raise ValueError("--per-class must be positive")
        groups = defaultdict(list)
        for record in records:
            groups[record["conversations"][1]["value"].strip()].append(record)
        selected = set()
        for group in groups.values():
            count = min(len(group), args.per_class)
            selected.update(group[index * len(group) // count]["token"] for index in range(count))
        records = [record for record in records if record["token"] in selected]
    return records


def main():
    args = parse_args()
    records = build_records(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} planning samples to {output}")
    print("Navigation distribution:", dict(Counter(record["navigation"] for record in records)))
    print("GT distribution:", dict(Counter(record["conversations"][1]["value"].strip() for record in records)))


if __name__ == "__main__":
    main()
