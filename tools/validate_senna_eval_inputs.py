"""Validate corrected data and produce deterministic, matched input ablations."""

import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from data_tools.senna_inputs import LEGACY_CAMERAS, planning_question, IMAGE_PROMPT, validate_sample


def main():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "data/senna_nusc_val_plan.json").read_text())
    new = json.loads((root / "data/senna_nusc_val_plan_corrected.json").read_text())
    original = {record["token"]: record for record in old}
    assert len(old) == len(new) == len(original) == 5119
    assert set(original) == {record["token"] for record in new}
    groups = defaultdict(list)
    for record in new:
        validate_sample(record)
        previous = original[record["token"]]
        assert record["conversations"][1] == previous["conversations"][1], record["token"]
        assert set(record["images"]) == set(previous["images"]), record["token"]
        groups[record["conversations"][1]["value"].strip()].append(record)
    chosen = set()
    for group in groups.values():
        count = min(20, len(group))
        chosen.update(group[index * len(group) // count]["token"] for index in range(count))
    corrected = [record for record in new if record["token"] in chosen]
    variants = {"corrected": corrected}
    # Same tokens, targets and speed inputs in every ablation; only one input factor varies.
    for name, old_navigation, old_images in (("legacy_inputs", True, True),
                                              ("navigation_only", False, True),
                                              ("cameras_only", True, False)):
        records = copy.deepcopy(corrected)
        for record in records:
            prior = original[record["token"]]
            if old_navigation:
                record["navigation"] = re.search(r"navigation command is '([^']+)'", prior["conversations"][0]["value"]).group(1)
                record["conversations"][0]["value"] = IMAGE_PROMPT + planning_question(record["speed_mps"], record["navigation"])
            if old_images:
                record["images"] = prior["images"]
                record["input_protocol"]["camera_order"] = list(LEGACY_CAMERAS)
            record["input_protocol"]["legacy_navigation"] = old_navigation
            record["input_protocol"]["legacy_camera_order"] = old_images
        variants[name] = records
    for name, records in variants.items():
        path = root / "data" / f"senna_val_check127_{name}.json"
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(records)} samples -> {path}")
    print("Full input validation passed: GT unchanged, token set unchanged, all camera paths aligned.")
    print("Navigation counts:", dict(Counter(record["navigation"] for record in new)))
    print("Check subset GT counts:", dict(Counter(record["conversations"][1]["value"].strip() for record in corrected)))


if __name__ == "__main__":
    main()
