"""Dependency-aware offline oracle. No inference shortcuts are implemented."""
from collections import defaultdict
import math

import numpy as np

from .build_query_sequences import canonical
from experiments.motivation_exp1.utils.data_utils import write_csv


def overlap(previous, current):
    a, b = set(previous["dependency_instance_tokens"]), set(current["dependency_instance_tokens"])
    if not a and not b:
        return 1.0 if current["query_type"] == previous["query_type"] == "existence" else None
    return len(a & b) / len(a | b)


def prior_answer(previous, current):
    """Apply only the immediately previous request's known oracle facts."""
    facts = previous["state_facts"]
    required = current["state_facts"]
    if not set(required) <= set(facts):
        return None
    kind = current["query_type"]
    instance = next(iter(current["dependency_instance_tokens"]), None)
    if kind == "category":
        return facts[f"category:{instance}"]
    if kind == "position":
        return facts[f"position:{instance}"]
    if kind == "existence":
        return "YES" if facts[next(iter(required))] else "NO"
    if kind == "motion_prediction":
        return "Object 1: " + facts[f"oracle_prediction:{instance}"]
    return facts["oracle_plan"]


def transition(previous, current, *, comparison="workload"):
    row = {"comparison": comparison, "record_id": current.get("record_id", ""),
           "sequence_id": current.get("sequence_id", ""), "scene_token": current["scene_token"],
           "query_type": current["query_type"], "query_id": current["query_id"],
           "workload": current.get("workload", "canonical"), "sample_token": current["sample_token"],
           "previous_sample_token": previous["sample_token"] if previous else "", "delta_t_s": None,
           "answer_stable": None, "dependency_stable": None, "dependency_overlap": None,
           "same_target": None, "position_delta_m": None, "speed_delta_mps": None,
           "position_label_changed": None, "answer_same_dependency_changed": None,
           "oracle_safe_reuse": 0, "stale_answer": 0, "dependency_change": 0,
           "state_information_unavailable": 1, "has_adjacent_history": 0,
           "oracle_scope": "GT labels and dependencies; prediction/planning include offline oracle labels"}
    if previous is None:
        row["unavailable_reason"] = "no_immediately_previous_valid_query"
        return row
    if (previous["scene_token"] != current["scene_token"]
            or previous["sample_token"] != current["prev_sample_token"]
            or previous["timestamp"] >= current["timestamp"]):
        raise ValueError("Transitions must use adjacent real samples within one scene")
    row["has_adjacent_history"] = 1
    row["delta_t_s"] = (current["timestamp"] - previous["timestamp"]) / 1e6
    row["dependency_overlap"] = overlap(previous, current)
    a, b = previous["dependency_attributes"], current["dependency_attributes"]
    same_template = previous["query_id"] == current["query_id"]
    same_instances = previous["dependency_instance_tokens"] == current["dependency_instance_tokens"]
    row["same_target"] = int(same_instances) if current["dependency_instance_tokens"] or previous["dependency_instance_tokens"] else None
    if same_template:
        row["answer_stable"] = int(previous["ground_truth"] == current["ground_truth"])
        row["dependency_stable"] = int(canonical(previous["dependency_signature"]) == canonical(current["dependency_signature"]))
        row["dependency_change"] = 1 - row["dependency_stable"]
        row["answer_same_dependency_changed"] = int(row["answer_stable"] and not row["dependency_stable"])
    if same_instances and len(current["dependency_instance_tokens"]) == 1:
        if "position_lidar_m" in a and "position_lidar_m" in b:
            row["position_delta_m"] = math.dist(a["position_lidar_m"], b["position_lidar_m"])
            row["position_label_changed"] = int(a["position_label"] != b["position_label"])
        if "current_speed_mps" in a and "current_speed_mps" in b:
            row["speed_delta_mps"] = b["current_speed_mps"] - a["current_speed_mps"]
    if same_template and current["query_type"] == "meta_action_planning":
        row["speed_delta_mps"] = b["current_speed_mps"] - a["current_speed_mps"]
    historical_answer = prior_answer(previous, current)
    available = historical_answer is not None
    row["state_information_unavailable"] = int(not available)
    row["unavailable_reason"] = "" if available else "previous_query_lacks_required_fact_types_or_target"
    row["stale_answer"] = int(available and historical_answer != current["ground_truth"])
    facts_valid = available and all(canonical(previous["state_facts"][k]) == canonical(v) for k, v in current["state_facts"].items())
    row["oracle_safe_reuse"] = int(facts_valid and historical_answer == current["ground_truth"]
                                   and (not same_template or row["dependency_stable"]))
    if not same_template and available:
        row["dependency_change"] = int(not facts_valid)
    return row


def analyze(rows, base, out):
    # Canonical pairs are counted once, independent of overlapping 5/10/20 prefixes.
    groups = defaultdict(list)
    for q in base:
        groups[q["scene_token"], q["query_id"]].append(q)
    canonical_pairs = []
    for group in groups.values():
        group.sort(key=lambda q: q["timestamp"])
        for a, b in zip(group, group[1:]):
            if b["prev_sample_token"] == a["sample_token"]:
                canonical_pairs.append(transition(a, b, comparison="canonical_adjacent_template"))
    groups = defaultdict(list)
    for q in rows:
        groups[q["sequence_id"]].append(q)
    transitions = []
    for group in groups.values():
        group.sort(key=lambda q: q["frame_index"])
        previous = None
        for current in group:
            adjacent = previous is not None and current["frame_index"] == previous["frame_index"] + 1
            transitions.append(transition(previous if adjacent else None, current))
            previous = current
    write_csv(out / "dependency_transitions.csv", transitions + canonical_pairs)
    return transitions, canonical_pairs
