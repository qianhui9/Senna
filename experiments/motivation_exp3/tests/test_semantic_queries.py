"""Semantic-query invariants using controlled annotations, never synthetic result claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.motivation_exp3.analyze_query_dependencies import analyze, transition
from experiments.motivation_exp3.build_query_sequences import (
    canonical, future_object, in_scope, make_queries, model_sample, past_speed,
    position_label, scene_samples,
)
from experiments.motivation_exp3.profile_queries import load_journal, strict_correct
from experiments.motivation_exp3.run_queries import configuration, initialize, parser, gpu_is_idle, wait_for_idle
from experiments.motivation_exp3.query_reporting import aggregate
from experiments.motivation_exp1.utils.data_utils import digest, write_json


def configuration_for_test():
    return configuration(parser().parse_args([]))


def query(kind="category", instance="a", truth="vehicle.car", sample="s1", prev="s0", timestamp=1_000_000):
    facts = {"selected_vehicle": instance, f"category:{instance}": truth}
    attrs = {"position_lidar_m": [0, 10, 0], "position_label": "FRONT"}
    signature = [instance, truth]
    instances = [instance]
    if kind == "existence":
        instances = [] if truth == "NO" else [instance]
        facts = {"scope": instances}
        signature = ["scope", instances]
    if kind == "position":
        facts = {"selected_vehicle": instance, f"position:{instance}": truth}
    return {"scene_token": "scene", "sample_token": sample, "prev_sample_token": prev,
            "timestamp": timestamp, "query_type": kind, "query_id": kind + "_v1", "ground_truth": truth,
            "dependency_instance_tokens": instances, "dependency_signature": signature, "state_facts": facts,
            "dependency_attributes": attrs, "query_text": "current query", "image_paths": ["a"] * 6,
            "offline_ground_truth": {"future": "sentinel"}, "input_protocol": {"future_label_in_prompt": False}}


def after(previous, **changes):
    result = copy.deepcopy(previous)
    result.update(sample_token="s2", prev_sample_token=previous["sample_token"], timestamp=previous["timestamp"] + 500_000)
    result.update(changes)
    return result


class DependencyTests(unittest.TestCase):
    def test_same_answer_new_instance_is_not_reuse(self):
        a, b = query(), query(instance="b", sample="s2", prev="s1", timestamp=1_500_000)
        r = transition(a, b)
        self.assertEqual(r["answer_stable"], 1)
        self.assertEqual(r["dependency_stable"], 0)
        self.assertEqual(r["answer_same_dependency_changed"], 1)
        self.assertEqual(r["oracle_safe_reuse"], 0)
        self.assertEqual(r["dependency_overlap"], 0)

    def test_stable_facts_reusable(self):
        a = query()
        r = transition(a, after(a))
        self.assertEqual(r["oracle_safe_reuse"], 1)
        self.assertEqual(r["delta_t_s"], .5)

    def test_negative_existence_requires_same_set(self):
        a = query("existence", truth="NO")
        self.assertEqual(transition(a, after(a))["oracle_safe_reuse"], 1)
        b = query("existence", truth="YES", sample="s2", prev="s1", timestamp=1_500_000)
        r = transition(a, b)
        self.assertEqual(r["stale_answer"], 1)
        self.assertEqual(r["oracle_safe_reuse"], 0)

    def test_same_existence_answer_changed_support_is_not_safe(self):
        a = query("existence", truth="YES")
        b = query("existence", instance="b", truth="YES", sample="s2", prev="s1", timestamp=1_500_000)
        self.assertEqual(transition(a, b)["oracle_safe_reuse"], 0)

    def test_mixed_query_missing_fact_type(self):
        a = query()
        b = query("position", truth="FRONT", sample="s2", prev="s1", timestamp=1_500_000)
        r = transition(a, b)
        self.assertEqual(r["state_information_unavailable"], 1)
        self.assertIsNone(r["answer_stable"])
        self.assertEqual(r["dependency_overlap"], 1)

    def test_continuous_position_changes_remain_visible(self):
        a = query("position", truth="FRONT")
        b = after(a)
        b["dependency_attributes"]["position_lidar_m"] = [0, 11, 0]
        r = transition(a, b)
        self.assertEqual(r["position_delta_m"], 1)
        self.assertEqual(r["dependency_stable"], 1)

    def test_cross_scene_or_nonadjacent_rejected(self):
        a = query()
        with self.assertRaises(ValueError):
            transition(a, after(a, scene_token="other"))
        with self.assertRaises(ValueError):
            transition(a, after(a, prev_sample_token="missing"))

    def test_missing_frame_not_bridged_or_canonical_double_counted(self):
        a = query(sample="s1", prev="s0")
        b = query(sample="s3", prev="s2", timestamp=2_000_000)
        rows = []
        for sequence in ("short", "long"):
            rows.extend([{**a, "sequence_id": sequence, "frame_index": 0}, {**b, "sequence_id": sequence, "frame_index": 2}])
        with tempfile.TemporaryDirectory() as d:
            transitions, pairs = analyze(rows, [a, b], Path(d))
        self.assertEqual(pairs, [])
        self.assertTrue(all(r["has_adjacent_history"] == 0 for r in transitions))

    def test_first_request_is_unavailable(self):
        self.assertEqual(transition(None, query())["state_information_unavailable"], 1)


class ConstructionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = configuration_for_test()
        self.selection = self.cfg["selection"]

    def test_position_edges_and_range(self):
        self.assertEqual(position_label([-2, 10], self.selection), "FRONT")
        self.assertEqual(position_label([2, 10], self.selection), "FRONT_RIGHT")
        self.assertFalse(in_scope([0, -1], self.selection, 40))
        self.assertFalse(in_scope([0, 40], self.selection, 40))
        self.assertTrue(in_scope([0, 20], self.selection, 40))

    def test_model_input_excludes_offline_metadata(self):
        a = query()
        first = model_sample(a)
        a["offline_ground_truth"] = {"future": [12345]}
        a["ground_truth"] = "SENTINEL_LEAK"
        a["dependency_signature"] = ["SENTINEL_LEAK"]
        self.assertEqual(first, model_sample(a))
        self.assertNotIn("SENTINEL_LEAK", canonical(model_sample(a)))
        self.assertEqual(first["conversations"][0]["value"].count("<image>"), 6)

    def test_current_velocity_uses_previous_annotation(self):
        previous = {"instance_token": "a", "translation": [0, 8, 0], "sample_token": "s0"}
        nusc = Mock()
        nusc.get.side_effect = lambda table, token: previous if table == "sample_annotation" else {"scene_token": "scene", "timestamp": 0}
        current = {"instance_token": "a", "translation": [0, 10, 0], "prev": "ann0"}
        sample = {"scene_token": "scene", "timestamp": 500_000, "prev": "s0"}
        self.assertEqual(past_speed(nusc, current, sample), 4)
        current["prev"] = ""
        with self.assertRaises(ValueError):
            past_speed(nusc, current, sample)

    def test_future_chain_requires_same_instance_every_keyframe(self):
        annotation = {"instance_token": "a", "token": "ann0", "next": "ann1"}
        sample = {"token": "s0", "next": "s1", "scene_token": "scene", "timestamp": 0}
        nusc = Mock()
        nusc.get.side_effect = lambda table, token: ({"instance_token": "other", "prev": "ann0", "sample_token": "s1"}
              if table == "sample_annotation" else {"token": "s1", "scene_token": "scene", "timestamp": 500_000})
        with self.assertRaisesRegex(ValueError, "invalid_future_object"):
            future_object(nusc, annotation, sample, self.selection)

    def test_scene_chain_cannot_cross_scene(self):
        nusc = Mock()
        nusc.get.return_value = {"scene_token": "wrong", "prev": ""}
        with self.assertRaises(ValueError):
            scene_samples(nusc, {"token": "scene", "first_sample_token": "s0"})

    def test_empty_annotations_are_real_existence_negative(self):
        sample = {"token": "s1", "scene_token": "scene", "prev": "s0", "next": "s2", "timestamp": 1_000_000, "anns": []}
        with patch("experiments.motivation_exp3.build_query_sequences.sample_images", return_value=["a"] * 6), \
             patch("experiments.motivation_exp3.build_query_sequences.current_speed", return_value=2), \
             patch("experiments.motivation_exp3.build_query_sequences.future_positions", return_value=np.array([[0, i, 0] for i in range(7)])):
            result, excluded = make_queries(Mock(), Mock(), sample, self.cfg)
        self.assertEqual(result["existence"]["ground_truth"], "NO")
        self.assertEqual(result["existence"]["dependency_instance_tokens"], [])
        self.assertEqual(excluded["category"], "no_deterministic_vehicle_target")

    def test_nearest_target_tie_break_is_deterministic(self):
        sample = {"token": "s1", "scene_token": "scene", "prev": "s0", "next": "s2", "timestamp": 1_000_000, "anns": ["ann_b", "ann_a"]}
        annotations = {"ann_" + instance: {"token": "ann_" + instance, "sample_token": "s1", "instance_token": instance,
                       "category_name": "vehicle.car", "translation": [0, 10, 0], "num_lidar_pts": 5, "num_radar_pts": 0, "prev": ""}
                       for instance in ("a", "b")}
        nusc = Mock()
        nusc.get.side_effect = lambda table, token: annotations[token]
        with patch("experiments.motivation_exp3.build_query_sequences.sample_images", return_value=["a"] * 6), \
             patch("experiments.motivation_exp3.build_query_sequences.lidar_coordinates", return_value=np.array([[0, 10, 0], [0, 10, 0]])), \
             patch("experiments.motivation_exp3.build_query_sequences.current_speed", return_value=2), \
             patch("experiments.motivation_exp3.build_query_sequences.future_positions", return_value=np.array([[0, i, 0] for i in range(7)])):
            result, _ = make_queries(nusc, Mock(), sample, self.cfg)
        self.assertEqual(result["category"]["dependency_instance_tokens"], ["a"])

    def test_future_motion_labels_do_not_change_current_prompt(self):
        sample = {"token": "s1", "scene_token": "scene", "prev": "s0", "next": "s2", "timestamp": 1_000_000, "anns": ["ann"]}
        annotation = {"token": "ann", "sample_token": "s1", "instance_token": "a", "category_name": "vehicle.car",
                      "translation": [0, 10, 0], "num_lidar_pts": 5, "num_radar_pts": 0}
        nusc = Mock()
        nusc.get.return_value = annotation
        with patch("experiments.motivation_exp3.build_query_sequences.sample_images", return_value=["a"] * 6), \
             patch("experiments.motivation_exp3.build_query_sequences.lidar_coordinates", return_value=np.array([[0, 10, 0]])), \
             patch("experiments.motivation_exp3.build_query_sequences.past_speed", return_value=2), \
             patch("experiments.motivation_exp3.build_query_sequences.current_speed", return_value=2), \
             patch("experiments.motivation_exp3.build_query_sequences.future_positions", return_value=np.array([[0, i, 0] for i in range(7)])), \
             patch("experiments.motivation_exp3.build_query_sequences.future_object") as future:
            future.return_value = (np.zeros((6, 2)), {"future_sentinel": 111})
            first, _ = make_queries(nusc, Mock(), sample, self.cfg)
            future.return_value = (np.array([[0, 3 * i] for i in range(6)]), {"future_sentinel": 999})
            second, _ = make_queries(nusc, Mock(), sample, self.cfg)
        self.assertEqual(model_sample(first["motion_prediction"]), model_sample(second["motion_prediction"]))
        self.assertNotEqual(first["motion_prediction"]["ground_truth"], second["motion_prediction"]["ground_truth"])

    def test_corrupt_annotation_does_not_become_negative(self):
        nusc = Mock()
        nusc.get.return_value = {"sample_token": "wrong"}
        sample = {"token": "s1", "anns": ["bad"]}
        with patch("experiments.motivation_exp3.build_query_sequences.sample_images", return_value=["a"] * 6):
            result, excluded = make_queries(nusc, Mock(), sample, self.cfg)
        self.assertFalse(result)
        self.assertTrue(all("incomplete_current_annotations" in v for v in excluded.values()))


class PersistenceAndScoringTests(unittest.TestCase):
    def test_plan_scoring_is_strict(self):
        q = query("meta_action_planning", truth="KEEP, STRAIGHT")
        self.assertTrue(strict_correct("KEEP, STRAIGHT.", q))
        self.assertFalse(strict_correct("I think KEEP, STRAIGHT or STOP, STRAIGHT", q))
        q = query("motion_prediction", truth="Object 1: KEEP, STRAIGHT")
        self.assertTrue(strict_correct("Object 1: KEEP, STRAIGHT", q))
        self.assertFalse(strict_correct("KEEP, STRAIGHT", q))

    def test_interrupted_journal_recovers_complete_lines_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "journal.jsonl"
            path.write_text('{"record_id":"a","repeat":0}\n{"unfinished":')
            self.assertEqual(len(load_journal(path)), 1)
            self.assertTrue(path.read_text().endswith("\n"))
            path.write_text('{"record_id":"a","repeat":0}\n' * 2)
            with self.assertRaises(ValueError):
                load_journal(path)

    def test_resume_rejects_modified_build_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            args = parser().parse_args(["--output-dir", str(Path(d) / "out")])
            cfg = configuration(args)
            with patch("experiments.motivation_exp3.run_queries.identity", return_value={"sha256": "test"}):
                out, manifest, lock = initialize(cfg, args)
                (out / "queries.json").write_text("[]")
                manifest["artifacts"]["queries.json"] = digest(out / "queries.json")
                write_json(out / "query_run_manifest.json", manifest)
                lock.close()
                (out / "queries.json").write_text("[1]")
                with self.assertRaisesRegex(ValueError, "Modified"):
                    initialize(cfg, args)

    def test_unrelated_output_directory_protected(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "old_report.md").write_text("keep")
            args = parser().parse_args(["--output-dir", d, "--existing", "overwrite"])
            with patch("experiments.motivation_exp3.run_queries.identity", return_value={"sha256": "test"}):
                with self.assertRaisesRegex(ValueError, "unrelated"):
                    initialize(configuration(args), args)
            self.assertEqual(Path(d, "old_report.md").read_text(), "keep")

    def test_idle_wait_requires_two_snapshots_and_never_loads_model(self):
        busy = "0, uuid, GPU, 600 MiB, 24000 MiB, 0 %, 50, 10 W\n1, uuid, GPU, 600 MiB, 24000 MiB, 0 %, 50, 10 W"
        idle = busy.replace("600 MiB", "100 MiB")
        self.assertFalse(gpu_is_idle(busy, 1))
        self.assertTrue(gpu_is_idle(idle, 1))
        args = parser().parse_args(["--wait-for-idle"])
        with tempfile.TemporaryDirectory() as d, \
             patch("experiments.motivation_exp3.run_queries.hardware_snapshot", side_effect=[busy, idle, idle]), \
             patch("experiments.motivation_exp3.run_queries.time.sleep") as sleep:
            wait_for_idle(configuration(args), args, Path(d))
            state = json.loads(Path(d, "query_invocation_status.json").read_text())
        self.assertEqual(state["consecutive_idle_snapshots"], 2)
        self.assertEqual(sleep.call_count, 2)

    def test_wait_and_shared_policies_cannot_be_combined(self):
        with self.assertRaises(SystemExit):
            parser().parse_args(["--wait-for-idle", "--allow-shared-gpu"])

    def test_invalid_config_rejected(self):
        args = parser().parse_args(["--sequence-lengths", "5", "5"])
        with self.assertRaises(ValueError):
            configuration(args)


if __name__ == "__main__":
    unittest.main()
