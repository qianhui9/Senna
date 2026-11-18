import unittest
from unittest.mock import patch
import subprocess

from experiments.motivation_exp1.analyze_object_persistence import counts
from experiments.motivation_exp1.profile_senna import profile_summary, select_samples
from experiments.motivation_exp1.tests.test_manifest import config
from experiments.motivation_exp1.utils.stats_utils import distribution, scene_bootstrap
from experiments.motivation_exp1.utils.data_utils import git_provenance
from experiments.motivation_exp1.match_visual_objects import match_objects


class AggregationTests(unittest.TestCase):
    def test_scene_weighting_not_frame_weighting(self):
        rows = [{"scene_token": "a", "metric": 0}] * 100 + [{"scene_token": "b", "metric": 1}]
        s = scene_bootstrap(rows, "metric", 42, 1000)
        self.assertEqual(s["scene_equal_weight_mean"], .5)
        self.assertEqual(s["scenes"], 2)
        self.assertEqual(s, scene_bootstrap(rows, "metric", 42, 1000))

    def test_single_scene_ci_unavailable(self):
        self.assertIsNone(scene_bootstrap([{"scene_token": "a", "metric": 1}], "metric")["ci95"])

    def test_missing_and_infinite_statistics(self):
        self.assertEqual(distribution(["", "inf", "nan", None, 1, 3])["p50"], 2)
        self.assertEqual(distribution([])["n"], 0)
        self.assertIsNone(distribution([])["p99"])

    def test_persistence_definition(self):
        s = counts({"a", "b"}, {"b", "c"})
        self.assertEqual(s["persistent_instance_count"], 1)
        self.assertEqual(s["new_instance_count"], 1)
        self.assertEqual(s["disappeared_instance_count"], 1)
        self.assertEqual(s["persistence_ratio"], .5)
        self.assertEqual(counts(set(), {"a"})["persistence_ratio"], "")

    def test_detailed_profiler_excluded(self):
        rows = [{"sample_token": "s", "measurement_mode": "timing", "timing_decomposition_valid": True, "llm_split_valid": True, "end_to_end_ms": 10, "llm_prefill_ms": 4, "llm_decode_ms": 3}, {"sample_token": "s", "measurement_mode": "detailed", "end_to_end_ms": 1000, "llm_prefill_ms": 400, "llm_decode_ms": 300}]
        self.assertEqual(profile_summary(rows)["metrics"]["end_to_end_ms"]["mean"], 10)
        self.assertEqual(profile_summary(rows)["detailed_records_excluded"], 1)
        self.assertEqual(profile_summary(rows)["metrics"]["llm_prefill_ms"]["mean"], 4)
        self.assertEqual(profile_summary(rows)["metrics"]["llm_decode_ms"]["mean"], 3)

    def test_matched_paths_required_for_linkage(self):
        cfg = config()
        records = [{"token": "s", "scene_token": "scene", "images": ["fake.jpg"] * 6}]
        selected, links = select_samples(records, [], cfg)
        self.assertEqual(len(selected), 1)
        self.assertFalse(links)

    def test_invalid_decomposition_does_not_enter_stage_summary(self):
        rows = [{"sample_token": "s", "measurement_mode": "timing", "timing_decomposition_valid": False, "llm_split_valid": False, "end_to_end_ms": 10, "vision_tower_ms": 90, "llm_prefill_ms": 80, "llm_decode_ms_per_token": 30}]
        summary = profile_summary(rows)
        self.assertEqual(summary["metrics"]["end_to_end_ms"]["mean"], 10)
        self.assertEqual(summary["metrics"]["vision_tower_ms"]["n"], 0)
        self.assertEqual(summary["metrics"]["llm_decode_ms_per_token"]["n"], 0)

    def test_global_change_joins_only_exact_same_camera_keyframes(self):
        pair = {"pair_id": "p", "scene_token": "scene", "camera_channel": "CAM_FRONT", "previous_sample_data_token": "a", "current_sample_data_token": "b", "previous_sample_token": "s1", "current_sample_token": "s2", "ssim": .99}
        frames = [{"sample_data_token": t, "sample_token": s, "is_key_frame": True, "scene_token": "scene", "camera_channel": "CAM_FRONT"} for t, s in (("a", "s1"), ("b", "s2"))]
        objects = [{"scene_token": "scene", "previous_sample_token": "s1", "current_sample_token": "s2", "previous_timestamp_us": 0, "current_timestamp_us": 500000, **counts({"x"}, {"x", "new"})}]
        rows = match_objects([pair], frames, objects)
        self.assertEqual(rows[0]["global_instance_set_changed"], 1)
        self.assertIn("GLOBAL_3D", rows[0]["annotation_scope"])
        for field, value in (("is_key_frame", False), ("scene_token", "foreign"), ("camera_channel", "CAM_BACK"), ("sample_token", "other")):
            modified = [dict(r) for r in frames]
            modified[0][field] = value
            self.assertFalse(match_objects([pair], modified, objects))

    def test_git_unavailable_does_not_block_source_provenance(self):
        with patch("experiments.motivation_exp1.utils.data_utils.subprocess.run", side_effect=subprocess.CalledProcessError(128, "git", stderr="dubious ownership")):
            result = git_provenance()
        self.assertIsNone(result["git_commit"])
        self.assertIn("dubious ownership", result["git_commit_unavailable_reason"])


if __name__ == "__main__":
    unittest.main()
