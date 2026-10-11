import unittest

from experiments.motivation_exp1.analyze_object_persistence import counts
from experiments.motivation_exp1.profile_senna import profile_summary, select_samples
from experiments.motivation_exp1.tests.test_manifest import config
from experiments.motivation_exp1.utils.stats_utils import distribution, scene_bootstrap


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
        rows = [{"sample_token": "s", "measurement_mode": "timing", "end_to_end_ms": 10}, {"sample_token": "s", "measurement_mode": "detailed", "end_to_end_ms": 1000}]
        self.assertEqual(profile_summary(rows)["metrics"]["end_to_end_ms"]["mean"], 10)
        self.assertEqual(profile_summary(rows)["detailed_records_excluded"], 1)

    def test_matched_paths_required_for_linkage(self):
        cfg = config()
        records = [{"token": "s", "scene_token": "scene", "images": ["fake.jpg"] * 6}]
        selected, links = select_samples(records, [], cfg)
        self.assertEqual(len(selected), 1)
        self.assertFalse(links)


if __name__ == "__main__":
    unittest.main()
