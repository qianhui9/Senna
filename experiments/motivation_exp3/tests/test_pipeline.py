import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from experiments.motivation_exp3.utils.common import parser, configuration, digest, read, save
from experiments.motivation_exp3.utils.statistics import summary, temporal_group
from experiments.motivation_exp3.aggregate_results import matched_margins, overlap_and_auc
from experiments.motivation_exp3.run_exp3 import completed


class PipelineTests(unittest.TestCase):
    def config(self, args=None):
        return configuration(parser().parse_args(args or []))

    def test_smoke_isolated_and_overrides(self):
        cfg = self.config(["--smoke-test", "--max-scenes", "8", "--max-object-pairs", "1000", "--feature-layers", "0", "12", "24", "--set", "analysis.static_speed_threshold_mps=0.2"])
        self.assertEqual(cfg["data"]["max_scenes"], 2)
        self.assertEqual(cfg["data"]["max_object_pairs"], 100)
        self.assertEqual(cfg["senna"]["feature_layers"], [0, 12, 24])
        self.assertEqual(cfg["analysis"]["static_speed_threshold_mps"], .2)
        self.assertTrue(cfg["output"]["directory"].endswith("_smoke"))

    def test_output_protection_and_original_path_required(self):
        for args in [["--output-dir", "outputs/motivation_exp2"], ["--output-dir", "experiments/motivation_exp3/dump"],
                     ["--set", "senna.use_original_preprocessing=false"], ["--set", "data.min_patch_coverage_ratio=2"]]:
            with self.assertRaises(ValueError):
                self.config(args)

    def test_scene_equal_mean_single_scene_no_fake_ci(self):
        cfg = self.config()
        frame = pd.DataFrame({"scene_token": ["a"]*10+["b"], "score": [0.]*10+[1.]})
        stats = summary(frame, "score", cfg)
        self.assertAlmostEqual(stats["mean"], 1/11)
        self.assertEqual(stats["scene_equal_weight_mean"], .5)
        self.assertIsNotNone(stats["ci95_low"])
        one = summary(frame.iloc[:10], "score", cfg)
        self.assertIsNone(one["ci95_low"])
        self.assertIsNone(one["ci95_high"])

    def test_resume_requires_matching_artifact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"test.csv"
            p.write_text("real file fixture")
            state = {"stage": {"artifacts": {str(p): digest(p)}}}
            self.assertTrue(completed("stage", Path(tmp), state))
            p.write_text("corruption")
            self.assertFalse(completed("stage", Path(tmp), state))

    def test_time_gap_labels_stay_strings_with_numeric_only_or_mixed_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            for values in [["0.5", "1.0", "2.0"], ["0.5"]*70000+["other"]]:
                save(Path(tmp), "groups", pd.DataFrame({"time_gap_group": values}))
                result = read(Path(tmp), "groups")
                self.assertTrue(all(isinstance(x, str) for x in result.time_gap_group.unique()))
                self.assertEqual(result.time_gap_group.tolist(), values)

    def test_matched_margins_require_same_anchor_layer_and_method(self):
        records = [{"object_pair_id": p, "layer_id": "layer_23", "aggregation": "area", "comparison_group": g,
                    "cosine_similarity": v} for p,g,v in [("a", "B1_same_instance", .8),
                    ("a", "B2_different_instance_same_category", .5), ("b", "B1_same_instance", .2)]]
        x = matched_margins(pd.DataFrame(records))
        self.assertEqual(len(x), 1)
        self.assertAlmostEqual(x.iloc[0].delta_similarity, .3)

    def test_overlap_auc_and_actual_gap(self):
        self.assertEqual(overlap_and_auc([.9, .8], [.2, .1])["descriptive_auc"], 1)
        self.assertEqual(overlap_and_auc([.5, .5], [.5, .5])["descriptive_auc"], .5)
        self.assertIsNone(overlap_and_auc([], [1])["descriptive_auc"])
        self.assertEqual(temporal_group(.501, self.config()), "0.5")
        self.assertEqual(temporal_group(4, self.config()), "other")
