import unittest
import numpy as np
import pandas as pd
from experiments.motivation_exp2.utils.geometry_utils import bbox_iou, displacement
from experiments.motivation_exp2.utils.statistics_utils import metric_summary, correlation_summary
from experiments.motivation_exp2.analyze_image_displacement import compute_rows


class DisplacementTests(unittest.TestCase):
    def test_identical_box_iou_one(self):
        self.assertEqual(bbox_iou([0,0,10,10],[0,0,10,10]),1)

    def test_no_overlap_iou_zero(self):
        self.assertEqual(bbox_iou([0,0,10,10],[20,20,30,30]),0)

    def test_partial_overlap(self):
        self.assertAlmostEqual(bbox_iou([0,0,10,10],[5,0,15,10]),1/3)

    def test_displacement_3_4_5_and_real_diagonal(self):
        m = displacement([0,0,10,10],[3,4,13,14],30,40)
        self.assertEqual(m["center_displacement_px"],5)
        self.assertEqual(m["normalized_displacement"],.1)

    def test_scene_bootstrap_not_frame_bootstrap(self):
        cfg = dict(analysis=dict(seed=42,bootstrap_iterations=100))
        df = pd.DataFrame([dict(scene_token="a",x=0)]*100+[dict(scene_token="b",x=1)])
        m = metric_summary(df,"x",cfg)
        self.assertEqual(m["scene_equal_weight_mean"],.5)
        self.assertAlmostEqual(m["observation_weighted_mean"],1/101)
        self.assertEqual(m["scenes"],2)
        self.assertEqual(m["ci95"],[0,1])

    def test_one_scene_and_constant_correlation(self):
        cfg = dict(analysis=dict(seed=42,bootstrap_iterations=100))
        df = pd.DataFrame([dict(scene_token="a",x=1,y=i) for i in range(4)])
        self.assertIsNone(metric_summary(df,"x",cfg)["ci95"])
        self.assertIsNone(correlation_summary(df,"x","y",cfg)["observation_spearman"])

    def test_camera_without_projection_does_not_match(self):
        pairs = pd.DataFrame([dict(pair_id="p",sample_token_t0="a",sample_token_t1="b",scene_token="s",scene_name="scene",gap_steps=1,delta_t_seconds=.5)])
        projected = pd.DataFrame(columns=["sample_token","camera_channel","instance_token"])
        cfg = dict(dataset=dict(cameras=["CAM_FRONT"]),analysis=dict(static_speed_threshold_mps=.5))
        result, issues = compute_rows(pairs,projected,cfg)
        self.assertEqual(len(result),0)
        self.assertIn("normalized_displacement", result.columns)
