import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image

from experiments.motivation_exp3.build_feature_pairs import build_pairs, stratified_sample
from experiments.motivation_exp3.analyze_feature_similarity import choose_negative
from experiments.motivation_exp3.utils.common import parser, configuration


class PairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = configuration(parser().parse_args(["--output-dir", self.tmp.name]))
        self.projections, self.frames = [], []
        self.match = {"pair_id": "p", "scene_name": "s", "scene_token": "scene", "camera_channel": "CAM_FRONT",
                      "instance_token": "official", "category_name": "vehicle.car", "camera_delta_t_seconds": .5,
                      "delta_t_seconds": .5, "normalized_displacement": .1}
        for t in [0, 1]:
            image = Path(self.tmp.name)/f"{t}.png"
            Image.new("RGB", (100, 50)).save(image)
            obj = {"scene_name": "s", "scene_token": "scene", "camera_channel": "CAM_FRONT", "instance_token": "official",
                   "category_name": "vehicle.car", "sample_token": f"sample{t}", "sample_data_token": f"frame{t}",
                   "box_xmin": 10, "box_ymin": 10, "box_xmax": 40, "box_ymax": 40,
                   "projection_status": "valid", "geometric_in_image_area_ratio": 1.0}
            frame = {k: obj[k] for k in ["scene_name", "scene_token", "camera_channel", "sample_token", "sample_data_token"]}
            frame.update({"image_path": str(image), "camera_timestamp": 1000000+500000*t, "sample_timestamp": 1000010+500000*t})
            self.projections.append(obj)
            self.frames.append(frame)
            self.match.update({f"sample_token_t{t}": f"sample{t}", f"sample_data_token_t{t}": f"frame{t}"})
            for k in ["box_xmin", "box_ymin", "box_xmax", "box_ymax"]:
                self.match[f"{k}_t{t}"] = obj[k]
        self.keyframe = {"pair_id": "p", "scene_token": "scene", "sample_token_t0": "sample0", "sample_token_t1": "sample1", "timestamp_t0": 1000010, "timestamp_t1": 1500010}

    def build(self, matches=None, objects=None, frames=None, pairs=None):
        return build_pairs(pd.DataFrame(matches or [self.match]), pd.DataFrame(objects or self.projections),
                           pd.DataFrame(frames or self.frames), pd.DataFrame(pairs or [self.keyframe]), self.cfg)

    def test_official_match_timestamps_and_metadata(self):
        rows, issues = self.build()
        self.assertEqual(len(rows), 1)
        self.assertFalse(issues)
        self.assertEqual(rows.iloc[0].instance_token, "official")
        self.assertEqual(rows.iloc[0].timestamp_t1, 1500000)
        self.assertEqual(json.loads(rows.iloc[0].bbox_t0), [10, 10, 40, 40])

    def test_different_scene_camera_or_identity_never_same_pair(self):
        for field, value in [("scene_token", "other"), ("camera_channel", "CAM_BACK"), ("instance_token", "other")]:
            objs = copy.deepcopy(self.projections)
            objs[1][field] = value
            rows, issues = self.build(objects=objs)
            self.assertEqual(len(rows), 0)
            self.assertTrue(issues)

    def test_invalid_projection_small_box_and_missing_image(self):
        objects = copy.deepcopy(self.projections)
        objects[0]["projection_status"] = "outside_image"
        self.assertEqual(len(self.build(objects=objects)[0]), 0)
        self.cfg["data"]["min_bbox_width_px"] = 40
        self.assertEqual(len(self.build()[0]), 0)
        self.cfg["data"]["min_bbox_width_px"] = 16
        Path(self.frames[0]["image_path"]).unlink()
        self.assertEqual(len(self.build()[0]), 0)

    def test_time_order_and_no_duplicate_count(self):
        frames = copy.deepcopy(self.frames)
        frames[1]["camera_timestamp"] = 900000
        self.assertEqual(len(self.build(frames=frames)[0]), 0)
        rows, excluded = self.build(matches=[self.match, self.match])
        self.assertEqual(len(rows), 1)
        self.assertEqual(excluded[0]["reason"], "duplicate_pair")

    def test_bbox_or_category_contradiction_fails(self):
        obj = copy.deepcopy(self.projections)
        obj[1]["box_xmin"] = 20
        with self.assertRaises(ValueError):
            self.build(objects=obj)
        obj = copy.deepcopy(self.projections)
        obj[1]["category_name"] = "human.pedestrian"
        with self.assertRaises(ValueError):
            self.build(objects=obj)

    def test_negative_control_constraints_and_reproducibility(self):
        anchor = {"scene_token": "s", "camera_channel": "CAM_FRONT", "instance_token": "a", "category_name": "car"}
        target = {"bbox_area": 100, "camera_distance_m": 20}
        base = {**anchor, "instance_token": "b", "bbox_area": 110, "camera_distance_m": 22, "observation_id": "good"}
        candidates = pd.DataFrame([base, {**base, "scene_token": "other", "observation_id": "cross_scene"},
                                   {**base, "bbox_area": 1000, "observation_id": "bad_area"},
                                   {**base, "camera_distance_m": 1000, "observation_id": "bad_depth"},
                                   {**base, "instance_token": "a", "observation_id": "same_id"}])
        negative = choose_negative(anchor, target, candidates, True, self.cfg, np.random.default_rng(42))
        self.assertEqual(negative["observation_id"], "good")
        self.assertIsNone(choose_negative(anchor, target, candidates, False, self.cfg, np.random.default_rng(42)))
