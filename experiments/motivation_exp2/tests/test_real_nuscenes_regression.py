"""Opt-in real devkit regressions; no synthetic values enter experiment outputs."""
import json
import os
from pathlib import Path
import unittest
import numpy as np
import yaml
from PIL import Image
from nuscenes.utils.geometry_utils import BoxVisibility, view_points

from experiments.motivation_exp2.utils.data_utils import load_nuscenes
from experiments.motivation_exp2.utils.geometry_utils import world_to_camera, project_box


@unittest.skipUnless(os.environ.get("SENNA_EXP2_REAL_OUTPUT"), "Set SENNA_EXP2_REAL_OUTPUT after a genuine E2 run")
class RealNuScenesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(os.environ["SENNA_EXP2_REAL_OUTPUT"])
        cls.cfg = yaml.safe_load((cls.out / "metadata/experiment_config.yaml").read_text())
        cls.nusc = load_nuscenes(cls.cfg)
        cls.records = json.loads((cls.out / "metadata/projection_regression.json").read_text())

    def test_real_boxes_independent_world_transform(self):
        self.assertGreaterEqual(len(self.records),6)
        for r in self.records:
            with self.subTest(token=r["sample_data_token"]):
                sd = self.nusc.get("sample_data",r["sample_data_token"])
                pose = self.nusc.get("ego_pose",sd["ego_pose_token"])
                cal = self.nusc.get("calibrated_sensor",sd["calibrated_sensor_token"])
                _, boxes,k = self.nusc.get_sample_data(sd["token"],box_vis_level=BoxVisibility.NONE)
                box = next(b for b in boxes if b.token == r["annotation_token"])
                world = self.nusc.get_box(box.token)
                np.testing.assert_allclose(world_to_camera(world.corners(),pose,cal),box.corners(),atol=1e-8)
                pixels = view_points(box.corners(),k,normalize=True)
                np.testing.assert_allclose(pixels,np.array(r["official_pixels"]),atol=1e-6)
                result = project_box(box.corners(),k,sd["width"],sd["height"],self.cfg["projection"])
                np.testing.assert_allclose([result[x] for x in ["box_xmin","box_ymin","box_xmax","box_ymax"]],r["box"],atol=1e-6)

    def test_real_visual_regression_artifacts(self):
        files = list((self.out / "visualizations").glob("projection_regression_*.png"))
        self.assertGreaterEqual(len(files),6)
        for p in files:
            with Image.open(p) as image:
                image.verify()
