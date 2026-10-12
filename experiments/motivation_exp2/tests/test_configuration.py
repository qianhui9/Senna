"""Configuration isolation and reproducible CLI override behavior."""
import unittest
from experiments.motivation_exp2.utils.data_utils import configuration, parser


class ConfigurationTests(unittest.TestCase):
    def args(self, extra):
        return parser().parse_args(["--dataroot", "/tmp/real-data-required-at-runtime", "--output-dir", "/tmp/exp2-config-test", *extra])

    def test_nested_list_and_numeric_cli_overrides(self):
        cfg = configuration(self.args(["--max-scenes", "3", "--set", "analysis.static_speed_threshold_mps=0.2", "--set", "projection.min_visible_area_ratio=0.4", "--set", "analysis.grid_shapes=[[16,16]]"]))
        self.assertEqual(cfg["dataset"]["max_scenes"],3)
        self.assertEqual(cfg["analysis"]["static_speed_threshold_mps"],.2)
        self.assertEqual(cfg["projection"]["min_visible_area_ratio"],.4)
        self.assertEqual(cfg["analysis"]["grid_shapes"],[[16,16]])

    def test_cross_scene_override_forbidden(self):
        with self.assertRaises(ValueError):
            configuration(self.args(["--set", "pairs.allow_cross_scene=true"]))

    def test_exp1_output_write_forbidden(self):
        with self.assertRaises(ValueError):
            configuration(self.args(["--output-dir", "outputs/motivation_exp1"]))

    def test_smoke_remains_real_keyframe_workload(self):
        cfg = configuration(self.args(["--smoke-test", "--max-scenes", "10"]))
        self.assertEqual(cfg["dataset"]["max_scenes"],2)
        self.assertEqual(cfg["pairs"]["keyframe_gaps"],[1,2,4])
        self.assertEqual(len(cfg["dataset"]["cameras"]),6)
        self.assertEqual(cfg["analysis"]["bootstrap_iterations"],1000)

    def test_default_smoke_output_is_separate(self):
        cfg = configuration(parser().parse_args(["--dataroot","/tmp/real-data-required-at-runtime","--smoke-test"]))
        self.assertTrue(cfg["output"]["directory"].endswith("motivation_exp2_smoke"))

    def test_explicit_smoke_output_is_preserved(self):
        cfg = configuration(self.args(["--smoke-test"]))
        self.assertEqual(cfg["output"]["directory"],"/tmp/exp2-config-test")
