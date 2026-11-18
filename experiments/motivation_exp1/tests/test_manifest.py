from pathlib import Path
import tempfile
import unittest

from experiments.motivation_exp1.build_nuscenes_manifest import collect_frames, make_pairs
from experiments.motivation_exp1.utils.data_utils import configuration, parser


def config():
    return configuration(parser().parse_args([]))


def frame(scene, camera, token, timestamp, next_token="", exists=True, keyframe=False):
    return {"scene_name": scene, "scene_token": scene, "camera_channel": camera, "sample_data_token": token, "sample_token": token, "timestamp_us": timestamp, "image_path": token + ".jpg", "next_sample_data_token": next_token, "file_exists": exists, "is_key_frame": keyframe}


class ManifestTests(unittest.TestCase):
    def test_real_timestamps_and_scene_boundary(self):
        frames = [frame("a", "CAM_FRONT", "a1", 0, "a2"), frame("a", "CAM_FRONT", "a2", 81001, "a3"), frame("a", "CAM_FRONT", "a3", 249100), frame("b", "CAM_FRONT", "b1", 300000, "b2"), frame("b", "CAM_FRONT", "b2", 389050)]
        pairs, _ = make_pairs(frames, config())
        self.assertTrue(pairs)
        for r in pairs:
            self.assertEqual(r["previous_sample_data_token"][0], r["current_sample_data_token"][0])
            self.assertAlmostEqual(r["actual_delta_t_ms"], (r["current_timestamp_us"] - r["previous_timestamp_us"]) / 1000)
        self.assertIn(81.001, [r["actual_delta_t_ms"] for r in pairs])

    def test_missing_images_do_not_bridge_adjacent(self):
        frames = [frame("a", "CAM_FRONT", "1", 0, "2"), frame("a", "CAM_FRONT", "2", 83000, "3", False), frame("a", "CAM_FRONT", "3", 166000)]
        self.assertFalse(make_pairs(frames, config())[0])

    def test_nonpositive_timestamp_and_unsorted_input(self):
        frames = [frame("a", "CAM_FRONT", "2", 0), frame("a", "CAM_FRONT", "1", 0, "2")]
        self.assertFalse(make_pairs(frames, config())[0])

    def test_reproducible_stratification(self):
        cfg = config()
        cfg["experiment"]["max_temporal_pairs"] = 12
        frames = [frame(s, c, f"{s}{c}{i}", i * 83000, f"{s}{c}{i + 1}" if i < 12 else "", keyframe=i % 6 == 0) for s in ("a", "b") for c in ("CAM_FRONT", "CAM_BACK") for i in range(13)]
        a, _ = make_pairs(frames, cfg)
        b, _ = make_pairs(list(reversed(frames)), cfg)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 12)
        self.assertEqual(len({(r["scene_token"], r["camera_channel"], r["gap_group"]) for r in a}), 12)

    def test_half_second_requires_successive_keyframes(self):
        cfg = config()
        cfg["experiment"]["temporal_gaps_seconds"] = [.5]
        frames = [frame("a", "CAM_FRONT", str(i), i * 83000, str(i + 1), keyframe=i % 6 == 0) for i in range(13)]
        pairs, _ = make_pairs(frames, cfg)
        self.assertEqual({(p["previous_sample_data_token"], p["current_sample_data_token"]) for p in pairs}, {("0", "6"), ("6", "12")})
        self.assertTrue(all(p["pair_kind"] == "adjacent_keyframes" for p in pairs))
        frames[6]["file_exists"] = False
        self.assertFalse(make_pairs(frames, cfg)[0])

    def test_gap_order_does_not_redefine_adjacent(self):
        cfg = config()
        cfg["experiment"]["temporal_gaps_seconds"] = [.25, .083]
        frames = [frame("a", "CAM_FRONT", str(i), i * 83000, str(i + 1)) for i in range(5)]
        pairs, _ = make_pairs(frames, cfg)
        for pair in pairs:
            self.assertAlmostEqual(pair["actual_delta_t_ms"], 83 if pair["gap_group"] == "adjacent" else 249)

    def test_adjacent_outside_timestamp_tolerance_is_excluded(self):
        cfg = config()
        cfg["experiment"]["temporal_gaps_seconds"] = [.083]
        self.assertFalse(make_pairs([frame("a", "CAM_FRONT", "1", 0, "2"), frame("a", "CAM_FRONT", "2", 150000)], cfg)[0])

    def test_sweep_link_cycle_and_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "f.jpg").touch()
            cfg = config()
            cfg["nuscenes"]["dataroot"] = tmp
            tables = {"sample": {"s": {"scene_token": "a", "data": {"CAM_FRONT": "f"}}}, "sample_data": {"f": {"sample_token": "s", "timestamp": 123, "filename": "f.jpg", "calibrated_sensor_token": "c", "ego_pose_token": "e", "prev": "", "next": "f", "is_key_frame": False}}, "calibrated_sensor": {"c": {"sensor_token": "sensor"}}, "sensor": {"sensor": {"channel": "CAM_FRONT"}}}
            class FakeDevkit:
                def get(self, table, token):
                    return tables[table][token]
            rows, issues = collect_frames(FakeDevkit(), [{"name": "a", "token": "a", "first_sample_token": "s"}], cfg)
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["is_key_frame"])
            self.assertEqual(issues[0]["reason"], "cycle")
            tables["sample_data"]["f"]["sample_token"] = "foreign"
            tables["sample"]["foreign"] = {"scene_token": "b"}
            rows, issues = collect_frames(FakeDevkit(), [{"name": "a", "token": "a", "first_sample_token": "s"}], cfg)
            self.assertFalse(rows)
            self.assertEqual(issues[0]["reason"], "scene_boundary")

    def test_configuration_invalid_grid(self):
        with self.assertRaises(ValueError):
            configuration(parser().parse_args(["--analysis-resize", "8", "8", "--patch-grid", "8", "8"]))

    def test_nonfinite_gaps_and_colliding_threshold_labels_rejected(self):
        for options in (["--temporal-gaps-seconds", "nan"], ["--ssim-thresholds", ".701", ".704"]):
            with self.assertRaises(ValueError):
                configuration(parser().parse_args(options))


if __name__ == "__main__":
    unittest.main()
