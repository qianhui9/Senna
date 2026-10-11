"""Regression checks for coordinate, camera, metric and checkpoint mistakes."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_tools.senna_inputs import CAMERAS, VIEWS, IMAGE_PROMPT, navigation_command, parse_plan, validate_sample, current_speed
from eval_tools.senna_plan_cmd_eval_multi_img import summarize, read_checkpoint


class InputTests(unittest.TestCase):
    def test_navigation_axes_and_thresholds(self):
        cases = [((-0.671, 15.787), "go straight"), ((0, 40), "go straight"),
                 ((-10, 15), "turn left"), ((10, 15), "turn right"),
                 ((-15, 25), "go straight and turn left"),
                 ((15, 25), "go straight and turn right")]
        for xy, expected in cases:
            with self.subTest(xy=xy):
                self.assertEqual(navigation_command(np.array([xy])), expected)
        self.assertEqual(navigation_command(np.array([[-0.671, 15.787]]), legacy=True), "turn left")

    def test_view_order_and_legacy_rejection(self):
        self.assertEqual([view.replace(" ", "_") for view in VIEWS],
                         [camera.removeprefix("CAM_") for camera in CAMERAS])
        sample = {"token": "old", "images": ["missing"] * 6,
                  "conversations": [{"value": IMAGE_PROMPT}, {"value": "KEEP, STRAIGHT"}]}
        with self.assertRaisesRegex(ValueError, "Legacy"):
            validate_sample(sample)

    def test_strict_parse_and_class_f1(self):
        self.assertEqual(parse_plan("KEEP, LEFT_TURN\n"), ("KEEP", "LEFT_TURN"))
        for answer in ("KEEP, LEFT_TURN or RIGHT_TURN", "ACCELERATE DECELERATE, STRAIGHT", "KEEPER, STRAIGHT"):
            self.assertEqual(parse_plan(answer), (None, None))
        records = [{"token": "a", "gt": "KEEP, LEFT_TURN", "pred": "KEEP, LEFT_TURN"},
                   {"token": "b", "gt": "STOP, RIGHT_TURN", "pred": "STOP, LEFT_TURN"},
                   {"token": "c", "gt": "KEEP, STRAIGHT", "pred": "KEEP, STRAIGHT or LEFT_TURN"}]
        metrics = summarize(records)["metrics"]
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["invalid_outputs"], 1)
        self.assertAlmostEqual(metrics["class_metrics"]["LEFT_TURN"]["f1"], 2 / 3)
        self.assertEqual(metrics["path_confusion"]["RIGHT_TURN"]["LEFT_TURN"], 1)

    def test_nearest_can_bus_not_previous(self):
        class Nusc:
            def get(self, *_):
                return {"name": "scene"}
        class Bus:
            def get_messages(self, *_):
                return [{"utime": 0, "vel": [1]}, {"utime": 10, "vel": [9]}]
        self.assertEqual(current_speed(Nusc(), Bus(), {"scene_token": "s", "timestamp": 9}), 9)

    def test_resume_metadata_and_partial_tail(self):
        with tempfile.TemporaryDirectory(prefix="senna_test_") as directory:
            path = Path(directory) / "checkpoint.jsonl"
            metadata = {"run": "a"}
            lines = [json.dumps({"type": "metadata", "value": metadata}),
                     json.dumps({"type": "prediction", "value": {"token": "s"}})]
            path.write_text("\n".join(lines) + '\n{"type": "prediction"', encoding="utf-8")
            self.assertEqual(read_checkpoint(path, metadata), [{"token": "s"}])
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            with self.assertRaisesRegex(ValueError, "differ"):
                read_checkpoint(path, {"run": "b"})


if __name__ == "__main__":
    unittest.main()
