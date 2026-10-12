import copy
import unittest
from experiments.motivation_exp2.build_keyframe_pairs import keyframe_pairs, sample_sequence


class PairTests(unittest.TestCase):
    def setUp(self):
        self.samples = [dict(token=str(i), scene_token="s", timestamp=1000000+i*501001, prev=str(i-1) if i else "", next=str(i+1) if i < 5 else "") for i in range(6)]
        self.scene = dict(token="s", name="scene", first_sample_token="0", last_sample_token="5", nbr_samples=6)

    def test_real_timestamp_gaps_and_counts(self):
        pairs = keyframe_pairs(self.samples, self.scene, [1,2,4])
        self.assertEqual(len(pairs), 5+4+2)
        for p in pairs:
            self.assertEqual(p["index_t1"]-p["index_t0"], p["gap_steps"])
            self.assertAlmostEqual(p["delta_t_seconds"], p["gap_steps"]*.501001)

    def test_no_cross_scene(self):
        self.samples[-1]["scene_token"] = "other"
        with self.assertRaises(ValueError):
            keyframe_pairs(self.samples, self.scene, [1])

    def test_next_link_order(self):
        class DB:
            def get(_, table, token):
                return self.samples[int(token)]
        self.assertEqual(len(sample_sequence(DB(), self.scene)), 6)
        self.samples[2]["prev"] = "0"
        with self.assertRaises(ValueError):
            sample_sequence(DB(), self.scene)

    def test_invalid_timestamp_order(self):
        self.samples[1]["timestamp"] = self.samples[0]["timestamp"]
        with self.assertRaises(ValueError):
            keyframe_pairs(self.samples, self.scene, [1])
