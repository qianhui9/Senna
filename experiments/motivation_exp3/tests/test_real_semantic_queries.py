"""Read-only validation of real nuScenes query artifacts (opt-in environment path)."""
import csv
import json
import os
from pathlib import Path
import unittest

import yaml

from data_tools.senna_inputs import CAMERAS, parse_plan
from experiments.motivation_exp3.build_query_sequences import model_sample


@unittest.skipUnless(os.environ.get("SENNA_EXP3_QUERY_OUTPUT"), "Set SENNA_EXP3_QUERY_OUTPUT to real query outputs")
class RealSemanticQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(os.environ["SENNA_EXP3_QUERY_OUTPUT"])
        cls.queries = json.loads((cls.out / "queries.json").read_text())
        cls.base = json.loads((cls.out / "base_queries.json").read_text())
        cls.summary = json.loads((cls.out / "aggregate_summary.json").read_text())
        cls.cfg = yaml.safe_load((cls.out / "resolved_query_config.yaml").read_text())

    def test_all_query_links_and_images_are_valid(self):
        sequences = {}
        for q in self.queries:
            self.assertEqual(len(q["image_paths"]), 6)
            self.assertEqual(tuple(Path(p).parent.name for p in q["image_paths"]), CAMERAS)
            self.assertTrue(all(Path(p).is_file() for p in q["image_paths"]))
            self.assertEqual(model_sample(q)["conversations"][0]["value"].count("<image>"), 6)
            sequences.setdefault(q["sequence_id"], []).append(q)
        for group in sequences.values():
            group.sort(key=lambda q: q["frame_index"])
            self.assertEqual(len({q["scene_token"] for q in group}), 1)
            for a, b in zip(group, group[1:]):
                self.assertLess(a["timestamp"], b["timestamp"])
                if b["frame_index"] == a["frame_index"] + 1:
                    self.assertEqual(b["prev_sample_token"], a["sample_token"])

    def test_planning_matches_existing_official_corrected_evaluation(self):
        root = Path(__file__).resolve().parents[3]
        evaluation = json.loads((root / "data/senna_nusc_val_plan_corrected.json").read_text())
        lookup = {s["token"]: s for s in evaluation}
        planning = [q for q in self.base if q["query_type"] == "meta_action_planning"]
        self.assertTrue(planning)
        for q in planning:
            original = lookup[q["sample_token"]]
            self.assertEqual(model_sample(q)["conversations"][0]["value"], original["conversations"][0]["value"])
            self.assertEqual(parse_plan(q["ground_truth"]), parse_plan(original["conversations"][1]["value"].strip()))
            self.assertEqual(q["input_protocol"]["navigation_source"], "offline_future_route_6s")

    def test_motion_futures_are_offline_complete_and_time_linked(self):
        motion = [q for q in self.base if q["query_type"] == "motion_prediction"]
        self.assertTrue(motion)
        for q in motion:
            offline = q["offline_ground_truth"]
            self.assertEqual(len(offline["future_annotation_tokens"]), 6)
            self.assertEqual(len(set(offline["future_annotation_tokens"])), 6)
            times = [q["timestamp"]] + offline["future_timestamps"]
            for a, b in zip(times, times[1:]):
                self.assertLessEqual(abs((b-a)/1e6 - .5), self.cfg["selection"]["interval_tolerance_s"])
            modified = {**q, "ground_truth": "FUTURE_SENTINEL", "offline_ground_truth": {"future": [999999]}}
            self.assertEqual(model_sample(q), model_sample(modified))
            self.assertEqual(q["dependency_attributes"]["velocity_source"], "previous_annotation_only")

    def test_primary_pairs_are_deduplicated_and_safe_requires_valid_facts(self):
        with (self.out / "dependency_transitions.csv").open() as f:
            pairs = [r for r in csv.DictReader(f) if r["comparison"] == "canonical_adjacent_template"]
        keys = {(r["scene_token"], r["query_id"], r["previous_sample_token"], r["sample_token"]) for r in pairs}
        self.assertEqual(len(keys), len(pairs))
        self.assertEqual(len(pairs), self.summary["canonical_adjacent_pairs"])
        for r in pairs:
            if r["oracle_safe_reuse"] == "1":
                self.assertEqual(r["answer_stable"], "1")
                self.assertEqual(r["dependency_stable"], "1")
                self.assertEqual(r["stale_answer"], "0")
                self.assertEqual(r["state_information_unavailable"], "0")

    def test_required_deliverables_and_plot_formats(self):
        from PIL import Image
        for name in ("query_traces.csv", "query_dependencies.csv", "dependency_transitions.csv", "query_profile_samples.csv",
                     "query_type_summary.csv", "sequence_summary.csv", "aggregate_summary.json", "experiment_report.md"):
            self.assertTrue((self.out / name).is_file(), name)
        pngs = list((self.out / "figures").glob("*.png"))
        self.assertEqual(len(pngs), 7)
        for p in pngs:
            with Image.open(p) as image:
                self.assertAlmostEqual(image.info["dpi"][0], 300, places=1)
            self.assertTrue(p.with_suffix(".pdf").read_bytes().startswith(b"%PDF"))

    def test_formal_measurements_exclude_shared_load(self):
        metadata = self.out / "query_model_metadata.json"
        if metadata.exists():
            self.assertNotIn("shared GPU allowed", json.loads(metadata.read_text())["concurrent_load_policy"])
        else:
            self.assertEqual(self.summary["profile_measurements"], 0)


if __name__ == "__main__":
    unittest.main()
