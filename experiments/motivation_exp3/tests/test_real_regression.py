"""Verify real generated outputs when SENNA_EXP3_OUTPUT is explicitly supplied."""
import json
import os
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from experiments.motivation_exp3.extract_senna_features import load_shard


@unittest.skipUnless(os.getenv("SENNA_EXP3_OUTPUT"), "Set SENNA_EXP3_OUTPUT to a completed real experiment")
class RealOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(os.environ["SENNA_EXP3_OUTPUT"])
        cls.pairs = pd.read_csv(cls.out/"data/feature_object_pairs.csv")
        cls.observations = pd.read_csv(cls.out/"data/object_observations.csv")
        cls.similarity = pd.read_csv(cls.out/"data/feature_similarity.csv")
        cls.model = json.loads((cls.out/"metadata/model_metadata.json").read_text())

    def test_real_identity_time_paths_and_unique_pairs(self):
        self.assertGreater(len(self.pairs), 0)
        self.assertFalse(self.pairs.object_pair_id.duplicated().any())
        self.assertTrue((self.pairs.timestamp_t1 > self.pairs.timestamp_t0).all())
        np.testing.assert_allclose((self.pairs.timestamp_t1-self.pairs.timestamp_t0)/1e6, self.pairs.delta_t_seconds, atol=1e-8)
        index = self.observations.set_index("observation_id")
        for row in self.pairs.itertuples():
            a, b = index.loc[row.observation_id_t0], index.loc[row.observation_id_t1]
            self.assertEqual(a.instance_token, b.instance_token)
            self.assertEqual(a.scene_token, b.scene_token)
            self.assertEqual(a.camera_channel, b.camera_channel)
            self.assertTrue(Path(row.image_path_t0).exists())
            self.assertTrue(Path(row.image_path_t1).exists())

    def test_real_original_forward_shapes_and_repeat_control(self):
        cfg = self.model["vision_config"]
        tokens = (cfg["image_size"]//cfg["patch_size"])**2
        for layer in self.model["feature_layers"]:
            record = self.model["runtime_shapes"][f"layer_{layer}"]
            self.assertEqual(record["shape"][1:], [tokens+1, cfg["hidden_size"]])
            self.assertEqual(record["patch_shape"][1:], [tokens, cfg["hidden_size"]])
        self.assertFalse(self.model["runtime_shapes"]["adapter_global"]["spatial_mapping"])
        checks = json.loads((self.out/"metadata/output_equivalence.json").read_text())
        self.assertTrue(checks and all(c["passed"] for c in checks))
        self.assertTrue(any(c.get("repeated_object_descriptors") for c in checks))

    def test_real_negatives_and_no_fabricated_invalid_metrics(self):
        s = self.similarity
        same = s[s.comparison_group == "B1_same_instance"]
        self.assertTrue((same.instance_token == same.comparison_instance_token_t1).all())
        neg = s[s.comparison_group.str.startswith(("B2", "B3"))]
        self.assertGreater(len(neg), 0)
        self.assertTrue((neg.instance_token != neg.comparison_instance_token_t1).all())
        obs = self.observations.set_index("observation_id")
        for row in neg.drop_duplicates(["object_pair_id", "comparison_group"]).itertuples():
            target = obs.loc[row.observation_id_t1]
            self.assertEqual(target.sample_data_token, row.sample_data_token_t1)
            self.assertEqual(target.scene_token, row.scene_token)
            self.assertEqual(target.camera_channel, row.camera_channel)
        self.assertTrue(s.loc[s.numeric_status != "valid", "cosine_similarity"].isna().all())
        self.assertTrue(np.isfinite(s.loc[s.numeric_status == "valid", "cosine_similarity"]).all())

    def test_real_shard_metadata_and_all_figures(self):
        token = self.observations.sample_data_token.iloc[0]
        meta, arrays = load_shard(self.out/"features/object_feature_shards"/(token+".npz"))
        self.assertEqual(meta["sample_data_token"], token)
        self.assertTrue(all(k in meta["observations"][0] for k in ["scene_name", "camera_channel", "instance_token", "camera_timestamp"]))
        self.assertEqual(arrays["layer_0__area"].shape[0], len(meta["observations"]))
        self.assertEqual(len(list((self.out/"figures").glob("M3*.png"))), 6)
        self.assertEqual(len(list((self.out/"figure_pdf").glob("M3*.pdf"))), 6)
        self.assertTrue((self.out/"experiment_report.md").is_file())
