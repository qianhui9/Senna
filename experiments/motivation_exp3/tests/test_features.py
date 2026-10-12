from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from transformers import CLIPVisionConfig, CLIPVisionModel

from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from llava.senna.senna_llava_llama import SennaLlavaLlamaForCausalLM
from llava.senna.senna_llava_arch import img_adapter
from experiments.motivation_exp3.extract_senna_features import (
    capture_features, original_visual_forward, compare_tensors, resolve_layers, save_shard, load_shard,
)
from experiments.motivation_exp3.extract_object_features import aggregate_patches
from experiments.motivation_exp3.analyze_feature_similarity import feature_metrics
from experiments.motivation_exp3.analyze_layer_stability import common_layer_cohort


class TinyOriginalPath(torch.nn.Module):
    """Real HF CLIP and real Senna methods with small random test-only weights."""
    encode_images = SennaLlavaLlamaForCausalLM.encode_images
    mm_project = SennaLlavaLlamaForCausalLM.mm_project

    def __init__(self):
        super().__init__()
        tower = CLIPVisionTower.__new__(CLIPVisionTower)
        torch.nn.Module.__init__(tower)
        tower.is_loaded = True
        tower.select_layer, tower.select_feature = -2, "patch"
        tower.vision_tower = CLIPVisionModel(CLIPVisionConfig(image_size=28, patch_size=14, hidden_size=8,
            intermediate_size=16, num_hidden_layers=2, num_attention_heads=2))
        self.tower = tower
        self.mm_projector = torch.nn.Sequential(torch.nn.Linear(8, 12), torch.nn.GELU(), torch.nn.Linear(12, 12))
        self.img_adapter = img_adapter(4, 2)

    def get_model(self):
        return self

    def get_vision_tower(self):
        return self.tower


class FeatureTests(unittest.TestCase):
    def test_actual_shapes_cls_indexing_equivalence_and_batch(self):
        torch.manual_seed(42)
        model = TinyOriginalPath().eval()
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        pixels = torch.randn(2, 3, 28, 28)
        with torch.inference_mode():
            a, c = original_visual_forward(model, pixels)
            with capture_features(model, [0, 1, 2]) as (captured, shapes):
                b, d = original_visual_forward(model, pixels)
            self.assertTrue(compare_tensors(a, b, 0, 0)["passed"])
            self.assertTrue(compare_tensors(c, d, 0, 0)["passed"])
            self.assertEqual(shapes["layer_0"]["shape"], [2, 5, 8])
            self.assertEqual(captured["layer_1"].shape, (2, 4, 8))
            self.assertEqual(captured["projector"].shape, (2, 4, 12))
            self.assertEqual(captured["adapter_global"].shape, (2, 12))
            self.assertTrue(torch.equal(captured["layer_1"], a))
            raw = model.tower.vision_tower(pixels, output_hidden_states=True)
            self.assertTrue(torch.equal(raw.hidden_states[0][:, 1:], captured["layer_0"]))
            with capture_features(model, [0, 1, 2]) as (repeat, _):
                original_visual_forward(model, pixels)
            for layer in captured:
                self.assertTrue(torch.equal(captured[layer], repeat[layer]))
            solo, _ = original_visual_forward(model, pixels[:1])
            torch.testing.assert_close(solo, a[:1], rtol=1e-5, atol=1e-5)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, initial[name]))

    def test_hooks_removed_on_exception(self):
        model = TinyOriginalPath()
        with self.assertRaises(RuntimeError):
            with capture_features(model, [0]):
                raise RuntimeError("unit fixture")
        self.assertEqual(len(model.tower.vision_tower._forward_hooks), 0)
        self.assertEqual(len(model.mm_projector._forward_hooks), 0)
        self.assertEqual(len(model.img_adapter._forward_hooks), 0)

    def test_layer_zero_selected_and_out_of_range(self):
        self.assertEqual(resolve_layers(24, [0, 6, 12, 18, 24], -2), [0, 6, 12, 18, 23, 24])
        self.assertEqual(resolve_layers(2, [], -1), [0, 1, 2])
        with self.assertRaises(ValueError):
            resolve_layers(24, [25], -2)

    def test_pooling_weights_and_empty(self):
        x = np.array([[1, 0], [0, 2], [9, 9]], np.float16)
        np.testing.assert_allclose(aggregate_patches(x, [0, 1], [1, 3]), [.25, 1.5])
        self.assertEqual(aggregate_patches(x, [], []), None)
        self.assertEqual(aggregate_patches(x, [0], [0]), None)
        self.assertEqual(aggregate_patches(x, [0], [np.nan]), None)

    def test_fp32_metrics_identical_orthogonal_and_t1_denominator(self):
        x = np.array([1000, 1000], np.float16)
        self.assertAlmostEqual(feature_metrics(x, x)["cosine_similarity"], 1, places=6)
        self.assertEqual(feature_metrics(x, x)["relative_l2_error"], 0)
        self.assertEqual(feature_metrics([1, 0], [0, 1])["cosine_similarity"], 0)
        self.assertAlmostEqual(feature_metrics([1, 0], [2, 0])["relative_l2_error"], .5)
        self.assertEqual(feature_metrics([0, 0], [0, 0])["numeric_status"], "near_zero_norm")
        self.assertEqual(feature_metrics([1, np.inf], [0, 1])["numeric_status"], "nonfinite")

    def test_shard_identity_metadata_and_atomic_serialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"x.npz"
            meta = {"scene_name": "real-format-unit-fixture", "camera_channel": "CAM_FRONT", "timestamp_us": 123,
                    "instance_token": "official-fixture", "layers": ["layer_0"]}
            arrays = {"layer_0__area": np.array([[1, 2]], np.float16)}
            save_shard(p, meta, arrays)
            actual, vectors = load_shard(p)
            self.assertEqual(actual, meta)
            np.testing.assert_array_equal(vectors["layer_0__area"], arrays["layer_0__area"])
            self.assertFalse(p.with_suffix(".npz.tmp").exists())

    def test_layer_cohort_excludes_missing_layer_and_invalid_mapping(self):
        import pandas as pd
        records = [{"object_pair_id": p, "comparison_group": "B1", "aggregation": "area", "layer_id": l,
                    "numeric_status": "valid" if p == "good" or l == "layer_0" else "invalid_patch_mapping"}
                   for p in ["good", "bad"] for l in ["layer_0", "layer_1"]]
        self.assertEqual(set(common_layer_cohort(pd.DataFrame(records)).object_pair_id), {"good"})
