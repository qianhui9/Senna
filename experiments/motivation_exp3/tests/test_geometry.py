import unittest
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor

from experiments.motivation_exp3.map_bbox_to_patches import (
    preprocessing_geometry, map_original_bbox_to_model_input, patch_weights,
)


class GeometryTests(unittest.TestCase):
    def test_non_square_resize(self):
        p = {"do_resize": True, "size": {"height": 100, "width": 200}, "do_center_crop": False}
        m = map_original_bbox_to_model_input((1000, 500), [100, 50, 300, 150], p, aspect_ratio="square")
        np.testing.assert_allclose(m["bbox"], [20, 10, 60, 30])
        self.assertEqual(m["retained_area_ratio"], 1)

    def test_center_crop_and_partial_retention(self):
        p = {"do_resize": True, "size": {"shortest_edge": 100}, "do_center_crop": True, "crop_size": 100}
        m = map_original_bbox_to_model_input((200, 100), [25, 25, 75, 75], p, aspect_ratio="square")
        np.testing.assert_allclose(m["bbox"], [0, 25, 25, 75])
        self.assertEqual(m["retained_area_ratio"], .5)

    def test_completely_cropped(self):
        p = {"do_resize": True, "size": 100, "do_center_crop": True, "crop_size": 100}
        m = map_original_bbox_to_model_input((200, 100), [0, 0, 40, 40], p, aspect_ratio="square")
        self.assertFalse(m["valid"])
        self.assertEqual(m["reason"], "completely_cropped")

    def test_square_padding(self):
        p = {"do_resize": True, "size": 100, "do_center_crop": True, "crop_size": 100}
        m = map_original_bbox_to_model_input((200, 100), [0, 0, 200, 100], p)
        np.testing.assert_allclose(m["bbox"], [0, 25, 100, 75])

    def test_center_crop_zero_padding(self):
        p = {"do_resize": False, "do_center_crop": True, "crop_size": {"width": 100, "height": 100}}
        m = map_original_bbox_to_model_input((60, 80), [0, 0, 60, 80], p, aspect_ratio="square")
        np.testing.assert_allclose(m["bbox"], [20, 10, 80, 90])

    def test_boundary_clipping_and_invalid(self):
        p = {"do_resize": False, "do_center_crop": False}
        m = map_original_bbox_to_model_input((100, 100), [-10, -10, 10, 10], p, aspect_ratio="square")
        np.testing.assert_allclose(m["bbox"], [0, 0, 10, 10])
        self.assertEqual(m["retained_area_ratio"], .25)
        for box in [[1, 1, 0, 0], [np.nan, 0, 1, 1], [0, 0, 0, 0]]:
            self.assertFalse(map_original_bbox_to_model_input((100, 100), box, p)["valid"])

    def test_patch_intersection_weights_and_order(self):
        indices, weights = patch_weights([5, 0, 15, 10], (20, 20), (2, 2), min_coverage=.25)
        np.testing.assert_array_equal(indices, [0, 1])
        np.testing.assert_allclose(weights, [50, 50])
        indices, _ = patch_weights([0, 10, 10, 20], (20, 20), (2, 2))
        np.testing.assert_array_equal(indices, [2])

    def test_center_mean_and_insufficient_coverage(self):
        indices, weights = patch_weights([0, 0, 10, 10], (20, 20), (2, 2), "center")
        np.testing.assert_array_equal(indices, [0])
        np.testing.assert_array_equal(weights, [1])
        indices, _ = patch_weights([0, 0, 1, 1], (20, 20), (2, 2), min_coverage=.25)
        self.assertEqual(len(indices), 0)

    def test_original_processor_raster_matches_mapping(self):
        from llava.mm_utils import process_images
        from types import SimpleNamespace
        pixels = np.zeros((8, 16, 3), np.uint8)
        pixels[:, 4:12, 1] = 255
        processor = CLIPImageProcessor(size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8},
                                       resample=Image.Resampling.NEAREST, do_normalize=False, do_rescale=False,
                                       image_mean=[0, 0, 0])
        result = process_images([Image.fromarray(pixels)], processor, SimpleNamespace(image_aspect_ratio="pad"))
        geometry = preprocessing_geometry((16, 8), processor)
        bbox = map_original_bbox_to_model_input((16, 8), [4, 0, 12, 8], transform=geometry)["bbox"]
        np.testing.assert_allclose(bbox, [2, 2, 6, 6])
        expected = np.zeros((8, 8), bool)
        expected[2:6, 2:6] = True
        np.testing.assert_array_equal(result[0, 1].numpy() > 0, expected)

    def test_resize_rounding_and_odd_hf_crop_dimensions(self):
        for w, h in [(161, 91), (8, 5), (5, 8), (11, 7)]:
            for do_resize in [True, False]:
                processor = CLIPImageProcessor(size={"shortest_edge": 12}, crop_size={"height": 12, "width": 12}, do_resize=do_resize)
                geometry = preprocessing_geometry((w, h), processor, "square")
                actual = processor(Image.new("RGB", (w, h)), return_tensors="pt")["pixel_values"]
                self.assertEqual(geometry["input_size"], list(actual.shape[-2:][::-1]))

    def test_anyres_refused(self):
        with self.assertRaises(ValueError):
            preprocessing_geometry((100, 100), {}, "anyres")
