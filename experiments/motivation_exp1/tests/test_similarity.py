import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments.motivation_exp1.analyze_temporal_redundancy import load_analysis_image, similarity_metrics, ssim


class SimilarityTests(unittest.TestCase):
    def test_identical(self):
        a = np.random.default_rng(42).random((24, 32, 3))
        m, p = similarity_metrics(a, a, (2, 2), [.7, .8, .9])
        self.assertAlmostEqual(m["ssim"], 1)
        self.assertTrue(math.isinf(m["psnr_db"]))
        self.assertEqual(m["normalized_pixel_difference"], 0)
        self.assertEqual(m["high_similarity_ratio_0.90"], 1)

    def test_analytic_constant_images(self):
        a, b = np.full((16, 16, 3), .5), np.ones((16, 16, 3))
        m, _ = similarity_metrics(a, b, (1, 1), [.7])
        self.assertAlmostEqual(m["ssim"], (2 * .5 + .01 ** 2) / (.5 ** 2 + 1 + .01 ** 2), places=9)
        self.assertAlmostEqual(m["psnr_db"], 10 * math.log10(4))
        self.assertAlmostEqual(m["normalized_pixel_difference"], .5)

    def test_patch_coverage_and_thresholds(self):
        a, b = np.zeros((25, 33, 3)), np.zeros((25, 33, 3))
        b[:, 16:] = 1
        m, patches = similarity_metrics(a, b, (1, 2), [.7, .8, .9])
        self.assertEqual(m["high_similarity_ratio_0.90"], .5)
        self.assertEqual(sum((p["pixel_y1"] - p["pixel_y0"]) * (p["pixel_x1"] - p["pixel_x0"]) for p in patches), 25 * 33)

    def test_dimensions_and_small_patches(self):
        with self.assertRaises(ValueError):
            ssim(np.zeros((2, 3, 3)), np.zeros((2, 3, 3)))
        with self.assertRaises(ValueError):
            ssim(np.zeros((4, 4, 3)), np.zeros((4, 5, 3)))
        self.assertEqual(ssim(np.zeros((3, 3, 3)), np.zeros((3, 3, 3))), 1)

    def test_decode_failure_and_rgb_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            Image.new("L", (20, 10), 128).save(path)
            a, size = load_analysis_image(path, (32, 24))
            self.assertEqual(size, (20, 10))
            self.assertEqual(a.shape, (24, 32, 3))
            path.write_text("corrupt")
            with self.assertRaises(OSError):
                load_analysis_image(path, (32, 24))

    def test_nonfinite_pixels(self):
        a = np.full((12, 12, 3), np.nan)
        with self.assertRaises(ValueError):
            ssim(a, a)

    def test_ssim_independent_gaussian_reference(self):
        import cv2
        a, b = np.random.default_rng(10).random((2, 20, 24, 3))
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = kernel @ kernel.T
        means_a, means_b = cv2.filter2D(a, -1, window), cv2.filter2D(b, -1, window)
        var_a = cv2.filter2D(a * a, -1, window) - means_a ** 2
        var_b = cv2.filter2D(b * b, -1, window) - means_b ** 2
        cov = cv2.filter2D(a * b, -1, window) - means_a * means_b
        ref = ((2 * means_a * means_b + .0001) * (2 * cov + .0009)) / ((means_a ** 2 + means_b ** 2 + .0001) * (var_a + var_b + .0009))
        self.assertAlmostEqual(ssim(a, b), float(ref[5:-5, 5:-5].mean()), places=10)


if __name__ == "__main__":
    unittest.main()
