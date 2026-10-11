"""Image-space metrics; no alignment, model input changes, or feature reuse."""
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .utils.data_utils import LOG, completed, mark, read_csv, write_csv, write_json


def load_analysis_image(path, size):
    with Image.open(path) as image:
        if min(image.size) < 3:
            raise ValueError(f"Invalid image dimensions: {image.size}")
        original_size = image.size
        resized = np.asarray(image.convert("RGB").resize(tuple(size), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    return resized, original_size


def ssim(a, b):
    """Wang SSIM: RGB channel mean, Gaussian sigma=1.5, population moments.

    11x11 window (truncate=3.5), valid interior only. Small patches use a
    smaller odd window; the chosen window is recorded in patch rows.
    """
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 3:
        raise ValueError("SSIM requires equal H x W x 3 images, at least 3 x 3")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite pixels")
    window = min(11, min(a.shape[:2]))
    window -= int(window % 2 == 0)
    pad = window // 2
    sigma = 1.5 if window == 11 else window / 7.0
    truncate = pad / sigma
    def blur(x):
        return gaussian_filter(x, sigma=(sigma, sigma, 0), truncate=truncate, mode="reflect")
    ma, mb = blur(a), blur(b)
    va, vb = np.maximum(blur(a * a) - ma * ma, 0), np.maximum(blur(b * b) - mb * mb, 0)
    cov = blur(a * b) - ma * mb
    score = ((2 * ma * mb + .01 ** 2) * (2 * cov + .03 ** 2)) / ((ma * ma + mb * mb + .01 ** 2) * (va + vb + .03 ** 2))
    return float(score[pad:-pad, pad:-pad].mean())


def similarity_metrics(a, b, grid, thresholds):
    difference = np.abs(a - b)
    mse = float(np.mean((a - b) ** 2))
    result = {"ssim": ssim(a, b), "psnr_db": 10 * math.log10(1 / mse) if mse > 0 else math.inf, "normalized_pixel_difference": float(difference.mean()), "identical_pixels": mse == 0}
    rows, cols = grid
    if min(rows, cols) < 1 or min(a.shape[0] // rows, a.shape[1] // cols) < 3:
        raise ValueError("Grid creates patches smaller than 3 x 3")
    ys, xs = np.linspace(0, a.shape[0], rows + 1, dtype=int), np.linspace(0, a.shape[1], cols + 1, dtype=int)
    patches = []
    for y in range(rows):
        for x in range(cols):
            aa, bb = a[ys[y]:ys[y + 1], xs[x]:xs[x + 1]], b[ys[y]:ys[y + 1], xs[x]:xs[x + 1]]
            window = min(11, *aa.shape[:2])
            patches.append({"patch_row": y, "patch_col": x, "pixel_y0": ys[y], "pixel_y1": ys[y + 1], "pixel_x0": xs[x], "pixel_x1": xs[x + 1], "ssim_window": window - int(window % 2 == 0), "patch_ssim": ssim(aa, bb), "patch_normalized_pixel_difference": float(np.abs(aa - bb).mean())})
    for t in thresholds:
        result[f"high_similarity_ratio_{t:.2f}"] = sum(p["patch_ssim"] >= t for p in patches) / len(patches)
    return result, patches


def run(cfg, args, out):
    if completed(out, "redundancy", args):
        return
    pairs = read_csv(out / "temporal_pairs.csv")
    if not pairs:
        raise ValueError("Run --mode index first")
    previous = read_csv(out / "temporal_similarity.csv") if args.existing == "resume" else []
    patch_rows = read_csv(out / "patch_similarity.csv") if args.existing == "resume" else []
    seen = {r["pair_id"] for r in previous if r.get("status") == "ok"}
    results = [r for r in previous if r["pair_id"] in seen]
    patch_rows = [r for r in patch_rows if r["pair_id"] in seen]
    # A pair is committed only together with its complete patch set.
    expected = np.prod(cfg["experiment"]["patch_grid"])
    valid_seen = {k for k in seen if sum(r["pair_id"] == k for r in patch_rows) == expected}
    results = [r for r in results if r["pair_id"] in valid_seen]
    patch_rows = [r for r in patch_rows if r["pair_id"] in valid_seen]
    issues = []
    for index, pair in enumerate(pairs):
        if pair["pair_id"] in valid_seen:
            continue
        try:
            a, size_a = load_analysis_image(pair["previous_image_path"], cfg["experiment"]["analysis_resize"])
            b, size_b = load_analysis_image(pair["current_image_path"], cfg["experiment"]["analysis_resize"])
            metrics, patches = similarity_metrics(a, b, cfg["experiment"]["patch_grid"], cfg["experiment"]["ssim_thresholds"])
            if size_a != size_b:
                issues.append({"pair_id": pair["pair_id"], "reason": "original_dimensions_differ", "previous": size_a, "current": size_b})
            results.append({**pair, "status": "ok", "previous_original_size": str(size_a), "current_original_size": str(size_b), **metrics, "error": ""})
            patch_rows.extend({**pair, **p} for p in patches)
        except (OSError, ValueError) as exc:
            results.append({**pair, "status": "error", "previous_original_size": "", "current_original_size": "", "ssim": "", "psnr_db": "", "normalized_pixel_difference": "", "identical_pixels": "", **{f"high_similarity_ratio_{t:.2f}": "" for t in cfg["experiment"]["ssim_thresholds"]}, "error": str(exc)})
            issues.append({"pair_id": pair["pair_id"], "reason": str(exc)})
        if (index + 1) % 25 == 0:
            write_csv(out / "patch_similarity.csv", patch_rows)
            write_csv(out / "temporal_similarity.csv", results)
            LOG.info("Analyzed %d/%d pairs", index + 1, len(pairs))
    write_csv(out / "patch_similarity.csv", patch_rows)
    write_csv(out / "temporal_similarity.csv", results)
    write_json(out / "similarity_issues.json", issues)
    valid = sum(r["status"] == "ok" for r in results)
    mark(out, "redundancy", "complete" if valid == len(pairs) else "partial", valid_pairs=valid, failed_pairs=len(pairs) - valid)
    LOG.info("Image similarity: %d valid pairs", valid)


if __name__ == "__main__":
    from .run import main
    raise SystemExit(main("redundancy"))
