"""CPU analysis of measured, spatially corresponding patch features."""
import math
import numpy as np


def token_distances(previous, current):
    a, b = np.asarray(previous, dtype=np.float32), np.asarray(current, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2 or min(a.shape) < 1:
        raise ValueError("Expected equal nonempty [patches, channels] feature arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite features; the entire pair is rejected")
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    if np.any(na <= 1e-12) or np.any(nb <= 1e-12):
        raise ValueError("Zero-norm token; cosine distance is undefined")
    a, b = a / na[:, None], b / nb[:, None]
    cosine = np.clip(1 - np.sum(a * b, axis=1), 0, 2)
    normalized_l2 = np.linalg.norm(a - b, axis=1)
    return cosine, normalized_l2


def drift_statistics(distances, thresholds):
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("Drift must be a finite nonnegative vector")
    ordered = np.sort(d)
    n, total = len(d), float(d.sum())
    result = {"mean_token_drift": float(d.mean()), "median_token_drift": float(np.median(d)),
              "p90_token_drift": float(np.percentile(d, 90)), "p95_token_drift": float(np.percentile(d, 95)),
              "gini": float(2 * np.dot(np.arange(1, n + 1), ordered) / (n * total) - (n + 1) / n) if total else 0.0}
    result["gini"] = max(0.0, result["gini"])
    for pct in (10, 20):
        k = math.ceil(n * pct / 100)
        result[f"top{pct}_contribution"] = float(ordered[-k:].sum() / total) if total else 0.0
        result[f"top{pct}_tokens"] = k
        result[f"top{pct}_uniform_reference"] = k / n
    for t in thresholds:
        result[stable_key(t)] = float(np.mean(d <= t))
    return result


def stable_key(threshold):
    return f"stable_ratio_{threshold:g}"


def padded_content_fraction(image_size, rows, cols):
    """Geometric patch overlap with original pixels after expand2square.

    Coordinates precede resampling; interpolation at the boundary is not modeled.
    This diagnostic never changes images, tokens, or the model forward pass.
    """
    width, height = image_size
    side = max(width, height)
    x0, y0 = (side - width) // 2, (side - height) // 2
    xs, ys = np.linspace(0, side, cols + 1), np.linspace(0, side, rows + 1)
    wx = np.maximum(0, np.minimum(xs[1:], x0 + width) - np.maximum(xs[:-1], x0)) / np.diff(xs)
    hy = np.maximum(0, np.minimum(ys[1:], y0 + height) - np.maximum(ys[:-1], y0)) / np.diff(ys)
    return np.outer(hy, wx).ravel()
