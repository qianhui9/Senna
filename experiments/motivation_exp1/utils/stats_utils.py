"""Scene-level bootstrap: repeated frames are not independent replicates."""
from collections import defaultdict
import numpy as np


def finite_values(values):
    result = []
    for value in values:
        try:
            value = float(value)
            if np.isfinite(value):
                result.append(value)
        except (ValueError, TypeError):
            pass
    return np.asarray(result, dtype=float)


def distribution(values):
    values = finite_values(values)
    if not len(values):
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {"n": len(values), "mean": float(values.mean()), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99))}


def scene_bootstrap(records, metric, seed=42, iterations=2000):
    groups = defaultdict(list)
    for row in records:
        values = finite_values([row.get(metric)])
        if len(values):
            groups[row["scene_token"]].append(values[0])
    means = np.array([np.mean(groups[k]) for k in sorted(groups)])
    result = {"scenes": len(means), "scene_equal_weight_mean": float(means.mean()) if len(means) else None, "ci95": None, "bootstrap_unit": "scene mean", "scene_means": {k: float(np.mean(v)) for k, v in sorted(groups.items())}}
    if len(means) >= 2:
        rng = np.random.default_rng(seed)
        draws = rng.choice(means, size=(iterations, len(means)), replace=True).mean(axis=1)
        result["ci95"] = np.percentile(draws, [2.5, 97.5]).tolist()
    return result


def latency_partition(end_to_end, loading, preprocessing, transfer, generate, preparation, vision, projector, adapter):
    """Disjoint stream timeline; reject invalid parents instead of clipping costs."""
    values = [end_to_end, loading, preprocessing, transfer, generate, preparation, vision, projector, adapter]
    if end_to_end <= 0:
        raise ValueError("End-to-end latency must be positive")
    if any(not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("Negative/non-finite timings")
    visual = vision + projector + adapter
    if visual > preparation + 0.1 or preparation > generate + 0.1:
        raise ValueError("Child timings exceed parent; stage decomposition is unreliable")
    prep_other = max(0, preparation - visual)
    llm = max(0, generate - preparation)
    residual = end_to_end - loading - preprocessing - transfer - generate
    if residual < -0.1:
        raise ValueError("Timed stages exceed end-to-end latency")
    return {"multimodal_preparation_ms": prep_other, "llm_generation_ms": llm, "other_ms": max(0, residual), "visual_path_ms": visual,
            "vision_latency_ratio": vision / end_to_end, "projector_latency_ratio": projector / end_to_end,
            "visual_path_latency_ratio": visual / end_to_end, "llm_generation_latency_ratio": llm / end_to_end,
            "ideal_visual_path_speedup_bound": end_to_end / (end_to_end - visual) if end_to_end > visual else None}
