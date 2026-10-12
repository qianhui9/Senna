"""Scene-level uncertainty and descriptive observation distributions."""
import numpy as np
import pandas as pd

from experiments.motivation_exp1.utils.stats_utils import scene_bootstrap
from experiments.motivation_exp2.utils.statistics_utils import correlation_summary

METRICS = ["cosine_similarity", "cosine_distance", "relative_l2_error"]


def summary(frame, metric, cfg):
    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(float)
    finite = np.isfinite(values)
    x = values[finite]
    # Bootstrap only scene means; avoid repeatedly copying bbox/path metadata.
    scene_values = pd.DataFrame({"scene_token": frame.loc[finite, "scene_token"].to_numpy(), metric: x})
    scene_means = scene_values.groupby("scene_token", as_index=False)[metric].mean()
    result = {"n": len(x), "excluded": len(frame)-len(x), "mean": float(x.mean()) if len(x) else None,
              "std": float(x.std(ddof=1)) if len(x) > 1 else None,
              **{f"p{q}": float(np.percentile(x, q)) if len(x) else None for q in [25, 50, 75, 90, 95]},
              **scene_bootstrap(scene_means.to_dict("records"), metric, cfg["analysis"]["seed"], cfg["analysis"]["bootstrap_iterations"])}
    ci = result.pop("ci95")
    result.pop("scene_means")
    result["ci95_low"], result["ci95_high"] = ci or [None, None]
    return result


def grouped(frame, dimensions, cfg, metrics=METRICS):
    frame = frame[list(dict.fromkeys(dimensions+metrics+["scene_token"]))]
    rows = []
    for keys, group in frame.groupby(dimensions, dropna=False, observed=True, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric in metrics:
            rows.append({**dict(zip(dimensions, keys)), "metric": metric, **summary(group, metric, cfg)})
    return pd.DataFrame(rows, columns=dimensions+["metric", "n", "excluded", "mean", "std", "p25", "p50", "p75", "p90", "p95", "scenes", "scene_equal_weight_mean", "bootstrap_unit", "ci95_low", "ci95_high"])


def temporal_group(dt, cfg):
    gaps = np.asarray(cfg["analysis"]["time_gap_groups"])
    chosen = float(gaps[np.argmin(abs(gaps-dt))])
    return str(chosen) if abs(chosen-dt) <= cfg["analysis"]["time_gap_tolerance_seconds"] else "other"
