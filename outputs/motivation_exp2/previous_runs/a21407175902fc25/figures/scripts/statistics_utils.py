"""Scene replication, scene-equal means and explicit observation weighting."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from experiments.motivation_exp1.utils.stats_utils import scene_bootstrap


def metric_summary(df, metric, cfg):
    values = pd.to_numeric(df[metric], errors="coerce")
    valid = df.loc[np.isfinite(values)].copy()
    valid[metric] = values[np.isfinite(values)]
    data = valid[metric].to_numpy(float)
    scene_means = valid.groupby("scene_token")[metric].mean()
    # Reuse E1's scene bootstrap implementation on one mean per scene.
    bootstrap = scene_bootstrap([{"scene_token": k, metric: v} for k, v in scene_means.items()],
                                metric, cfg["analysis"]["seed"], cfg["analysis"]["bootstrap_iterations"])
    result = {"n": len(data), "excluded_nonfinite": len(df)-len(data),
              "observation_weighted_mean": float(data.mean()) if len(data) else None,
              **{f"p{q}": float(np.percentile(data, q)) if len(data) else None for q in [50, 75, 90, 95]},
              **bootstrap}
    if metric == "persistent_ratio":
        denom = valid.objects_t0.sum()
        result["instance_weighted_ratio"] = float(valid.persistent_objects.sum()/denom) if denom else None
        result["weighting_note"] = "Observation mean weights valid keyframe pairs equally; instance ratio pools counts; scene mean weights scene pair-means equally"
    return result


def grouped_summary(df, dimensions, metrics, cfg):
    result = []
    for keys, group in df.groupby(dimensions, dropna=False, observed=True, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        result.append({**dict(zip(dimensions, keys)), "rows": len(group),
                       "metrics": {m: metric_summary(group, m, cfg) for m in metrics}})
    return result


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def correlation_summary(df, x, y, cfg, cluster_ci=True):
    good = df[np.isfinite(df[x]) & np.isfinite(df[y])]
    medians = good.groupby("scene_token")[[x, y]].median()
    result = {"n": len(good), "scenes": len(medians), "observation_spearman": spearman(good[x], good[y]),
              "scene_median_spearman": spearman(medians[x], medians[y]), "scene_median_ci95": None,
              "bootstrap_unit": "scene", "iid_p_value": None,
              "note": "Observation rho is descriptive; scene rho compares scene medians; no independent-object/frame p-value"}
    if cluster_ci and len(medians) >= 3 and result["scene_median_spearman"] is not None:
        rng = np.random.default_rng(cfg["analysis"]["seed"])
        draws = rng.integers(len(medians), size=(cfg["analysis"]["bootstrap_iterations"], len(medians)))
        rx, ry = [rankdata(medians[col].to_numpy()[draws], axis=1) for col in [x, y]]
        rx -= rx.mean(axis=1, keepdims=True)
        ry -= ry.mean(axis=1, keepdims=True)
        denom = np.sqrt((rx*rx).sum(axis=1)*(ry*ry).sum(axis=1))
        rho = np.divide((rx*ry).sum(axis=1), denom, out=np.full(len(draws), np.nan), where=denom > 0)
        finite = rho[np.isfinite(rho)]
        result["valid_bootstrap_draws"] = len(finite)
        if len(finite):
            result["scene_median_ci95"] = np.percentile(finite, [2.5, 97.5]).tolist()
    return result
