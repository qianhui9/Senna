"""FP32 offline pooling of original full-image spatial tensors."""
import json
import numpy as np
import pandas as pd

from .utils.common import read, save, standalone, write_json


def aggregate_patches(patches, indices, weights):
    patches = np.asarray(patches, dtype=np.float32)
    indices, weights = np.asarray(indices, int), np.asarray(weights, np.float32)
    if patches.ndim != 2 or len(indices) != len(weights) or np.any(indices < 0) or np.any(indices >= len(patches)):
        raise ValueError("Invalid patch tensor/index/weight shape")
    if not len(indices) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        return None
    selected = patches[indices]
    if not np.isfinite(selected).all():
        return None
    return np.sum(selected*weights[:, None], axis=0, dtype=np.float32)/np.sum(weights, dtype=np.float32)


def run(cfg, out):
    """Extraction pools immediately; this stage validates and indexes its shards."""
    from .extract_senna_features import load_shard
    observations = read(out, "object_observations")
    rows, failures = [], []
    for token, group in observations.groupby("sample_data_token", sort=True):
        path = out/"features/object_feature_shards"/(token+".npz")
        if not path.exists():
            failures.append({"sample_data_token": token, "reason": "missing_feature_shard"})
            continue
        meta, arrays = load_shard(path)
        if set(meta["observation_ids"]) != set(group.observation_id):
            raise ValueError(f"Shard observation identity mismatch: {token}")
        for i, obs in enumerate(meta["observations"]):
            rows.append({**obs, "shard": str(path), "shard_row": i,
                         "layers": json.dumps(meta["layers"]), "methods": json.dumps(["area", "mean", "center"])})
        for layer in meta["layers"]:
            for method in ["area", "mean", "center"]:
                if arrays[f"{layer}__{method}"].shape[0] != len(meta["observation_ids"]):
                    raise ValueError("Shard descriptor count mismatch")
    save(out, "object_feature_index", pd.DataFrame(rows, columns=list(observations.columns)+["shard", "shard_row", "layers", "methods"]))
    write_json(out/"metadata/object_feature_summary.json", {"observations_indexed": len(rows), "missing_images": failures,
        "descriptor": "Offline bbox patch pooling, includes background and global attention context; not native object tokens"})
    if failures:
        raise RuntimeError(f"{len(failures)} images have missing shards; resume extract-features")


if __name__ == "__main__":
    standalone("object-features")
