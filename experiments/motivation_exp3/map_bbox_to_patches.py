"""Exact pad/resize/crop coordinate bookkeeping; patch 0 excludes CLS."""
import json
import numpy as np
import pandas as pd

from .utils.common import read, save, standalone, write_json


def preprocessing_geometry(original_size, processor, aspect_ratio="pad"):
    """Match llava.mm_utils.process_images and installed CLIPImageProcessor.

    Coordinates are continuous pixel edges. Resize uses the actual rounded image
    dimensions, with independent x/y scales. HF center crop can zero-pad inputs.
    Unsupported anyres/S2 must never be approximated by a single affine mapping.
    """
    config = processor.to_dict() if hasattr(processor, "to_dict") else processor
    w, h = map(int, original_size)
    if min(w, h) <= 0 or aspect_ratio == "anyres":
        raise ValueError("Invalid image dimensions or unsupported anyres spatial mapping")
    sx = sy = 1.0
    tx = ty = 0.0
    operations = []
    if aspect_ratio == "pad":
        side = max(w, h)
        dx, dy = (side-w)//2, (side-h)//2
        tx, ty = dx, dy
        operations.append({"operation": "expand2square", "left": dx, "top": dy, "size": [side, side]})
        w = h = side
    if config.get("do_resize", True):
        size = config["size"]
        size = {"shortest_edge": size} if isinstance(size, int) else size
        if "shortest_edge" in size:
            short = int(size["shortest_edge"])
            nw, nh = (short, int(short*h/w)) if w <= h else (int(short*w/h), short)
        elif "height" in size and "width" in size:
            nw, nh = int(size["width"]), int(size["height"])
        else:
            raise ValueError(f"Unsupported resize strategy: {size}")
        rx, ry = nw/w, nh/h
        sx, sy, tx, ty = sx*rx, sy*ry, tx*rx, ty*ry
        operations.append({"operation": "resize", "from": [w, h], "to": [nw, nh]})
        w, h = nw, nh
    if config.get("do_center_crop", True):
        size = config["crop_size"]
        size = {"height": size, "width": size} if isinstance(size, int) else size
        cw, ch = int(size["width"]), int(size["height"])
        left, top = (w-cw)//2, (h-ch)//2
        # Match transformers.image_transforms.center_crop, including small inputs.
        pw, ph = max(w, cw), max(h, ch)
        pad_x, pad_y = (pw-w)//2, (ph-h)//2
        crop_x, crop_y = max(0, left+pad_x), max(0, top+pad_y)
        tx += pad_x-crop_x
        ty += pad_y-crop_y
        nw, nh = min(pw, left+cw+pad_x)-crop_x, min(ph, top+ch+pad_y)-crop_y
        operations.append({"operation": "center_crop", "left": left, "top": top,
                           "padding": [pad_x, pad_y], "output_size": [nw, nh]})
        w, h = nw, nh
    return {"original_size": list(original_size), "input_size": [w, h],
            "scale": [sx, sy], "offset": [tx, ty], "operations": operations,
            "do_normalize": config.get("do_normalize", True), "image_mean": config.get("image_mean"),
            "image_std": config.get("image_std")}


def map_original_bbox_to_model_input(original_size, bbox, image_processor=None,
                                     transform=None, aspect_ratio="pad"):
    geometry = transform or preprocessing_geometry(original_size, image_processor, aspect_ratio)
    b = np.asarray(bbox, dtype=float)
    ow, oh = original_size
    if b.shape != (4,) or not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
        return {"valid": False, "reason": "invalid_bbox", "retained_area_ratio": 0.0}
    # First respect the actual original camera image boundaries.
    clipped = b.copy()
    clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, ow)
    clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, oh)
    sx, sy = geometry["scale"]
    tx, ty = geometry["offset"]
    mapped = clipped * [sx, sy, sx, sy] + [tx, ty, tx, ty]
    raw = b * [sx, sy, sx, sy] + [tx, ty, tx, ty]
    w, h = geometry["input_size"]
    mapped[[0, 2]] = np.clip(mapped[[0, 2]], 0, w)
    mapped[[1, 3]] = np.clip(mapped[[1, 3]], 0, h)
    area = max(0, mapped[2]-mapped[0])*max(0, mapped[3]-mapped[1])
    retained = area/((raw[2]-raw[0])*(raw[3]-raw[1]))
    return {"bbox": mapped.tolist(), "valid": bool(area > 0),
            "reason": "valid" if area > 0 else "completely_cropped",
            "retained_area_ratio": float(retained), "input_size": [w, h],
            "valid_crop_range_original": [max(0, -tx/sx), max(0, -ty/sy),
                                          min(ow, (w-tx)/sx), min(oh, (h-ty)/sy)],
            "transform": geometry}


def patch_weights(bbox, input_size, grid_shape, method="area", min_coverage=0.25):
    """Return row-major spatial indices and intersection-area weights (no CLS)."""
    w, h = input_size
    rows, cols = grid_shape
    x, y = np.meshgrid(np.arange(cols)*w/cols, np.arange(rows)*h/rows)
    pw, ph = w/cols, h/rows
    x0, y0, x1, y1 = bbox
    areas = np.maximum(0, np.minimum(x+pw, x1)-np.maximum(x, x0)) * np.maximum(0, np.minimum(y+ph, y1)-np.maximum(y, y0))
    if method == "center":
        selected = (x+pw/2 >= x0) & (x+pw/2 < x1) & (y+ph/2 >= y0) & (y+ph/2 < y1)
    elif method in {"area", "mean"}:
        selected = (areas > 0) & (areas/(pw*ph) >= min_coverage)
    else:
        raise ValueError(f"Unknown aggregation: {method}")
    indices = np.flatnonzero(selected)
    weights = areas.ravel()[indices] if method == "area" else np.ones(len(indices))
    return indices, weights.astype(np.float32)


def mapping_rows(observations, processor, model_config, vision_config, cfg):
    result = []
    patch = vision_config["patch_size"]
    for obs in observations.to_dict("records"):
        geometry = preprocessing_geometry([obs["image_width"], obs["image_height"]], processor,
                                          model_config.get("image_aspect_ratio", "square"))
        size = geometry["input_size"]
        if any(x % patch for x in size) or size != [vision_config["image_size"]]*2:
            raise ValueError(f"Processor / CLIP geometry mismatch: {size}, patch={patch}")
        grid = [size[1]//patch, size[0]//patch]
        mapped = map_original_bbox_to_model_input(geometry["original_size"], json.loads(obs["bbox"]), transform=geometry)
        for method in ["area", "mean", "center"]:
            indices, weights = patch_weights(mapped.get("bbox", [0]*4), size, grid, method, cfg["data"]["min_patch_coverage_ratio"])
            reason = mapped["reason"]
            if mapped["valid"] and mapped["retained_area_ratio"] < cfg["data"]["min_retained_area_ratio"]:
                reason = "low_retained_area"
            elif mapped["valid"] and len(indices) < cfg["data"]["min_valid_patch_count"]:
                reason = "insufficient_patches"
            result.append({**obs, "aggregation": method, "mapped_bbox": json.dumps(mapped.get("bbox")),
                           "retained_area_ratio": mapped["retained_area_ratio"], "patch_count": len(indices),
                           "patch_indices": json.dumps(indices.tolist()), "patch_weights": json.dumps(weights.tolist()),
                           "mapping_valid": reason == "valid", "mapping_status": reason,
                           "grid_rows": grid[0], "grid_cols": grid[1], "transform": json.dumps(geometry)})
    return pd.DataFrame(result)


def run(cfg, out):
    from .extract_senna_features import inspect_model
    meta, processor = inspect_model(cfg)
    result = mapping_rows(read(out, "object_observations"), processor, meta["model_config"], meta["vision_config"], cfg)
    save(out, "object_patch_mapping", result)
    write_json(out/"metadata/mapping_summary.json", {"counts_by_method_and_status":
        result.groupby(["aggregation", "mapping_status"]).size().reset_index(name="n").to_dict("records"),
        "patch_indices_exclude_cls": True})


if __name__ == "__main__":
    standalone("map-patches")
