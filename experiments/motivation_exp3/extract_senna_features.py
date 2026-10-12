"""Read-only hooks on the original encode_images -> mm_project path."""
from contextlib import contextmanager
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from .utils.common import LOG, read, json_read, write_json, standalone, digest
from .map_bbox_to_patches import mapping_rows, preprocessing_geometry
from .extract_object_features import aggregate_patches


def resolve_layers(depth, requested, selected):
    indices = list(requested) if requested else [0, depth//4, depth//2, 3*depth//4, depth]
    if any(type(i) is not int or i < 0 or i > depth for i in indices):
        raise ValueError(f"Hidden state indices must be in [0,{depth}]; 0=embedding after pre-LN, i=block i output")
    actual_selected = selected if selected >= 0 else depth+1+selected
    if not 0 <= actual_selected <= depth:
        raise ValueError("Invalid Senna selected hidden-state layer")
    return sorted(set(indices+[actual_selected]))


def inspect_model(cfg):
    from transformers import CLIPVisionConfig, CLIPImageProcessor
    cp = Path(cfg["senna"]["checkpoint_path"])
    if not (cp/"config.json").exists():
        raise FileNotFoundError("Set --checkpoint-path or SENNA_MODEL_PATH (E1 fallback has no usable checkpoint)")
    model_config = json_read(cp/"config.json")
    if model_config.get("s2", False):
        raise ValueError("S2 checkpoint requires per-scale/tile mapping. This single-scale extractor refuses S2 instead of fabricating patch correspondence")
    if model_config.get("image_aspect_ratio") == "anyres":
        raise ValueError("anyres requires separate tile mappings; unsupported by this single-scale experiment")
    tower = model_config["mm_vision_tower"]
    vc = CLIPVisionConfig.from_pretrained(tower, local_files_only=True).to_dict()
    processor = CLIPImageProcessor.from_pretrained(tower, local_files_only=True)
    layers = resolve_layers(vc["num_hidden_layers"], cfg["senna"]["feature_layers"], model_config["mm_vision_select_layer"])
    metadata = {"checkpoint": str(cp), "model_config": model_config, "vision_tower_class": "CLIPVisionTower",
                "vision_tower_name": tower, "vision_config": vc, "feature_layers": layers,
                "selected_layer": model_config["mm_vision_select_layer"] % (vc["num_hidden_layers"]+1),
                "selected_feature": model_config.get("mm_vision_select_feature", "patch"),
                "precision": cfg["senna"]["precision"], "device": cfg["senna"]["device"],
                "grid_shape": [vc["image_size"]//vc["patch_size"]]*2,
                "spatial_projector": bool(model_config.get("mm_projector_type", "linear") in {"linear", "identity"} or re.fullmatch(r"mlp\d+x_gelu", model_config.get("mm_projector_type", "linear"))),
                "adapter_spatial_mapping": False, "adapter_note": "Learned token-axis Linear/GELU network; globally mixes 576 into 128 tokens. Only global analysis permitted.",
                "layer_zero_note": "CLIP hidden_states[0] is embedding+position after pre_layernorm, before transformer block 1",
                "status": "configuration_inspected_runtime_unverified"}
    if metadata["selected_feature"] != "patch":
        raise ValueError("Senna 576-token adapter requires patch selection; refusing unsupported cls_patch checkpoint")
    index = json_read(cp/"pytorch_model.bin.index.json", {})
    for path in set(index.get("weight_map", {}).values()):
        if not (cp/path).is_file():
            raise FileNotFoundError(cp/path)
    return metadata, processor


@contextmanager
def capture_features(model, layers, include_projector=True):
    """Hooks return None and copy only selected original outputs to CPU."""
    captured, shapes, handles = {}, {}, []
    def vision_hook(module, inputs, outputs):
        for layer in layers:
            value = outputs.hidden_states[layer]
            if value.ndim != 3:
                raise ValueError("Expected [batch, CLS+patches, channels] hidden states")
            captured[f"layer_{layer}"] = value[:, 1:].detach().cpu()
            shapes[f"layer_{layer}"] = {"layer_id": layer, "shape": list(value.shape),
                "patch_shape": list(value[:, 1:].shape), "contains_cls_before_removal": True,
                "dtype": str(value.dtype), "spatial_mapping": True, "spatial_downsampling": False}
    handles.append(model.get_vision_tower().vision_tower.register_forward_hook(vision_hook))
    if include_projector:
        def projector_hook(module, inputs, outputs):
            captured["projector"] = outputs.detach().cpu()
            shapes["projector"] = {"shape": list(outputs.shape), "dtype": str(outputs.dtype),
                                   "contains_cls_before_removal": False, "spatial_mapping": True}
        handles.append(model.get_model().mm_projector.register_forward_hook(projector_hook))
    def adapter_hook(module, inputs, outputs):
        captured["adapter_global"] = outputs.detach().float().mean(dim=1).cpu()
        shapes["adapter_global"] = {"shape": list(outputs.shape), "dtype": str(outputs.dtype),
                                    "spatial_mapping": False, "analysis": "global mean only"}
    handles.append(model.get_model().img_adapter.register_forward_hook(adapter_hook))
    try:
        yield captured, shapes
    finally:
        for handle in handles:
            handle.remove()


def original_visual_forward(model, pixels):
    selected = model.encode_images(pixels)
    compressed = model.mm_project(selected.unsqueeze(1))
    return selected, __import__("torch").stack(compressed)


def compare_tensors(a, b, atol, rtol):
    import torch
    same_layout = a.shape == b.shape and a.dtype == b.dtype
    difference = (a.float()-b.float()).abs()
    return {"shape": list(a.shape), "dtype": str(a.dtype), "same_layout": same_layout,
            "max_absolute_error": float(difference.max()),
            "passed": bool(same_layout and torch.allclose(a, b, atol=atol, rtol=rtol))}


def save_shard(path, metadata, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata)), **arrays)
    tmp.replace(path)


def load_shard(path):
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata"])), {k: data[k] for k in data.files if k != "metadata"}


def fixed_positions(size, grid, diagnostic_grid):
    rows, cols = grid
    nr, nc = diagnostic_grid
    yy, xx = np.meshgrid(np.minimum(rows-1, ((np.arange(nr)+.5)*rows/nr).astype(int)),
                         np.minimum(cols-1, ((np.arange(nc)+.5)*cols/nc).astype(int)), indexing="ij")
    return np.unique((yy*cols+xx).ravel())


def run(cfg, out):
    import torch
    import transformers
    from data_tools.senna_inputs import check_model
    from llava.model.builder import load_senna_pretrained_model
    from llava.mm_utils import process_images
    metadata, _ = inspect_model(cfg)
    device = cfg["senna"]["device"]
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; execute on a GPU host. CPU build-pairs/map-patches remain executable")
    check_model(cfg["senna"]["checkpoint_path"])
    torch.manual_seed(cfg["analysis"]["seed"])
    torch.cuda.set_device(torch.device(device))
    observations = read(out, "object_observations")
    old_status = json_read(out/"metadata/extraction_status.json", {})
    groups = list(observations.groupby("sample_data_token", sort=True))
    # Validate completed files by digest before accepting their offline descriptors.
    pending = [(token, group) for token, group in groups if not (
        (out/"features/object_feature_shards"/(token+".npz")).exists() and
        old_status.get("shards", {}).get(token) == digest(out/"features/object_feature_shards"/(token+".npz")))]
    if not pending and old_status.get("complete"):
        LOG.info("All %d verified offline shards already complete", len(groups))
        return
    LOG.info("Loading original Senna checkpoint; %d image forwards pending", len(pending))
    _, model, processor, _ = load_senna_pretrained_model(cfg["senna"]["checkpoint_path"], None, "llava", device_map=device)
    model.eval()
    layers = metadata["feature_layers"]
    tower = model.get_vision_tower()
    metadata.update({"loaded_model_class": type(model).__name__, "vision_tower_class": type(tower).__name__,
                     "projector_structure": str(model.get_model().mm_projector),
                     "img_adapter_structure": str(model.get_model().img_adapter),
                     "actual_vision_dtype": str(tower.dtype), "actual_device": str(tower.device),
                     "torch": torch.__version__, "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(), "status": "runtime_verified"})
    if type(tower).__name__ != "CLIPVisionTower":
        raise ValueError("Only a verified single-scale CLIPVisionTower may be mapped")
    projector_spatial = metadata["spatial_projector"] and cfg["senna"]["extract_projector_features"]
    mappings = mapping_rows(observations, processor, model.config.to_dict(), tower.config.to_dict(), cfg)
    checks = [c for c in json_read(out/"metadata/output_equivalence.json", []) if c.get("passed")]
    status = {"complete": False, "shards": old_status.get("shards", {}), "failed_images": [], "images_total": len(groups)}
    raw_files = list((out/"features/raw_debug").glob("*.npz"))
    raw_bytes = sum(p.stat().st_size for p in raw_files)
    raw_count = len(raw_files)

    def encode_batch(batch):
        nonlocal raw_bytes, raw_count
        images = []
        for token, group in batch:
            paths = group.image_path.unique()
            if len(paths) != 1:
                raise ValueError("One sample_data_token points to multiple image paths")
            with Image.open(paths[0]) as im:
                rgb = im.convert("RGB")
            if rgb.size != tuple(group[["image_width", "image_height"]].iloc[0].astype(int)):
                raise ValueError(f"Actual image dimensions contradict E2: {token}")
            images.append(rgb)
        pixels = process_images(images, processor, model.config)
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
            raise ValueError("Expected full-image BCHW tensor from original preprocessing")
        for im in images:
            expected = preprocessing_geometry(im.size, processor, model.config.image_aspect_ratio)["input_size"]
            if list(pixels.shape[-2:][::-1]) != expected:
                raise ValueError("Recorded preprocessing geometry disagrees with real pixel tensor")
        pixels = pixels.to(device=tower.device, dtype=tower.dtype)
        with torch.inference_mode():
            do_validate = len(checks) < cfg["senna"]["validation_images"]
            if do_validate:
                ref_selected, ref_compressed = original_visual_forward(model, pixels)
            with capture_features(model, layers, projector_spatial) as (captured, shapes):
                selected, compressed = original_visual_forward(model, pixels)
            if do_validate:
                check = {"sample_data_tokens": [t for t, _ in batch], "atol": cfg["senna"]["equivalence_atol"], "rtol": cfg["senna"]["equivalence_rtol"],
                         "vision_output": compare_tensors(ref_selected, selected, cfg["senna"]["equivalence_atol"], cfg["senna"]["equivalence_rtol"]),
                         "compressed_output": compare_tensors(ref_compressed, compressed, cfg["senna"]["equivalence_atol"], cfg["senna"]["equivalence_rtol"])}
                with capture_features(model, layers, projector_spatial) as (repeated, _):
                    original_visual_forward(model, pixels)
                check["repeated_hidden_outputs"] = {k: compare_tensors(captured[k], repeated[k], cfg["senna"]["equivalence_atol"], cfg["senna"]["equivalence_rtol"]) for k in captured}
                from .analyze_feature_similarity import feature_metrics
                repeated_descriptors = []
                for i, (token, _) in enumerate(batch):
                    for m in mappings[(mappings.sample_data_token == token) & (mappings.aggregation == "area") & mappings.mapping_valid].to_dict("records"):
                        indices, weights = json.loads(m["patch_indices"]), json.loads(m["patch_weights"])
                        for layer in [k for k in captured if k != "adapter_global"]:
                            a = aggregate_patches(captured[layer][i].float().numpy(), indices, weights)
                            b = aggregate_patches(repeated[layer][i].float().numpy(), indices, weights)
                            if a is not None and b is not None:
                                repeated_descriptors.append({"sample_data_token": token, "observation_id": m["observation_id"],
                                    "scene_token": m["scene_token"], "layer_id": layer,
                                    **feature_metrics(a.astype(np.float16), b.astype(np.float16))})
                check["repeated_object_descriptors"] = repeated_descriptors
                check["passed"] = all(x["passed"] for x in [check["vision_output"], check["compressed_output"], *check["repeated_hidden_outputs"].values()])
                checks.append(check)
                write_json(out/"metadata/output_equivalence.json", checks)
                if not check["passed"]:
                    raise RuntimeError("Instrumented or repeated forward exceeds numerical tolerance; reject extraction")
                del ref_selected, ref_compressed, repeated
            metadata["runtime_shapes"] = shapes
            metadata["input_shape"] = list(pixels.shape)
            write_json(out/"metadata/model_metadata.json", metadata)
            write_json(out/"metadata/image_processor_config.json", processor.to_dict())
            spatial = [k for k in captured if k != "adapter_global"]
            grid = metadata["grid_shape"]
            if any(captured[k].shape[1] != grid[0]*grid[1] for k in spatial):
                raise ValueError("Runtime tensor has no verified one-to-one patch layout")
            fixed = fixed_positions(pixels.shape[-2:][::-1], grid, cfg["analysis"]["fixed_position_grid"])
            for i, (token, group) in enumerate(batch):
                group = group.reset_index(drop=True)
                relevant = mappings[mappings.sample_data_token == token]
                arrays = {}
                for layer in spatial:
                    patches = captured[layer][i].float().numpy()
                    for method in ["area", "mean", "center"]:
                        lookup = relevant[relevant.aggregation == method].set_index("observation_id")
                        vectors = []
                        for obs in group.observation_id:
                            m = lookup.loc[obs]
                            vec = aggregate_patches(patches, json.loads(m.patch_indices), json.loads(m.patch_weights)) if m.mapping_valid else None
                            vectors.append(vec if vec is not None else np.full(patches.shape[-1], np.nan, np.float32))
                        arrays[f"{layer}__{method}"] = np.asarray(vectors, dtype=np.float16)
                    arrays[f"{layer}__fixed"] = patches[fixed].astype(np.float16)
                    arrays[f"{layer}__global"] = patches.mean(axis=0, dtype=np.float32).astype(np.float16)
                arrays["adapter_global"] = captured["adapter_global"][i].float().numpy().astype(np.float16)
                geometry = preprocessing_geometry(images[i].size, processor, model.config.image_aspect_ratio)
                sx, sy = geometry["scale"]
                tx, ty = geometry["offset"]
                centers_x = (fixed % grid[1]+.5)*tower.config.patch_size
                centers_y = (fixed // grid[1]+.5)*tower.config.patch_size
                ow, oh = images[i].size
                valid_fixed = (centers_x >= tx) & (centers_x < tx+ow*sx) & (centers_y >= ty) & (centers_y < ty+oh*sy)
                shard_meta = {"sample_data_token": token, "observation_ids": group.observation_id.tolist(),
                              "observations": json.loads(group.to_json(orient="records")), "layers": spatial,
                              "fixed_indices": fixed.tolist(), "fixed_valid": valid_fixed.tolist(),
                              "preprocessing_geometry": geometry, "feature_dtype": "float16", "pooling_dtype": "float32"}
                path = out/"features/object_feature_shards"/(token+".npz")
                save_shard(path, shard_meta, arrays)
                status["shards"][token] = digest(path)
                write_json(out/"metadata/extraction_status.json", status)
                if cfg["output"]["save_raw_features"] and raw_count < cfg["output"]["raw_feature_max_images"]:
                    raw = {k: captured[k][i].float().numpy().astype(np.float16) for k in spatial}
                    estimated = sum(v.nbytes for v in raw.values())
                    if raw_bytes+estimated <= cfg["output"]["raw_feature_budget_mb"]*1024**2:
                        raw_path = out/"features/raw_debug"/(token+".npz")
                        save_shard(raw_path, shard_meta, raw)
                        raw_bytes += raw_path.stat().st_size
                        raw_count += 1
                LOG.info("Extracted %d/%d images: %s (%d objects)", len(status["shards"]), len(groups), token, len(group))
        del captured, selected, compressed, pixels

    for start in range(0, len(pending), cfg["senna"]["batch_size"]):
        batch = pending[start:start+cfg["senna"]["batch_size"]]
        try:
            encode_batch(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) > 1:
                LOG.warning("CUDA OOM: retry this batch with one full image at a time")
                for item in batch:
                    try:
                        encode_batch([item])
                    except Exception as exc:
                        status["failed_images"].append({"sample_data_token": item[0], "reason": str(exc), "during_oom_retry": True})
                        write_json(out/"metadata/extraction_status.json", status)
                        raise RuntimeError("Single-image retry failed; see extraction_status.json. Completed offline shards are resumable") from exc
            else:
                status["failed_images"].append({"sample_data_token": batch[0][0], "reason": "CUDA OOM at batch size 1; free GPU memory"})
                write_json(out/"metadata/extraction_status.json", status)
                raise RuntimeError("CUDA OOM at batch size 1. Offline completed image shards are resumable")
        except Exception as exc:
            status["failed_images"].extend({"sample_data_token": t, "reason": str(exc)} for t, _ in batch)
            write_json(out/"metadata/extraction_status.json", status)
            raise
    status["complete"] = len(status["shards"]) == len(groups)
    write_json(out/"metadata/extraction_status.json", status)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    standalone("extract-features")
