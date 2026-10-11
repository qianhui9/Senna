"""Senna-specific LoRA plus visual-adapter checkpoint IO (trusted local files only)."""

import hashlib
import json
from pathlib import Path
import re

import torch

MANIFEST = "senna_lora_manifest.json"
EXTRA_WEIGHTS = "non_lora_trainables.bin"
VISUAL_PREFIXES = ("model.mm_projector.", "model.img_adapter.")
CHECKPOINT_MARKER = "checkpoint_complete.json"
RESUME_FILES = (MANIFEST, "adapter_config.json", "adapter_model.bin", EXTRA_WEIGHTS,
                "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth",
                "tokenizer.model", "tokenizer_config.json", "config.json")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def mark_complete(directory):
    directory = Path(directory)
    payload = {name: {"size": (directory / name).stat().st_size, "sha256": digest(directory / name)}
               for name in RESUME_FILES}
    temporary = directory / (CHECKPOINT_MARKER + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(directory / CHECKPOINT_MARKER)


def latest_complete_checkpoint(output):
    candidates = [p for p in Path(output).iterdir() if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)]
    for directory in sorted(candidates, key=lambda p: int(p.name.rsplit("-", 1)[-1]), reverse=True):
        try:
            marker = json.loads((directory / CHECKPOINT_MARKER).read_text())
            if set(marker) != set(RESUME_FILES):
                raise ValueError("Incomplete checkpoint marker")
            for name, identity in marker.items():
                path = directory / name
                if path.stat().st_size != identity["size"] or digest(path) != identity["sha256"]:
                    raise ValueError(f"Incomplete/corrupt checkpoint: {name}")
            return str(directory)
        except (OSError, ValueError, KeyError, TypeError):
            print(f"Skipping incomplete checkpoint: {directory}", flush=True)
    return None


def base_identity(path):
    path = Path(path).resolve()
    index = json.loads((path / "pytorch_model.bin.index.json").read_text())
    files = sorted(set(index["weight_map"].values())) + ["tokenizer.model"]
    return {"path": str(path), "config_sha256": digest(path / "config.json"),
            "files": {name: {"size": (path / name).stat().st_size,
                              "mtime_ns": (path / name).stat().st_mtime_ns} for name in files}}


def save_visual_weights(model, directory):
    base = model.get_base_model()
    state = {name: value.detach().cpu() for name, value in base.named_parameters()
             if value.requires_grad and name.startswith(VISUAL_PREFIXES)}
    torch.save(state, Path(directory) / EXTRA_WEIGHTS)


def load_visual_weights(model, directory):
    state = torch.load(Path(directory) / EXTRA_WEIGHTS, map_location="cpu")
    base = model.get_base_model()
    parameters = dict(base.named_parameters())
    manifest_path = Path(directory) / MANIFEST
    if manifest_path.exists():
        frozen = json.loads(manifest_path.read_text())["settings"]["freeze_visual_adapters"]
        expected = set() if frozen else {name for name in parameters if name.startswith(VISUAL_PREFIXES)}
        if set(state) != expected:
            raise ValueError("Missing/unexpected trained visual parameters in checkpoint")
    for name, value in state.items():
        if not name.startswith(VISUAL_PREFIXES) or name not in parameters:
            raise ValueError(f"Unexpected Senna visual checkpoint parameter: {name}")
        if value.shape != parameters[name].shape:
            raise ValueError(f"Visual checkpoint shape mismatch: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite visual checkpoint parameter: {name}")
    base.load_state_dict(state, strict=False)


def check_lora(directory, base_path):
    directory = Path(directory)
    for name in (MANIFEST, "adapter_config.json", "adapter_model.bin", EXTRA_WEIGHTS):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Incomplete Senna LoRA checkpoint: {directory / name}")
    manifest = json.loads((directory / MANIFEST).read_text())
    if manifest["base_model"] != base_identity(base_path):
        raise ValueError("LoRA checkpoint belongs to a different base model/configuration")
    return manifest


def load_lora_for_inference(base_model, directory, base_path):
    from peft import PeftModel
    check_lora(directory, base_path)
    model = PeftModel.from_pretrained(base_model, directory, is_trainable=False)
    load_visual_weights(model, directory)
    # PEFT 0.4's CausalLM generate wrapper accepts kwargs only and assumes the
    # standard text-generation API. Senna has a custom positional inputs/images
    # path; merge in memory to use that original multimodal generate unchanged.
    # This does not modify the base files or the saved adapter on disk.
    model = model.merge_and_unload()
    model.config.use_cache = True
    model.eval()
    return model
