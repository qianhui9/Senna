"""Download and verify the pinned Senna and CLIP inference checkpoints."""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/qh/phdCode/work4/.cache/Senna/huggingface")
if not os.environ.get("HTTPS_PROXY"):
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path("/data/qh/phdCode/work4/models")
SENNA_DIR = ROOT / "Senna"
CLIP_DIR = ROOT / "clip-vit-large-patch14-336"

SENNA_REVISION = "48aa4218b586ce89c1778a5dedecbbe0e2887fcf"
CLIP_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"

CHECKSUMS = {
    SENNA_DIR / "pytorch_model-00001-of-00002.bin": "706b29a5eadde68b0f36e421683e96751bba7b830234aaf6dbdc3e032af766fa",
    SENNA_DIR / "pytorch_model-00002-of-00002.bin": "4018dec4a10459b270d78b3944ae666a21939ce8b0608f61cc9a374ecd8adf1a",
    CLIP_DIR / "pytorch_model.bin": "c6032c2e0caae3dc2d4fba35535fa6307dbb49df59c7e182b1bc4b3329b81801",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retry(label, operation, attempts=60):
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == attempts:
                raise
            delay = min(15 * attempt, 60)
            print(
                f"{label} failed on attempt {attempt}/{attempts}: {error}\n"
                f"Keeping partial files; retrying in {delay} seconds.",
                flush=True,
            )
            time.sleep(delay)


def main():
    SENNA_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    senna_sizes = {
        SENNA_DIR / "pytorch_model-00001-of-00002.bin": 9_976_631_486,
        SENNA_DIR / "pytorch_model-00002-of-00002.bin": 4_149_654_020,
    }
    senna_downloads = []
    for path, size in senna_sizes.items():
        if not path.is_file() or path.stat().st_size != size:
            operation = partial(
                hf_hub_download,
                "rb93dett/Senna",
                path.name,
                revision=SENNA_REVISION,
                local_dir=str(SENNA_DIR),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            senna_downloads.append((path.name, operation))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(retry, f"Senna download: {name}", operation)
            for name, operation in senna_downloads
        ]
        for future in futures:
            future.result()

    retry(
        "Senna configuration download",
        lambda: snapshot_download(
            "rb93dett/Senna",
            revision=SENNA_REVISION,
            local_dir=str(SENNA_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=["*.json", "tokenizer.model", "README.md", ".gitattributes"],
            max_workers=2,
        ),
    )

    clip_weight = CLIP_DIR / "pytorch_model.bin"
    if not clip_weight.is_file() or clip_weight.stat().st_size != 1_711_974_081:
        retry(
            "CLIP weight download",
            partial(
                hf_hub_download,
                "openai/clip-vit-large-patch14-336",
                "pytorch_model.bin",
                revision=CLIP_REVISION,
                local_dir=str(CLIP_DIR),
                local_dir_use_symlinks=False,
                resume_download=True,
            ),
        )
    retry(
        "CLIP configuration download",
        lambda: snapshot_download(
            "openai/clip-vit-large-patch14-336",
            revision=CLIP_REVISION,
            local_dir=str(CLIP_DIR),
            local_dir_use_symlinks=False,
            allow_patterns=["config.json", "preprocessor_config.json"],
            resume_download=True,
            max_workers=3,
        ),
    )

    config_path = SENNA_DIR / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["mm_vision_tower"] = str(CLIP_DIR)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for path, expected in CHECKSUMS.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {path}: {actual} != {expected}")
        print(f"OK {actual}  {path}")


if __name__ == "__main__":
    main()
