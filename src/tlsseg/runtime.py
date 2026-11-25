"""Run metadata, deterministic seeding and strict checkpoint I/O."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}


def environment():
    packages = ("torch", "torchvision", "segmentation-models-pytorch", "timm", "einops", "numpy", "Pillow", "PyYAML")
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    optional = {}
    for name in ("shapely", "openslide-python", "openslide-bin"):
        try:
            optional[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            optional[name] = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "optional_preparation_packages": optional,
        "cuda_runtime": torch.version.cuda, "git_commit": revision,
        "source_sha256": source_hashes(),
        "deterministic_algorithms": "requested with warn_only=True; CUDA interpolation/pooling backward may be nondeterministic",
        "precision": {
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "automatic_mixed_precision": "disabled in training/evaluation/prediction execution contexts",
        },
    }


def select_device(name="auto"):
    # MPS is explicit because scientific operator determinism differs by backend.
    name = ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return device


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # CUDA interpolation/pooling backward in the supplied architecture lacks
    # deterministic implementations. Warn rather than make CUDA training fail.
    torch.use_deterministic_algorithms(True, warn_only=True)


def configure_precision():
    """Use FP32 and disable both CUDA TF32 paths for the patent evaluation protocol."""
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def load_checkpoint(path, device="cpu"):
    """Only the repository's versioned tensor/primitive checkpoint schema."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema; use a checkpoint created by tlsseg train.")
    from .models import PatentSegmenter
    model = PatentSegmenter(**checkpoint["model_config"])
    if checkpoint["architecture"] != model.architecture_config():
        raise ValueError("Checkpoint architecture contract differs from this implementation.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    return model, checkpoint
