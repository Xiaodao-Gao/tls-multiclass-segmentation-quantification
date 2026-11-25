"""Train/validation selection and frozen-checkpoint evaluation."""
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import TileDataset, load_manifest, validate_patient_splits
from .losses import segmentation_loss
from .metrics import ConfusionMatrix
from .models import PatentSegmenter
from .runtime import (
    atomic_checkpoint, atomic_json, environment, load_checkpoint, restore_rng,
    rng_state, seed_all, seed_worker, select_device, sha256, source_hashes, configure_precision,
)

DEFAULT_CONFIG = {
    "initialization": "native_checkpoint",
    "seed": 42, "epochs": 50, "batch_size": 4, "num_workers": 0,
    "patience": 15, "native_lr": 5e-6, "branch_lr": 3e-4,
    "weight_decay": 1e-4, "diffusion_weight": 0.01, "augment": True,
    "model": {"with_dit": True, "timesteps": 50},
}


def read_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    unknown = set(raw) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    cfg = dict(DEFAULT_CONFIG, **raw)
    if cfg["initialization"] not in ("native_checkpoint", "random"):
        raise ValueError("initialization must be native_checkpoint or random.")
    if not isinstance(cfg["model"], dict) or set(cfg["model"]) - {"with_dit", "timesteps"}:
        raise ValueError("model accepts with_dit and timesteps only.")
    cfg["model"] = dict(DEFAULT_CONFIG["model"], **cfg["model"])
    for key in ("epochs", "batch_size", "patience"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("num_workers", "seed"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    for key in ("native_lr", "branch_lr", "weight_decay", "diffusion_weight"):
        if not isinstance(cfg[key], (int, float)) or not 0 <= cfg[key] < float("inf"):
            raise ValueError(f"Invalid {key}.")
    if cfg["native_lr"] <= 0 or cfg["branch_lr"] <= 0:
        raise ValueError("Learning rates must be positive.")
    if type(cfg["augment"]) is not bool or type(cfg["model"]["with_dit"]) is not bool:
        raise ValueError("augment and with_dit must be YAML booleans.")
    if type(cfg["model"]["timesteps"]) is not int or cfg["model"]["timesteps"] < 1:
        raise ValueError("timesteps must be a positive integer.")
    return cfg


def labeled_split(rows, split):
    selected = [r for r in rows if r["split"] == split]
    if not selected:
        raise ValueError(f"Manifest has no {split} samples.")
    if any(not r.get("mask_path") for r in selected):
        raise ValueError(f"Every {split} sample must have a mask for training/evaluation.")
    return selected


def patient_hashes(rows):
    return {
        split: sorted({hashlib.sha256(str(r["patient_id"]).encode("utf-8")).hexdigest()
                       for r in rows if r["split"] == split})
        for split in sorted({r["split"] for r in rows})
    }


def loader_for(rows, cfg, device, train=False, epoch=0):
    generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
    return DataLoader(
        TileDataset(rows, augment=train and cfg["augment"]),
        batch_size=cfg["batch_size"], shuffle=train, num_workers=cfg["num_workers"],
        worker_init_fn=seed_worker, generator=generator, drop_last=False,
        pin_memory=device.type == "cuda",
    )


def run_epoch(model, loader, device, diffusion_weight, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    cm = ConfusionMatrix()
    sums = {"total": 0.0, "ce": 0.0, "dice": 0.0, "noise_mse": 0.0}
    count = batches = skipped = 0
    start = time.perf_counter()
    with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=False):
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if not (masks != 255).any():
                skipped += 1
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            result = model(images)
            loss, parts = segmentation_loss(
                result["logits"], masks, result["noise_mse"], diffusion_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite objective.")
            if training:
                loss.backward()
                optimizer.step()
            n = len(images)
            count += n
            batches += 1
            sums["total"] += float(loss.detach()) * n
            for key in ("ce", "dice", "noise_mse"):
                sums[key] += float(parts[key].detach()) * n
            cm.update(result["logits"].detach().cpu(), masks.detach().cpu())
    if not count:
        raise ValueError("No labeled pixels were processed.")
    report = cm.compute()
    report["objective"] = {key: value / count for key, value in sums.items()}
    if not training:
        # Inference never samples noise; zero in the forward API is not an MSE measurement.
        report["objective"]["noise_mse"] = None
    report.update(
        samples=count, batches=batches, all_ignore_batches_skipped=skipped,
        seconds=time.perf_counter() - start,
        objective_aggregation="Sample-weighted mean of batch objectives; metrics use one global pixel confusion matrix.",
    )
    return report


def make_optimizer(model, cfg):
    native = list(model.network.parameters())
    native_ids = {id(p) for p in native}
    branch = [p for p in model.parameters() if id(p) not in native_ids]
    return torch.optim.AdamW([
        {"params": native, "lr": cfg["native_lr"], "name": "native_unet"},
        {"params": branch, "lr": cfg["branch_lr"], "name": "fusion_refinement"},
    ], weight_decay=cfg["weight_decay"])


def train(args):
    cfg = read_config(args.config)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("epochs must be positive.")
        cfg["epochs"] = args.epochs
    if not args.resume:
        if cfg["initialization"] == "native_checkpoint" and not args.native_checkpoint:
            raise ValueError("The patent recipe requires --native-checkpoint. Use configs/synthetic_smoke.yaml only for an explicit random-initialization software demonstration.")
        if cfg["initialization"] == "random" and args.native_checkpoint:
            raise ValueError("A random-initialization configuration cannot accept --native-checkpoint; use the patent configuration.")
    configure_precision()
    seed_all(cfg["seed"])
    device = select_device(args.device)
    rows = load_manifest(args.manifest)
    summary = validate_patient_splits(rows)
    train_rows, val_rows = labeled_split(rows, "train"), labeled_split(rows, "val")
    manifest_hash = sha256(args.manifest)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output is nonempty. Choose a new directory or explicitly resume last.pt.")
    output.mkdir(parents=True, exist_ok=True)
    history, start_epoch, bad_epochs, best = [], 1, 0, -1.0
    if args.resume:
        if args.native_checkpoint:
            raise ValueError("resume and native-checkpoint are mutually exclusive.")
        resume_path = Path(args.resume).resolve()
        if resume_path.name != "last.pt" or resume_path.parent != output.resolve() or not (output / "best.pt").is_file():
            raise ValueError("Resume last.pt in its original output directory, keeping best.pt alongside it.")
        model, prior = load_checkpoint(args.resume, device)
        for key in cfg:
            if key != "epochs" and cfg[key] != prior["train_config"].get(key):
                raise ValueError(f"Resume configuration changed: {key}.")
        if prior["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume manifest SHA256 mismatch.")
        if prior["environment"]["source_sha256"] != source_hashes():
            raise ValueError("Resume source SHA256 mismatch; start a new run after code changes.")
        optimizer = make_optimizer(model, cfg)
        optimizer.load_state_dict(prior["optimizer_state_dict"])
        # Optimizer tensors were deserialized on CPU; move them to the chosen device.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
        history = prior["history"]
        start_epoch, bad_epochs, best = prior["epoch"] + 1, prior["bad_epochs"], prior["best_val_dice"]
        restore_rng(prior["rng_state"])
    else:
        model = PatentSegmenter(**cfg["model"]).to(device)
        if args.native_checkpoint:
            state = torch.load(args.native_checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.network.load_state_dict(state, strict=True)
        optimizer = make_optimizer(model, cfg)
    if start_epoch > cfg["epochs"]:
        raise ValueError("Requested epochs do not extend this checkpoint.")
    env = environment()
    native_hash = (prior.get("native_initialization_sha256") if args.resume else
                   sha256(args.native_checkpoint) if args.native_checkpoint else None)
    partial = (args.max_train_batches is not None or args.max_val_batches is not None or
               (args.resume is not None and prior["partial_smoke_run"]))
    limits = {"train": args.max_train_batches, "val": args.max_val_batches}
    metadata = {
        "config": cfg, "architecture": model.architecture_config(), "environment": env,
        "manifest_sha256": manifest_hash, "dataset_summary": summary,
        "native_initialization_sha256": native_hash,
        "initialization": cfg["initialization"],
        "device": str(device), "partial_smoke_run": partial,
        "batch_limits": limits,
        "selection": "best validation global foreground macro Dice; test never read by train",
    }
    atomic_json(output / "run.json", metadata)
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        training = run_epoch(model, loader_for(train_rows, cfg, device, True, epoch), device,
                             cfg["diffusion_weight"], optimizer, args.max_train_batches)
        validation = run_epoch(model, loader_for(val_rows, cfg, device), device,
                               cfg["diffusion_weight"], max_batches=args.max_val_batches)
        score = validation["macro_fg"]["dice"]
        if score is None:
            raise ValueError("Foreground validation Dice is undefined; review class coverage.")
        improved = score > best
        best, bad_epochs = (score, 0) if improved else (best, bad_epochs + 1)
        history.append({"epoch": epoch, "train": training, "val": validation, "batch_limits": limits})
        checkpoint = {
            "schema_version": 1, "architecture": model.architecture_config(),
            "model_config": cfg["model"], "train_config": cfg,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "best_val_dice": best, "bad_epochs": bad_epochs,
            "rng_state": rng_state(), "history": history,
            "manifest_sha256": manifest_hash, "environment": env, "partial_smoke_run": partial,
            "native_initialization_sha256": native_hash,
            "patient_id_sha256_by_split": patient_hashes(rows), "batch_limits": limits,
        }
        atomic_checkpoint(output / "last.pt", checkpoint)
        if improved:
            atomic_checkpoint(output / "best.pt", checkpoint)
        atomic_json(output / "history.json", history)
        print(json.dumps({"epoch": epoch, "val_macro_fg_dice": score, "selected": improved}), flush=True)
        if bad_epochs >= cfg["patience"]:
            break
    atomic_json(output / "status.json", {
        "completed": True, "epochs_completed": history[-1]["epoch"],
        "best_val_dice": best, "partial_smoke_run": partial,
        "test_evaluated": False,
    })


def evaluate(args):
    configure_precision()
    device = select_device(args.device)
    rows = load_manifest(args.manifest)
    summary = validate_patient_splits(rows)
    selected = labeled_split(rows, args.split)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if args.external_cohort:
        prior_patients = {p for group in checkpoint["patient_id_sha256_by_split"].values() for p in group}
        current_patients = {p for group in patient_hashes(selected).values() for p in group}
        if prior_patients & current_patients:
            raise ValueError("External cohort contains patient IDs present in the original training manifest.")
    elif sha256(args.manifest) != checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from training. Use the original manifest or explicitly --external-cohort for new patients.")
    cfg = dict(checkpoint["train_config"], batch_size=args.batch_size, num_workers=args.num_workers)
    report = run_epoch(model, loader_for(selected, cfg, device), device, cfg["diffusion_weight"])
    report.update(
        split=args.split, checkpoint_sha256=sha256(args.checkpoint),
        manifest_sha256=sha256(args.manifest), dataset_summary=summary,
        architecture=model.architecture_config(),
        checkpoint_epoch=checkpoint["epoch"],
        checkpoint_partial_smoke_run=checkpoint["partial_smoke_run"],
        inference="deterministic t=0; no sampled noise; no calibration/test tuning",
        decoder_gate_max_abs=float(model.decoder_gate.detach().abs().max()),
        evaluation_cohort="external_new_patient_ids" if args.external_cohort else "original_manifest_split",
        evaluation_environment=environment(),
        source_matches_training=source_hashes() == checkpoint["environment"]["source_sha256"],
    )
    atomic_json(args.output, report)
    print(json.dumps({"split": args.split, "macro_fg": report["macro_fg"]}), flush=True)
