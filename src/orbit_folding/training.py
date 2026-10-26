"""Geometric Orbit Representatives for Unaligned 3D Tokenization.

Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig.
Contact: Xiang Chen <xiang.chen@uni-leipzig.de>. SPDX-License-Identifier: MIT.

One torchrun-compatible engine for reconstruction and masked completion.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import yaml

from .checkpoints import load_checkpoint, save_checkpoint
from .data import geometry_dataset, write_json
from .token_cache import TokenDataset
from .monitoring import format_progress, plot_training
from .provenance import dataset_fingerprint, file_sha256
from .validation import validate


def build_model(config):
    if config.get("Task") == "tokenizer":
        from .models.tokenizer import ReconstructiveTokenizer
        return ReconstructiveTokenizer(config)
    if config.get("Task") == "prior":
        from .models.prior import build_completion_prior
        return build_completion_prior(config)
    raise ValueError("Configuration Task must be tokenizer or prior")


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "numpy" in state:
        name, keys, position, gaussian, cached = state["numpy"]
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _rank_zero_call(function, rank, world):
    payload = [None]
    if rank == 0:
        try:
            payload[0] = {"value": function()}
        except Exception as error:
            payload[0] = {"error": str(error), "type": type(error).__name__}
    if world > 1:
        dist.broadcast_object_list(payload, src=0)
    if "error" in payload[0]:
        kind = FileExistsError if payload[0]["type"] == "FileExistsError" else ValueError
        raise kind(payload[0]["error"])
    return payload[0]["value"]


def _check_output(output, resume, step):
    output = Path(output)
    if not output.exists() or not any(output.iterdir()):
        return
    if resume is None:
        raise FileExistsError("Training output is not empty; use a new directory or explicit --resume")
    if Path(resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("Branching from another or older checkpoint requires a fresh output directory")
    log = output / "metrics.jsonl"
    steps = [json.loads(line)["step"] for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    if (steps and (steps[-1] != step or any(b <= a for a, b in zip(steps, steps[1:])))) or (not steps and step):
        raise ValueError("Run logs do not end at the resume checkpoint; use a fresh output directory to preserve them")


def _optimizer_update(model, optimizer, scaler, scheduler, clip):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip,
                                         error_if_nonfinite=not scaler.is_enabled())
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = not scaler.is_enabled() or scaler.get_scale() >= previous_scale
    optimizer.zero_grad(set_to_none=True)
    if updated and scheduler is not None:
        scheduler.step()
    return updated, float(norm)


def _training_device(device, local_rank, world):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for the demo")
        index = local_rank if world > 1 else device.index
        device = torch.device("cuda", torch.cuda.current_device() if index is None else index)
        torch.cuda.set_device(device)
    return device


def train(config, data_root, output, *, device="auto", epochs=None, batch_size=None,
          max_steps=None, workers=0, resume=None, init_checkpoint=None, seed=None, threads=4):
    if resume is not None and init_checkpoint is not None:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    config = copy.deepcopy(config)
    seed = config.get("seed", 2026) if seed is None else seed
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = _training_device(device, local_rank, world)
    torch.set_num_threads(max(int(threads), 1))
    if world > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    try:
        return _train(config, data_root, output, device, epochs, batch_size,
                      max_steps, workers, resume, init_checkpoint, seed, rank, world)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _train(config, data_root, output, device, epochs, batch_size, max_steps,
           workers, resume, init_checkpoint, seed, rank, world):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    task = config.get("Task")
    dataset = geometry_dataset(data_root, config, seed=seed) if task == "tokenizer" else TokenDataset(data_root, "train")
    data_hash = hashlib.sha256((Path(data_root) / "manifest.json").read_bytes()).hexdigest()
    fingerprint = _rank_zero_call(lambda: dataset_fingerprint(data_root), rank, world)
    categorical = task == "prior" and config.get("Prior", {}).get("type", "bit_field") == "categorical"
    if task == "prior":
        if categorical:
            expected = config.get("vocab_size", config.get("Tokenizer", {}).get("codebook_size", 4096))
            if dataset.codebook_size != expected:
                raise ValueError("Cache vocabulary differs from the categorical prior")
        elif dataset.index_cache or dataset.num_bits != config.get("num_bits", 12):
            raise ValueError("Bit-field prior requires matching LFQ bits, not VQ indices")
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed) if world > 1 else None
    batch_size = int(config.get("batch_size", 2) if batch_size is None else batch_size)
    epochs = int(config.get("num_epochs", config.get("epochs", 1)) if epochs is None else epochs)
    settings = config.get("Trainer", {})
    schedule_interval = settings.get("lr_scheduler_interval", "step")
    if schedule_interval not in {"step", "epoch"}:
        raise ValueError("Trainer.lr_scheduler_interval must be step or epoch")
    accumulation = int(settings.get("accumulate_grad_batches", 1))
    if epochs <= 0 or batch_size <= 0 or accumulation <= 0 or workers < 0:
        raise ValueError("Positive epochs, batch size and accumulation, and nonnegative workers are required")
    if max_steps is None:
        max_steps = settings.get("max_steps")
    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    validation_config = config.get("Validation", {})
    validation_data = None
    validation_period = int(validation_config.get("every_epochs", 1))
    if validation_period < 1:
        raise ValueError("Validation.every_epochs must be positive")
    if validation_config.get("enabled", True):
        validation_data = (geometry_dataset(data_root, config, "val", seed=seed) if task == "tokenizer"
                           else TokenDataset(data_root, "val"))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None,
                        sampler=sampler, num_workers=workers, pin_memory=device.type == "cuda",
                        persistent_workers=False, generator=generator, drop_last=False)
    model = build_model(config).to(device)
    optimizer_config = config.get("Optimizer", config.get("optimizer", {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optimizer_config.get("lr", 1e-4)),
                                  weight_decay=float(optimizer_config.get("weight_decay", 0.01)))
    total_steps = max(epochs * math.ceil(len(loader) / accumulation), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs if schedule_interval == "epoch" else total_steps,
        eta_min=float(optimizer_config.get("eta_min", 1e-6)))
    precision = settings.get("precision", "32-true")
    mixed = device.type == "cuda" and precision in {"16-mixed", "bf16-mixed"}
    dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    initial_scale = float(settings.get("amp_init_scale", 65536))
    if not math.isfinite(initial_scale) or initial_scale <= 0:
        raise ValueError("Trainer.amp_init_scale must be finite and positive")
    scaler = torch.amp.GradScaler("cuda", enabled=mixed and dtype == torch.float16, init_scale=initial_scale)
    epoch_start, skip_batches, step, skipped_updates = 0, 0, 0, 0
    epoch_initial_step = 0
    best_score, best_step = None, None
    initialization = None
    if init_checkpoint is not None:
        initial = load_checkpoint(init_checkpoint)
        if initial["config"]["Task"] != task:
            raise ValueError("Initialization checkpoint task does not match the training configuration")
        if task == "prior" and initial.get("extra", {}).get("tokenizer_hash") != dataset.tokenizer_hash:
            raise ValueError("Prior initialization requires a verified matching tokenizer binding")
        if any(not torch.isfinite(value).all() for value in initial["state_dict"].values() if value.is_floating_point()):
            raise ValueError("Initialization checkpoint contains nonfinite weights")
        model.load_state_dict(initial["state_dict"], strict=True)
        initialization = {"source_sha256": file_sha256(init_checkpoint), "source_step": initial["step"],
                          "changed_config_keys": sorted(key for key in config.keys() | initial["config"].keys()
                                                        if config.get(key) != initial["config"].get(key))}
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint(resume)
        if checkpoint["config"] != config:
            raise ValueError("Resume requires exactly the saved model configuration; use --init-checkpoint for a new run from weights")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if checkpoint["optimizer_state_dict"] is None:
            raise ValueError("Checkpoint has no optimizer state; it is not a resumable training checkpoint")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        state = checkpoint["extra"]["engine"]
        if state["world_size"] != world or state["batch_size"] != batch_size or state["accumulation"] != accumulation:
            raise ValueError("Exact resume requires unchanged world size, per-rank batch size, and accumulation")
        if state["planned_epochs"] != epochs or state["seed"] != seed or state["data_manifest_sha256"] != data_hash:
            raise ValueError("Exact resume requires the same planned epochs, seed, and dataset manifest")
        if state.get("data_fingerprint") != fingerprint:
            raise ValueError("Data content fingerprint differs or is missing; exact resume is not possible")
        if task == "prior" and checkpoint["extra"].get("tokenizer_hash") != dataset.tokenizer_hash:
            raise ValueError("Resume prior and cache use different tokenizer weights")
        epoch_start, skip_batches, step = state["epoch"], state["next_batch"], checkpoint["step"]
        best_score, best_step = state.get("best_score"), state.get("best_step")
        skipped_updates = state.get("skipped_updates", 0)
        epoch_initial_step = state.get("epoch_initial_step", step)
        initialization = checkpoint["extra"].get("initialization")
        scaler.load_state_dict(state["scaler"])
    _rank_zero_call(lambda: _check_output(output, resume, step), rank, world)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                       broadcast_buffers=False, find_unused_parameters=False)
    torch.manual_seed(seed + rank)
    if checkpoint:
        _restore_rng(checkpoint["extra"]["engine"]["rng_by_rank"][rank])
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        write_json(output / "run.json", {"task": task, "seed": seed, "world_size": world,
                   "prior_type": "categorical" if categorical else ("bit_field" if task == "prior" else None),
                   "batch_per_rank": batch_size, "accumulation": accumulation,
                   "effective_batch_upper_bound": batch_size * world * accumulation,
                   "device_type": device.type, "torch": str(torch.__version__),
                   "data_fingerprint": fingerprint,
                   "initialization": initialization,
                   "sampling": dataset.sampling_settings() if task == "tokenizer" else None,
                   "validation_samples": len(validation_data) if validation_data is not None else 0,
                   "training_samples": len(dataset), "precision": precision,
                   "lr_scheduler_interval": schedule_interval})
    if world > 1:
        dist.barrier()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    next_epoch, next_batch = epoch_start, skip_batches
    starting_step, starting_skips = step, skipped_updates

    def checkpoint_training(save_best=False):
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, _rng_state())
        else:
            states[0] = _rng_state()
        if rank == 0:
            extra = {"engine": {"epoch": next_epoch, "next_batch": next_batch,
                     "world_size": world, "batch_size": batch_size, "accumulation": accumulation,
                     "planned_epochs": epochs, "seed": seed, "data_manifest_sha256": data_hash,
                     "data_fingerprint": fingerprint, "best_score": best_score, "best_step": best_step,
                     "skipped_updates": skipped_updates,
                     "epoch_initial_step": epoch_initial_step,
                     "rng_by_rank": states, "scaler": scaler.state_dict()}}
            if task == "prior":
                extra["tokenizer_hash"] = dataset.tokenizer_hash
            if initialization is not None:
                extra["initialization"] = initialization
            save_checkpoint(output / "last.pt", model, config, optimizer, scheduler,
                            epoch=next_epoch, step=step, extra=extra)
            if save_best:
                save_checkpoint(output / "best.pt", model, config, optimizer, scheduler,
                                epoch=next_epoch, step=step, extra=extra)
        if world > 1:
            dist.barrier()

    saved = False
    for epoch in range(epoch_start, epochs):
        if max_steps is not None and step >= max_steps:
            break
        if sampler:
            sampler.set_epoch(epoch)
        if epoch != epoch_start or skip_batches == 0:
            epoch_initial_step = step
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        generator.manual_seed(seed + epoch)
        for index, batch in enumerate(loader):
            if epoch == epoch_start and index < skip_batches:
                continue
            batch = move_batch(batch, device)
            update = (index + 1) % accumulation == 0 or index + 1 == len(loader)
            group_size = min(accumulation, len(loader) - (index // accumulation) * accumulation)
            sync = model.no_sync() if world > 1 and not update else contextlib.nullcontext()
            with sync:
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=mixed):
                    loss, metrics = model(batch, step=step)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at update {step}")
                scaler.scale(loss / group_size).backward()
            if not update:
                continue
            updated, gradient_norm = _optimizer_update(model, optimizer, scaler,
                                                       scheduler if schedule_interval == "step" else None,
                                                       float(settings.get("gradient_clip_val", 1.0)))
            next_epoch, next_batch = (epoch + 1, 0) if index + 1 == len(loader) else (epoch, index + 1)
            if not updated:
                skipped_updates += 1
                if rank == 0:
                    with (output / "skipped_updates.jsonl").open("a") as stream:
                        stream.write(json.dumps({"epoch": epoch, "batch": index, "step": step,
                                                 "scale": scaler.get_scale()}) + "\n")
                    print(f"{task}: skipped non-finite gradient update; optimizer step remains {step}", flush=True)
                continue
            step += 1
            values = {key: value.detach().to(device=device, dtype=torch.float64).mean()
                      for key, value in metrics.items() if isinstance(value, torch.Tensor)}
            values["objective"] = loss.detach().double()
            names = sorted(values)
            scalars = torch.stack([values[name] for name in names])
            if world > 1:
                dist.all_reduce(scalars)
                additive = {"loss_sum", "loss_count", "bit_correct", "bit_count", "token_correct", "token_count"}
                for scalar_index, name in enumerate(names):
                    if name not in additive:
                        scalars[scalar_index] /= world
            report = {name: float(value) for name, value in zip(names, scalars.cpu())}
            if "bit_count" in report:
                report["bit_accuracy"] = report["bit_correct"] / max(report["bit_count"], 1)
                report["token_exact"] = report["token_correct"] / max(report["token_count"], 1)
                report["loss"] = report["loss_sum"] / max(report["bit_count"], 1)
                report["bit_acc"] = report["bit_accuracy"]
            elif "token_count" in report:
                report["token_exact"] = report["token_correct"] / max(report["token_count"], 1)
                report["loss"] = report["loss_sum"] / max(report["loss_count"], 1)
            report.update(epoch=epoch, step=step, lr=optimizer.param_groups[0]["lr"])
            if rank == 0:
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(report) + "\n")
                print(format_progress(task, report), flush=True)
            if max_steps is not None and step >= max_steps:
                break
        improved = False
        if schedule_interval == "epoch" and next_epoch > epoch and step > epoch_initial_step:
            scheduler.step()
        if validation_data is not None and next_epoch > epoch and ((epoch + 1) % validation_period == 0 or next_epoch == epochs):
            rng_state = _rng_state()
            try:
                validation = validate(model, validation_data, task=task, device=device, rank=rank,
                                      world=world, batch_size=batch_size, seed=seed,
                                      mask_ratios=validation_config.get("mask_ratios"),
                                      selection_metric=validation_config.get("selection_metric"))
            finally:
                _restore_rng(rng_state)
            validation.update(epoch=epoch, step=step)
            if best_score is None or validation["selection_score"] < best_score:
                best_score, best_step, improved = validation["selection_score"], step, True
            if rank == 0:
                with (output / "validation.jsonl").open("a") as stream:
                    stream.write(json.dumps(validation, allow_nan=False) + "\n")
                print(f"Validation: {validation['selection_metric']}={validation['selection_score']:.6f}", flush=True)
        checkpoint_training(save_best=improved)
        saved = True
    if not saved:
        checkpoint_training()
    if step == starting_step and skipped_updates > starting_skips:
        raise FloatingPointError("No optimizer updates completed: every attempted update overflowed. "
                                 "Use bf16-mixed/32-true or a smaller Trainer.amp_init_scale. "
                                 "The checkpoint and skipped-update log are retained for inspection.")
    if rank == 0 and settings.get("plot_training", True) and (output / "metrics.jsonl").exists():
        print(f"Training curves: {plot_training(output)}", flush=True)
    return {"step": step, "epoch": next_epoch, "checkpoint": str(output / "last.pt")}
