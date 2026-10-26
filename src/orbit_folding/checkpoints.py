# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Restricted, explicit-config checkpoints and audited original-weight conversion."""

from __future__ import annotations

import copy
import hashlib
import os
import pickle
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


CHECKPOINT_FORMAT = "orbit_folding.checkpoint.v1"
TASKS = frozenset(("tokenizer", "prior"))


def _configuration(config: dict, kind: str | None = None) -> dict:
    if not isinstance(config, dict):
        raise TypeError("config must be an explicit dictionary, never a path to another file.")
    config = copy.deepcopy(config)
    task = config.get("Task", kind)
    if task not in TASKS or (kind is not None and task != kind):
        raise ValueError("Task must be 'tokenizer' or 'prior' and match the requested kind.")
    config["Task"] = task
    _safe_value(config, "config")
    return config


def _safe_value(value, name: str):
    if value is None or type(value) in (str, bool, int, float, bytes):
        return
    if isinstance(value, torch.Tensor):
        return
    if type(value) in (dict, list, tuple):
        if isinstance(value, dict):
            for key, item in value.items():
                if type(key) not in (str, int, float, bool):
                    raise TypeError(f"Unsupported dictionary key in {name}.")
                _safe_value(item, name)
        else:
            for item in value:
                _safe_value(item, name)
        return
    raise TypeError(f"{name} contains an unsafe/unsupported object: {type(value).__name__}")


def _new_model(config: dict) -> nn.Module:
    # Only these constructors are allowed; config cannot select an import path.
    if config["Task"] == "prior":
        from .models.prior import build_completion_prior
        return build_completion_prior(config)
    from .models.tokenizer import ReconstructiveTokenizer
    return ReconstructiveTokenizer(config)


def _validate_state(state, target: Mapping | None = None):
    if not isinstance(state, Mapping) or not state:
        raise ValueError("state_dict must be a nonempty mapping of named tensors.")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in state.items()):
        raise ValueError("state_dict may contain only string keys and tensors.")
    if target is not None:
        missing = sorted(set(target) - set(state))
        unexpected = sorted(set(state) - set(target))
        wrong_shape = [key for key in target.keys() & state.keys()
                       if target[key].shape != state[key].shape]
        if missing or unexpected or wrong_shape:
            raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}, "
                             f"wrong_shape={sorted(wrong_shape)}")


def save_checkpoint(path, model, config, optimizer=None, scheduler=None,
                    epoch=0, step=0, extra=None):
    """Save portable tensors and explicit configuration, never a pickled module.

    Engine RNG/sampler/accumulation state can be supplied as safe values in extra.
    Optimizer and scheduler states are named *_state_dict in the returned payload.
    """
    config = _configuration(config)
    model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    _validate_state(state)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_type": config["Task"],
        "config": config,
        "state_dict": state,
        "epoch": int(epoch),
        "step": int(step),
        "extra": {} if extra is None else extra,
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
    }
    _safe_value(payload, "checkpoint")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload


def load_checkpoint(path, device="cpu") -> dict:
    """Load release checkpoints only, always using torch's restricted unpickler."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Not a release checkpoint; convert original checkpoints explicitly.")
    checkpoint["config"] = _configuration(checkpoint.get("config"))
    if checkpoint.get("model_type") != checkpoint["config"]["Task"]:
        raise ValueError("Checkpoint model_type does not match config.Task.")
    _validate_state(checkpoint.get("state_dict"))
    _safe_value(checkpoint, "checkpoint")
    return checkpoint


def load_model(path, device="cpu") -> tuple[nn.Module, dict]:
    """Construct an allowlisted release model from the embedded config, strictly."""
    checkpoint = load_checkpoint(path, device="cpu")
    config = checkpoint["config"]
    model = _new_model(config)
    _validate_state(checkpoint["state_dict"], model.state_dict())
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.checkpoint_metadata = copy.deepcopy(checkpoint.get("extra", {}))
    return model.to(device).eval(), config


def verify_prior_binding(prior, expected_hash, *, allow_unverified=False):
    """Reject a known mismatch even when legacy, missing provenance is allowed."""
    declared = getattr(prior, "checkpoint_metadata", {}).get("tokenizer_hash")
    if declared is None:
        if allow_unverified:
            return False
        raise ValueError("Prior checkpoint has no tokenizer binding; migrate a verified pair or explicitly allow an unverified legacy prior")
    if declared != expected_hash:
        raise ValueError("Prior and tokenizer/cache use different tokenizer weights")
    return True


def _original_checkpoint(source, trusted: bool):
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as error:
        if not trusted:
            raise ValueError("Restricted original-checkpoint loading failed. Only retry "
                             "unsafe pickle with trusted=True after independently "
                             "verifying its source; corruption may also cause this error.") from error
        return torch.load(source, map_location="cpu", weights_only=False)


def convert_original_checkpoint(source, config, output, kind, *, trusted=False):
    """Convert original weights using caller-supplied config, with strict matching.

    Only unused pose_predictor keys (nonlearned canonicalization) and unused
    token_field_head keys (ShareTokenFieldHead=True) may be dropped. Source paths
    are not recorded; provenance contains a SHA-256 hash instead.
    """
    if kind not in TASKS:
        raise ValueError("kind must be 'tokenizer' or 'prior'.")
    if type(trusted) is not bool:
        raise TypeError("trusted must be an explicit boolean.")
    config = _configuration(config, kind=kind)
    for key in ("DataSource", "TrainSplit", "TestSplit", "CanonicalRotationCache",
                "token_path", "label_path", "log_dir", "config"):
        config.pop(key, None)
    source = Path(source)
    if source.resolve() == Path(output).resolve():
        raise ValueError("Conversion must not overwrite the original checkpoint.")
    checkpoint = _original_checkpoint(source, trusted=trusted)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Original checkpoint must be a tensor mapping or Lightning checkpoint.")
    state = checkpoint.get("state_dict", checkpoint)
    _validate_state(state)
    model = _new_model(config)
    target = model.state_dict()
    dropped = []
    retained = {}
    canonicalizer = str(config.get("Canonicalizer", "pca_representative")).lower()
    unused_pose = canonicalizer in {"pca", "pca_representative", "none", "identity",
                                   "ambient", "no_folding"}
    shared_head = config.get("ShareTokenFieldHead", False) is True
    for key, value in state.items():
        if key in target:
            retained[key] = value
        elif kind == "tokenizer" and (
            (key.startswith("pose_predictor.") and unused_pose)
            or (key.startswith("token_field_head.") and shared_head)
        ):
            dropped.append(key)
        else:
            raise ValueError(f"Refusing to drop an unrecognized/used original key: {key}")
    _validate_state(retained, target)
    if kind == "prior":
        hparams = checkpoint.get("hyper_parameters", {})
        categorical = model.prior_type == "categorical"
        expected = dict(config.get("model", {}), **({"vocab_size": model.codebook_size,
            "max_resolution": model.model.max_resolution, "num_classes": 0} if categorical else {"num_bits": model.num_bits}))
        defaults = {**({"vocab_size": 4096, "max_resolution": 128, "num_classes": 0} if categorical else {"num_bits": 12}),
                    "hidden_channels": 256 if categorical else 128, "num_blocks": 12 if categorical else 8,
                    "dilations": (1, 2, 4, 8, 16, 32), "dropout": 0.1}
        for key, default in defaults.items():
            if key not in hparams:
                continue
            actual, desired = hparams[key], expected.get(key, default)
            if key == "dilations":
                actual, desired = tuple(actual), tuple(desired)
            if actual != desired:
                raise ValueError(f"Original prior hyperparameter mismatch for {key}.")
    model.load_state_dict(retained, strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    extra = {"conversion": {"source_sha256": digest.hexdigest(),
                            "dropped_keys": sorted(dropped),
                            "source_format": "lightning" if "state_dict" in checkpoint
                            else "state_dict"}}
    return save_checkpoint(output, model, config,
                           epoch=checkpoint.get("epoch", 0),
                           step=checkpoint.get("global_step", 0), extra=extra)
