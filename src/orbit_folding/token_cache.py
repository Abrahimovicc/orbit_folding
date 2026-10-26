# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Split-preserving LFQ bit/VQ index caches, independent of training data paths."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from .provenance import dataset_fingerprint, file_sha256


CACHE_FORMAT = "orbit_folding.token_cache.v1"
INDEX_CACHE_FORMAT = "orbit_folding.index_cache.v1"
SPLITS = frozenset(("train", "val", "test"))


def tokenizer_hash(tokenizer: torch.nn.Module) -> str:
    """Hash configuration and tensor state, without serializing executable objects."""
    digest = hashlib.sha256()
    configuration = json.dumps(getattr(tokenizer, "specs", {}), sort_keys=True).encode()
    digest.update(len(configuration).to_bytes(8, "big"))
    digest.update(configuration)
    for name, tensor in sorted(tokenizer.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        description = json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode()
        digest.update(len(description).to_bytes(8, "big"))
        digest.update(description)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _identifier(value, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a nonempty relative identifier.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError(f"{label} must not contain an absolute/private source path.")
    return value


def _strings(batch: Mapping, key: str, count: int) -> list[str]:
    values = batch.get(key)
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"A collated batch must have {count} {key} strings.")
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"{key} must contain strings.")
    return list(values)


@torch.no_grad()
def export_tokens(tokenizer: torch.nn.Module, data_loader, output_dir,
                  device) -> dict:
    """Export LFQ uint8 CHW bits or VQ int64 HW indices as per-sample NPZs.

    sample_id, split, and category come from the collated GeometryDataset batch.
    No source data paths or configuration paths are included in the manifest.
    The returned dictionary is also written to output_dir/manifest.json.
    """
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Token export requires an empty output directory.")
    root.mkdir(parents=True, exist_ok=True)
    source_hash = tokenizer_hash(tokenizer)
    provenance = {"producer": "orbit_folding.export_tokens", "tokenizer_hash": source_hash}
    dataset = getattr(data_loader, "dataset", None)
    if hasattr(dataset, "root"):
        provenance["geometry_fingerprint"] = dataset_fingerprint(dataset.root)
        provenance["encoder_sampling"] = (dataset.sampling_settings() if hasattr(dataset, "sampling_settings")
            else {key: getattr(dataset, key) for key in ("seed", "pc_size") if hasattr(dataset, key)})
        provenance["encoder_sampling"]["epoch"] = getattr(dataset, "epoch", 0)
    entries = []
    seen = set()
    bit_shape = None
    quantizer_type = getattr(tokenizer, "quantizer_type", "lfq")
    index_cache = quantizer_type == "standard_vq"
    field = "indices" if index_cache else "bits"
    codebook_size = tokenizer.tokenizer.codebook_size if index_cache else None
    training = tokenizer.training
    tokenizer.to(device)
    try:
        tokenizer.eval()
        for batch in data_loader:
            encode = getattr(tokenizer, "encode_tokens", tokenizer.encode)
            output = encode(batch["points"].to(device), frame=batch["frame"].to(device))
            if not isinstance(output, Mapping) or field not in output:
                raise ValueError(f"tokenizer.encode must return a mapping with {field}.")
            bits = output[field]
            if not isinstance(bits, torch.Tensor) or bits.ndim != (3 if index_cache else 4) or min(bits.shape) < 1:
                raise ValueError("Encoded tokens must be nonempty BHW indices or BCHW bits.")
            if index_cache:
                from .models.quantization import spatial_indices
                bits = spatial_indices(bits, codebook_size)
            elif not torch.all((bits == 0) | (bits == 1)):
                raise ValueError("Encoded bits must be binary before caching.")
            shape = list(bits.shape[1:])
            if bit_shape is not None and shape != bit_shape:
                raise ValueError("A cache must have one consistent CHW bit shape.")
            bit_shape = shape
            count = bits.shape[0]
            ids = _strings(batch, "sample_id", count)
            splits = _strings(batch, "split", count)
            categories = _strings(batch, "category", count)
            arrays = bits.detach().to(device="cpu", dtype=torch.int64 if index_cache else torch.uint8).numpy()
            for index, array in enumerate(arrays):
                sample_id = _identifier(ids[index], "sample_id")
                category = _identifier(categories[index], "category")
                split = splits[index]
                if split not in SPLITS:
                    raise ValueError("Every sample must explicitly belong to train, val, or test.")
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample identity in cache: {sample_id}")
                seen.add(sample_id)
                filename = f"sample_{len(entries):08d}.npz"
                np.savez_compressed(
                    root / filename, **{field: array}, sample_id=np.str_(sample_id),
                    split=np.str_(split), category=np.str_(category),
                    tokenizer_hash=np.str_(source_hash),
                    provenance=np.str_(json.dumps(provenance, sort_keys=True)),
                )
                entries.append({"id": sample_id, "path": filename, "sha256": file_sha256(root / filename),
                                "split": split, "category": category})
    finally:
        tokenizer.train(training)
    if not entries:
        raise ValueError("No token samples were exported.")
    manifest = {
        "format": INDEX_CACHE_FORMAT if index_cache else CACHE_FORMAT,
        "dtype": "int64" if index_cache else "uint8",
        "layout": "HW" if index_cache else "CHW",
        **({"token_shape": bit_shape, "codebook_size": codebook_size} if index_cache
           else {"bit_shape": bit_shape, "num_bits": bit_shape[0]}),
        "quantizer_type": quantizer_type,
        "tokenizer_hash": source_hash,
        "provenance": provenance,
        "num_samples": len(entries),
        "split_counts": dict(Counter(entry["split"] for entry in entries)),
        "items": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class TokenDataset(Dataset):
    """Read only the requested manifest split; never create an implicit holdout.

    Return float32 CHW LFQ bits or int64 HW VQ indices, plus sample_id, split
    and category. No artificial binary representation is assigned to VQ.
    """

    def __init__(self, root, split: str = "train"):
        if split not in SPLITS | {"all"}:
            raise ValueError("split must be train, val, test, or all.")
        self.root = Path(root).resolve()
        self.split = split
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        manifest = self.manifest
        self.index_cache = manifest.get("format") == INDEX_CACHE_FORMAT
        self.quantizer_type = manifest.get("quantizer_type", "lfq")
        if manifest.get("format") not in {CACHE_FORMAT, INDEX_CACHE_FORMAT}:
            raise ValueError("Unsupported token-cache format.")
        if self.quantizer_type != ("standard_vq" if self.index_cache else "lfq"):
            raise ValueError("Cache format and quantizer type disagree")
        if self.index_cache:
            self.token_shape = tuple(manifest["token_shape"])
            self.codebook_size = manifest["codebook_size"]
            if (manifest.get("dtype") != "int64" or manifest.get("layout") != "HW"
                    or len(self.token_shape) != 2 or min(self.token_shape) < 1
                    or type(self.codebook_size) is not int or self.codebook_size < 2):
                raise ValueError("Invalid categorical index cache dimensions/vocabulary")
            self.bit_shape, self.num_bits = None, None
        elif manifest.get("dtype") != "uint8" or manifest.get("layout") != "CHW":
            raise ValueError("Expected uint8 CHW cache bits.")
        else:
            self.bit_shape = tuple(manifest["bit_shape"])
            self.num_bits = int(manifest["num_bits"])
            if (len(self.bit_shape) != 3 or min(self.bit_shape) < 1
                    or self.bit_shape[0] != self.num_bits):
                raise ValueError("Invalid manifest bit shape.")
            self.token_shape, self.codebook_size = self.bit_shape[1:], 1 << self.num_bits
        self.tokenizer_hash = manifest["tokenizer_hash"]
        if (not isinstance(self.tokenizer_hash, str) or len(self.tokenizer_hash) != 64
                or any(c not in "0123456789abcdef" for c in self.tokenizer_hash)):
            raise ValueError("Invalid source tokenizer hash.")
        entries = manifest["items"]
        if len(entries) != manifest["num_samples"]:
            raise ValueError("Manifest sample count does not match its entries.")
        ids, paths = set(), set()
        for entry in entries:
            sample_id = _identifier(entry["id"], "sample_id")
            _identifier(entry["category"], "category")
            self._sample_path(entry["path"])
            if entry["split"] not in SPLITS:
                raise ValueError("Manifest contains an unknown sample split.")
            if sample_id in ids or entry["path"] in paths:
                raise ValueError("Manifest contains duplicate identities or cache paths.")
            ids.add(sample_id)
            paths.add(entry["path"])
        if dict(Counter(entry["split"] for entry in entries)) != manifest["split_counts"]:
            raise ValueError("Manifest split counts do not match its entries.")
        self.items = [dict(entry) for entry in entries
                      if split == "all" or entry["split"] == split]
        if not self.items:
            raise ValueError(f"No cached samples explicitly labeled {split!r}.")

    def _sample_path(self, relative: str) -> Path:
        relative = _identifier(relative, "cache path")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path.suffix != ".npz":
            raise ValueError("Cache paths must be NPZ files inside the cache root.")
        return path

    def __len__(self):
        return len(self.items)

    def set_epoch(self, epoch: int):
        """Match GeometryDataset's engine interface; cached bits have no augmentation."""
        self.epoch = int(epoch)

    def get_item(self, index: int) -> dict:
        entry = self.items[index]
        with np.load(self._sample_path(entry["path"]), allow_pickle=False) as data:
            for key, expected in (("sample_id", entry["id"]),
                                  ("split", entry["split"]),
                                  ("category", entry["category"]),
                                  ("tokenizer_hash", self.tokenizer_hash)):
                if str(data[key].item()) != expected:
                    raise ValueError(f"Cached {key} does not match its manifest.")
            provenance = json.loads(str(data["provenance"].item()))
            if provenance != self.manifest["provenance"]:
                raise ValueError("Cached provenance does not match its manifest.")
            field = "indices" if self.index_cache else "bits"
            bits = data[field]
            if self.index_cache:
                if (bits.dtype != np.int64 or bits.shape != self.token_shape
                        or not np.all((bits >= 0) & (bits < self.codebook_size))):
                    raise ValueError("Cached indices must be int64 HW within the manifest vocabulary")
                tensor = torch.from_numpy(bits.copy())
            else:
                if (bits.dtype != np.uint8 or bits.shape != self.bit_shape
                        or not np.all((bits == 0) | (bits == 1))):
                    raise ValueError("Cached bits must be binary uint8 with the manifest CHW shape.")
                tensor = torch.from_numpy(bits.astype(np.float32))
        return {field: tensor, "sample_id": entry["id"],
                "split": entry["split"], "category": entry["category"]}

    def __getitem__(self, index):
        return self.get_item(index)
