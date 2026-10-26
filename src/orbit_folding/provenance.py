# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Content fingerprints for geometry, token caches and released weights."""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(root):
    """Hash the manifest AND every referenced file; check declared file hashes."""
    root = Path(root).resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = manifest.get("samples", manifest.get("items"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("A fingerprint requires a nonempty geometry or token manifest")
    digest = hashlib.sha256(manifest_bytes)
    seen = set()
    for item in sorted(entries, key=lambda entry: (entry["id"], entry["path"])):
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or path in seen:
            raise ValueError("Manifest paths must be unique files inside the dataset root")
        seen.add(path)
        checksum = file_sha256(path)
        if item.get("sha256") and checksum != item["sha256"]:
            raise ValueError(f"Data checksum mismatch for {item['id']}; regenerate the manifest and caches")
        description = json.dumps([item["id"], item["path"], checksum], separators=(",", ":")).encode()
        digest.update(len(description).to_bytes(8, "big"))
        digest.update(description)
    return digest.hexdigest()
