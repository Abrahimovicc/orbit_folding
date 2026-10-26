#!/usr/bin/env python3
"""Geometric Orbit Representatives for Unaligned 3D Tokenization.

Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig.
Contact: Xiang Chen <xiang.chen@uni-leipzig.de>. SPDX-License-Identifier: MIT.

Produce a clean GitHub-upload archive, excluding generated data and checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"src", "tests", "tools", "configs", "docs", "data", "assets",
               "applications", "licenses", ".github"}
ROOT_FILES = {"README.md", "LICENSE", "CITATION.cff", "THIRD_PARTY_NOTICES.md",
              "pyproject.toml", "MANIFEST.in", ".gitignore", ".gitattributes"}


def release_files(root=ROOT):
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"Symlink not allowed in an upload archive: {relative}")
        if not path.is_file():
            continue
        if relative.parts[0] not in DIRECTORIES and str(relative) not in ROOT_FILES:
            continue
        if any(part in {"__pycache__", ".pytest_cache", "processed"} or part.endswith(".egg-info")
               for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo", ".pt", ".pth", ".ckpt"}:
            continue
        if path.name.startswith(".env") or path.stat().st_size > 10_000_000:
            raise ValueError(f"Unexpected private/oversized artifact: {relative}")
        if path.suffix in {".py", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8")
            if "Chair of Multimodal Machine Learning" not in text[:700] or "xiang.chen@uni-leipzig.de" not in text[:700]:
                raise ValueError(f"Project header missing: {relative}")
            private_prefixes = tuple("/" + part + "/" for part in ("home", "mnt"))
            if any(prefix in text for prefix in private_prefixes):
                raise ValueError(f"Machine-specific path leaked: {relative}")
        files.append(path)
    if len(list((root / "data/samples/raw").glob("*.ply"))) != 4:
        raise ValueError("The public sample bundle must contain exactly four attributed meshes")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "orbit_folding_release.zip")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = release_files()
    if args.check:
        print(json.dumps({"files": len(files), "bytes": sum(p.stat().st_size for p in files)}, indent=2))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive checksum verification failed")
    print(json.dumps({"archive": str(args.output), "files": len(files),
                      "bytes": args.output.stat().st_size,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
