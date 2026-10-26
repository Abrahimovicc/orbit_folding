# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Export an identity- and tokenizer-versioned token cache for real assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    release = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(release / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Release-format tokenizer checkpoint")
    parser.add_argument("--data-root", required=True, type=Path, help="Validated, processed geometry dataset")
    parser.add_argument("--output", required=True, type=Path, help="New or empty cache directory")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--pc-size", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    from torch.utils.data import DataLoader
    from orbit_folding.checkpoints import load_model
    from orbit_folding.data import geometry_dataset
    from orbit_folding.token_cache import export_tokens

    model, config = load_model(args.checkpoint, device="cpu")
    if config["Task"] != "tokenizer":
        parser.error("--checkpoint must contain a tokenizer, not a prior")
    pc_size = args.pc_size if args.pc_size is not None else int(config.get("PCsize", 8192))
    if pc_size < 3:
        parser.error("--pc-size must be at least three")
    dataset = geometry_dataset(args.data_root, config, split=args.split, pc_size=pc_size,
                               seed=args.seed, points_only=True)
    manifest = export_tokens(model, DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0),
                              args.output, device="cpu")
    result = {key: manifest[key] for key in ("num_samples", "split_counts", "bit_shape", "token_shape",
                                           "quantizer_type", "tokenizer_hash") if key in manifest}
    result.update(manifest=str(args.output / "manifest.json"),
                  cache_bytes=sum(path.stat().st_size for path in args.output.iterdir() if path.is_file()),
                  caveat="An asset cache, not semantic deduplication. Rotation repeatability does not guarantee identical tokens. Refresh after geometry, sampling, frame-policy or tokenizer changes; preserve asset licenses.")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
