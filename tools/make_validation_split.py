#!/usr/bin/env python3
# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Hold out validation IDs from train only; never change the reference test set."""

import argparse
import json
from pathlib import Path
import numpy as np


def make_split(source, ratio=.05, seed=20260511):
    train, test = source["train"], source["test"]
    if (source.get("val") or not 0 < ratio < 1 or len(train) < 2
            or len(set(train + test)) != len(train) + len(test)):
        raise ValueError("Need disjoint train/test IDs without an existing validation split")
    order = np.random.default_rng(seed).permutation(len(train))
    count = min(len(train) - 1, max(1, round(len(train) * ratio)))
    held_out = set(map(int, order[:count]))
    return {"train": [value for i, value in enumerate(train) if i not in held_out],
            "val": [value for i, value in enumerate(train) if i in held_out], "test": list(test)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratio", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Split output exists; choose a new path")
    result = make_split(json.loads(Path(args.source).read_text()), args.ratio, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: len(values) for key, values in result.items()}))
