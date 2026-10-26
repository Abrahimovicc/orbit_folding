#!/usr/bin/env python3
"""Geometric Orbit Representatives for Unaligned 3D Tokenization.

Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig.
Contact: Xiang Chen <xiang.chen@uni-leipzig.de>. SPDX-License-Identifier: MIT.
"""
import argparse
import json

from orbit_folding.weights import prepare_hf_release


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a verified local Hugging Face model folder; never upload")
    parser.add_argument("--sources", required=True, help="Private JSON source-provenance specification")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-original", action="store_true", help="Allow unsafe pickle only for independently trusted original checkpoints")
    args = parser.parse_args()
    result = prepare_hf_release(args.sources, args.output, trusted=args.trust_original)
    print(json.dumps({"output": args.output, "pairs": [pair["id"] for pair in result["pairs"]], "uploaded": False}, indent=2))
