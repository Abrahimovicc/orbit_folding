# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""CPU orientation-repeatability example using real mesh/point observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    release = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(release / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=release / "data")
    parser.add_argument("--output", type=Path, default=release / "artifacts" / "inspection")
    parser.add_argument("--split", default="all")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--jitter", type=float, default=0.0, help="Gaussian sigma in radius-0.5 model units")
    noise.add_argument("--jitter-unit-radius", type=float, help="Unit-radius sigma; divided by two for model space")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--rotation-pairs", type=int, default=8)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--points", type=int, default=768)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args(argv)
    from orbit_folding.evaluation import eval_stability
    from orbit_folding.visualization import render_alignment

    jitter = args.jitter_unit_radius / 2 if args.jitter_unit_radius is not None else args.jitter
    result = {}
    if not args.render_only:
        report = eval_stability(args.data_root, split=args.split, output=args.output, device="cpu",
                                seed=args.seed, jitter=jitter, max_samples=args.max_samples,
                                rotation_pairs=args.rotation_pairs)
        result["stability"] = {key: report[key] for key in ("output", "successful_samples", "failed_samples",
                                                            "pca_only_cd_x1e4", "locked_cd_x1e4")}
    result["alignment"] = render_alignment(args.data_root, args.output, sample_id=args.sample_id,
                                             frames=args.frames, points=args.points, split=args.split,
                                             seed=args.seed, jitter=jitter)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
