#!/usr/bin/env python3
# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Compare independently measured token/state results at the paper's precision."""

import argparse
import json
import math
from pathlib import Path


def check_report(report, reference):
    if reference.get("format") != "orbit_folding.paper_reference.v1":
        raise ValueError("Unsupported paper-reference format")
    checks = []
    for key in ("protocol", "samples", "seed", "batch_size", "mask_device"):
        checks.append({"metric": key, "expected": reference[key], "actual": report.get(key),
                       "matches": report.get(key) == reference[key]})
    observed = {row["mask_ratio"]: row for row in report["rows"]}
    aliases = {"masked_cells": "token_count", "bit_accuracy_percent": "bit_accuracy",
               "token_exact_percent": "token_exact", "uncertain_percent": "uncertain_fraction",
               "hard_wrong_percent": "hard_wrong_fraction"}
    for target in reference["rows"]:
        ratio = target["mask_ratio"]
        row = observed.get(ratio, {})
        for key, expected in target.items():
            if key == "mask_ratio":
                continue
            actual = row.get(aliases.get(key, key.removesuffix("_percent")))
            if isinstance(expected, str):
                digits = len(expected.partition(".")[2])
                actual = (format(actual * (100 if key.endswith("_percent") else 1), f".{digits}f")
                          if actual is not None and math.isfinite(actual) else None)
            checks.append({"mask_ratio": ratio, "metric": key, "expected": expected, "actual": actual,
                           "matches": actual == expected})
    return {"matches_reported_precision": all(row["matches"] for row in checks), "checks": checks,
            "scope": "Checkpoint evaluation only. Matching reported numbers does not verify from-scratch training, data independence, or the other paper tables."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--reference", default="data/paper_reference.json")
    parser.add_argument("--output", default="outputs/paper_check.json")
    args = parser.parse_args()
    report = check_report(json.loads(Path(args.report).read_text()), json.loads(Path(args.reference).read_text()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"matches_reported_precision": report["matches_reported_precision"],
                      "checks": len(report["checks"]), "output": str(output)}, indent=2))
    if not report["matches_reported_precision"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
