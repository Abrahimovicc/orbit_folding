#!/usr/bin/env python3
# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Repeated mesh sampling and paired edge diagnostics on a fixed decoded cohort."""

import argparse
import json
from pathlib import Path

import numpy as np

from orbit_folding.mesh_benchmark import evaluate_mesh_directory
from orbit_folding.provenance import file_sha256


def summarize_repeats(reports, *, bootstrap=4096, bootstrap_seed=20261025):
    if not reports or bootstrap < 1:
        raise ValueError("Supply reports and a positive bootstrap budget")
    first = reports[0]
    if "relaxed" not in first["summary"]:
        raise ValueError("Boundary comparison requires a relaxed decode path")
    if len({r["seed"] for r in reports}) != len(reports):
        raise ValueError("Sampling seeds must be distinct")
    identities = {(row["sample_id"], row["mode"]) for row in first["rows"]}
    expected = first["samples"] * len(first["summary"])
    if len(identities) != expected:
        raise ValueError("Reports must contain every unique object/mode pair")
    for report in reports:
        if (report["mesh_hashes"] != first["mesh_hashes"] or report["samples"] != first["samples"]
                or set(report["summary"]) != set(first["summary"])
                or {(row["sample_id"], row["mode"]) for row in report["rows"]} != identities
                or len(report["rows"]) != expected
                or any(report[key] != first[key] for key in ("sample_points", "edge_percentile", "drift_threshold"))):
            raise ValueError("Repeated sampling requires unchanged meshes, cohort and metric settings")
    result = {"format":"orbit_folding.boundary_repetition.v1", "samples":first["samples"],
        "sample_points":first["sample_points"], "seeds":[r["seed"] for r in reports],
        "edge_percentile":first["edge_percentile"], "drift_threshold":first["drift_threshold"],
        "per_seed":[{"seed":r["seed"], "summary":r["summary"]} for r in reports],
        "paired_edge":{}, "bootstrap_samples":bootstrap, "bootstrap_seed":bootstrap_seed,
        "scope":"Fixed meshes and masks; repeats vary only Monte Carlo surface sampling, not model training seeds.",
        "definitions":{
            "mean_delta":"Comparator edge minus relaxed edge, after averaging repeats per object; positive favors relaxed.",
            "bootstrap_ci95":"Percentile paired bootstrap over objects, NOT treating sampling repeats as independent shapes.",
            "valid_pairs":"Both paths valid with finite edge distance in every repeat; exclusions explicitly counted.",
            "failure_inclusive":"Per-seed summaries retain all requested objects, including missing/invalid meshes."}}
    indexed = [{(row["sample_id"],row["mode"]):row for row in report["rows"]} for report in reports]
    objects = sorted({identity for identity,mode in identities})
    rng = np.random.default_rng(bootstrap_seed)
    for mode in first["summary"]:
        if mode == "relaxed":
            continue
        differences = []
        for identity in objects:
            pairs = [(rows[identity,"relaxed"],rows[identity,mode]) for rows in indexed]
            if not all(a["valid"] and b["valid"] and a["edge_distance"] is not None
                       and b["edge_distance"] is not None and np.isfinite([a["edge_distance"],b["edge_distance"]]).all()
                       for a,b in pairs):
                continue
            differences.append(np.mean([b["edge_distance"]-a["edge_distance"] for a,b in pairs]))
        differences = np.asarray(differences)
        boot = []
        if len(differences):
            for start in range(0, bootstrap, 256):
                indices = rng.integers(0,len(differences),size=(min(256,bootstrap-start),len(differences)))
                boot.append(differences[indices].mean(1))
        result["paired_edge"][mode] = {
            "valid_pairs":len(differences), "excluded_pairs":first["samples"]-len(differences),
            "mean_delta":float(differences.mean()) if len(differences) else None,
            "bootstrap_ci95":np.quantile(np.concatenate(boot),[.025,.975]).tolist() if boot else None,
            "relaxed_win_fraction":float((differences>0).mean()) if len(differences) else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-dir",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--seeds",nargs="+",type=int,default=list(range(20260518,20260523)))
    parser.add_argument("--sample-points",type=int,default=1024)
    parser.add_argument("--expected-samples",type=int,default=512)
    parser.add_argument("--bootstrap",type=int,default=4096)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh repetition output directory")
    if len(set(args.seeds)) != len(args.seeds) or args.bootstrap < 1:
        raise ValueError("Distinct seeds and a positive bootstrap budget are required")
    reports = []
    for seed in args.seeds:
        folder = output/f"seed_{seed}"
        report = evaluate_mesh_directory(args.decode_dir,folder,sample_points=args.sample_points,
            seed=seed,expected_samples=args.expected_samples)
        report["report_sha256"] = file_sha256(folder/'geometry.json')
        reports.append(report)
        print(f"Sampling seed {seed}: {args.expected_samples} objects evaluated",flush=True)
    result = summarize_repeats(reports,bootstrap=args.bootstrap)
    result["source_reports"] = [{"seed":r["seed"],"sha256":r["report_sha256"]} for r in reports]
    manifest = Path(args.decode_dir)/"decoded_manifest.json"
    if manifest.exists():
        source = json.loads(manifest.read_text())
        result["decode"] = {key:source[key] for key in ("tokenizer_hash","source","prior_bindings_verified","prior_sha256","protocol") if key in source}
        result["decode_manifest_sha256"] = file_sha256(manifest)
    (output/'boundary_repetition.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({"output":str(output/'boundary_repetition.json'),"paired_edge":result['paired_edge']},indent=2))


if __name__ == "__main__":
    main()
