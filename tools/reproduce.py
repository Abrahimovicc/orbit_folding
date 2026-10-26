#!/usr/bin/env python3
# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Explicit train-to-metrics workflow, using only the public CLI and data contracts."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml

from orbit_folding.provenance import dataset_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Small models/two updates; NEVER a benchmark-quality claim")
    parser.add_argument("--resolution", type=int, default=56)
    parser.add_argument("--precision", choices=["32-true", "bf16-mixed", "16-mixed"])
    parser.add_argument("--tokenizer-batch-size", type=int)
    parser.add_argument("--tokenizer-accumulation", type=int)
    parser.add_argument("--categorical-batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-selection", choices=["last", "best"], default="last",
                        help="Paper uses fixed-budget last checkpoints; best is a separate validation-selected experiment")
    args = parser.parse_args()
    data, output = Path(args.data).resolve(), Path(args.output).resolve()
    if args.gpus < 1 or args.workers < 0 or (args.gpus > 1 and not args.device.startswith("cuda")):
        raise ValueError("Positive GPU count, nonnegative workers; multi-GPU requires CUDA")
    if output.is_relative_to(data) or data.is_relative_to(output):
        raise ValueError("Workflow output and input dataset must be separate trees")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Workflow needs a fresh output directory; resume individual CLI runs explicitly")
    manifest = json.loads((data / "manifest.json").read_text())
    test_ids = {entry["id"] for entry in manifest["samples"] if entry["split"] == "test"}
    source_split = json.loads((ROOT / "data/splits/matched_completion.json").read_text())
    expected_ids = {identity.replace("/", "__").replace(".npz", "") for identity in source_split["test"]}
    expected_train = {identity.replace("/", "__").replace(".npz", "") for identity in source_split["train"]}
    train_ids = {entry["id"] for entry in manifest["samples"] if entry["split"] in {"train", "val"}}
    if not args.smoke and (test_ids != expected_ids or train_ids != expected_train):
        raise ValueError("Full primary experiment requires the original 512 test IDs; --smoke is functionality only")
    if not any(entry["split"] == "val" for entry in manifest["samples"]):
        raise ValueError("This workflow requires an explicit validation split")
    output.mkdir(parents=True, exist_ok=True)
    progress = {"format": "orbit_folding.reproduction_workflow.v1", "smoke": args.smoke,
        "dataset_fingerprint": dataset_fingerprint(data), "primary_test_cohort_verified": test_ids == expected_ids,
        "primary_train_cohort_verified": train_ids == expected_train,
        "steps": [], "status": "running", "checkpoint_selection": args.checkpoint_selection,
        "scope": "Training budget completion is not proof of convergence or reported benchmark quality."}
    progress["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src/orbit_folding").rglob("*.py"))}
    path = output / "workflow.json"

    def save():
        path.write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")

    def run(name, arguments, distributed=False):
        prefix = [sys.executable, "-m", "orbit_folding"]
        if distributed and args.gpus > 1:
            prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={args.gpus}", "-m", "orbit_folding"]
        row = {"name": name, "status": "running", "command": list(map(str, prefix + arguments))}
        progress["steps"].append(row)
        save()
        started = time.perf_counter()
        with (output / f"{name}.log").open("w") as stream:
            with subprocess.Popen(row["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT) as process:
                notice = started
                try:
                    while process.poll() is None:
                        time.sleep(.2)
                        if time.perf_counter() - notice >= 30:
                            notice = time.perf_counter()
                            with (output / f"{name}.log").open("rb") as log:
                                log.seek(max(0, log.seek(0, 2) - 2048))
                                lines = log.read().decode("utf-8", errors="replace").splitlines()
                            row.update(elapsed_seconds=notice-started, last_log_line=lines[-1] if lines else "initializing")
                            save()
                            print(f"{name}: {row['elapsed_seconds']/60:.1f} min | {row['last_log_line']}", flush=True)
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
                returncode = process.returncode
        row.update(returncode=returncode, elapsed_seconds=time.perf_counter()-started,
                   status="complete" if returncode == 0 else "failed")
        save()
        print(f"{name}: {row['status']} ({row['elapsed_seconds']:.1f}s)", flush=True)
        if returncode:
            raise RuntimeError(f"{name} failed; inspect its log before retrying in a new run directory")

    def selected(folder):
        return folder / ("best.pt" if args.checkpoint_selection == "best" and (folder / "best.pt").exists() else "last.pt")

    def configured(name, tokenizer=False):
        config = yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text())
        if args.precision:
            config.setdefault("Trainer", {})["precision"] = args.precision
        if tokenizer and args.tokenizer_batch_size is not None:
            if args.tokenizer_batch_size < 1:
                raise ValueError("tokenizer batch must be positive")
            config["batch_size"] = args.tokenizer_batch_size
        if tokenizer and args.tokenizer_accumulation is not None:
            if args.tokenizer_accumulation < 1:
                raise ValueError("tokenizer accumulation must be positive")
            config.setdefault("Trainer", {})["accumulate_grad_batches"] = args.tokenizer_accumulation
        target = output / f"{name}.yaml"
        target.write_text(yaml.safe_dump(config, sort_keys=False))
        return target

    try:
        tokenizer = output / "tokenizer"
        cache = output / "tokens"
        model_config = "tokenizer_demo" if args.smoke else "tokenizer_matched"
        training = ["--device", args.device, "--workers", str(args.workers)]
        budget = ["--epochs", "2", "--max-steps", "2"] if args.smoke else []
        run("train_tokenizer", ["train-tokenizer", "--config", configured(model_config, True), "--data", data,
            "--output", tokenizer, *training, *budget], True)
        run("cache_tokens", ["cache-tokens", "--checkpoint", selected(tokenizer), "--data", data,
            "--output", cache, "--device", args.device, "--workers", str(args.workers)])
        for kind, config in (("bit", "prior_demo" if args.smoke else "prior_paper"),
                             ("categorical", "prior_categorical_demo" if args.smoke else "prior_categorical_paper")):
            folder = output / kind
            run(f"train_{kind}", ["train-prior", "--config", configured(config), "--data", cache,
                "--output", folder, *training, *budget, *(["--batch-size", str(args.categorical_batch_size)] if kind == "categorical" else [])], True)
            run(f"evaluate_{kind}", ["evaluate", "tokens", "--tokens", cache, "--prior", selected(folder),
                "--tokenizer", selected(tokenizer), "--device", args.device, "--output", output / f"metrics_{kind}"])
        run("decoded_geometry", ["benchmark", "decode", "--tokens", cache, "--tokenizer", selected(tokenizer),
            "--bit-prior", selected(output / "bit"), "--categorical-prior", selected(output / "categorical"),
            "--resolution", str(8 if args.smoke else args.resolution), "--device", args.device,
            "--output", output / "decoded"])
        measured = json.loads((output / "metrics_bit/token_prior.json").read_text())
        row = next(row for row in measured["rows"] if row["mask_ratio"] == .5)
        progress.update(status="complete", full_training_budget_executed=not args.smoke,
            measured_50pct={key:row[key] for key in ("bit_accuracy", "token_exact")},
            numerical_agreement="not_applicable_smoke" if args.smoke else "inspect measurements; no convergence claim is inferred")
        save()
    except Exception:
        progress["status"] = "failed"
        save()
        raise


if __name__ == "__main__":
    main()
