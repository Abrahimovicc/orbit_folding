"""Geometric Orbit Representatives for Unaligned 3D Tokenization.

Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig.
Contact: Xiang Chen <xiang.chen@uni-leipzig.de>. SPDX-License-Identifier: MIT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def _train_arguments(parser, kind):
    parser.add_argument("--config", default=f"configs/{kind}_demo.yaml")
    parser.add_argument("--data", default="data/samples/processed" if kind == "tokenizer" else "data/tokens")
    parser.add_argument("--output", default=f"outputs/{kind}")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto; torchrun assigns LOCAL_RANK")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Per-process batch size")
    parser.add_argument("--max-steps", type=int, help="Maximum optimizer updates, not data batches")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, help="Override config seed (default: config seed, otherwise 2026)")
    initialize = parser.add_mutually_exclusive_group()
    initialize.add_argument("--resume", help="Release checkpoint with optimizer/RNG state")
    initialize.add_argument("--init-checkpoint", help="Load compatible weights into a fresh run; reset optimizer, schedules and RNG")


def _evaluation_arguments(parser):
    parser.add_argument("--data", default="data/samples/processed")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=2026)


def parser():
    root = argparse.ArgumentParser(prog="orbit", description="Geometric representatives, reconstructive tokens, and masked bit-field completion")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("preprocess", help="Validate meshes and create model-space SDF data")
    prepare.add_argument("--input-dir", default="data/samples/raw")
    prepare.add_argument("--output", default="data/samples/processed")
    prepare.add_argument("--surface-points", type=int, default=8192)
    prepare.add_argument("--query-points", type=int, default=32768)
    prepare.add_argument("--uniform-points", type=int, default=8192)
    prepare.add_argument("--split-file")
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.add_argument("--normalization", choices=["centroid", "legacy-bbox"], default="centroid")
    prepare.add_argument("--axis-lock", action=argparse.BooleanOptionalAction, default=True)
    imported = commands.add_parser("import-data", help="Import existing validated NPZs using an explicit split")
    imported.add_argument("--input-dir", required=True)
    imported.add_argument("--split-file", required=True)
    imported.add_argument("--output", required=True)
    imported.add_argument("--input-scale", type=float, default=2.0)
    imported.add_argument("--axis-lock", action=argparse.BooleanOptionalAction, default=True)
    imported.add_argument("--legacy-uniform-start", type=int, help="Explicit opt-in query suffix offset for validated legacy NPZs without segment metadata")
    imported.add_argument("--frame-cache", nargs="+", help="Optional original full-observation frame-cache NPZs, covering every imported identity")
    imported.add_argument("--storage", choices=["compressed", "uncompressed"], default="compressed",
                          help="Uncompressed NPZs trade disk space for faster import/loading")
    verify = commands.add_parser("verify-data", help="Verify bundled geometry and license-file SHA-256 hashes")
    verify.add_argument("--manifest", default="data/manifest.json")
    for kind in ["tokenizer", "prior"]:
        _train_arguments(commands.add_parser(f"train-{kind}", help=f"Train the {kind} on one process or torchrun"), kind)
    calibrated = commands.add_parser("calibrate-prior", help="Fit one posterior temperature on the explicit validation split")
    calibrated.add_argument("--checkpoint", required=True)
    calibrated.add_argument("--tokens", required=True)
    calibrated.add_argument("--output", required=True)
    calibrated.add_argument("--device", default="cpu")
    calibrated.add_argument("--mask-ratio", type=float, default=.75)
    calibrated.add_argument("--batch-size", type=int, default=4)
    calibrated.add_argument("--seed", type=int, default=20260511)
    calibrated.add_argument("--threads", type=int, default=4)
    field_calibration = commands.add_parser("calibrate-field", help="Fit one shared SDF-head bias on train with held-out validation guards")
    field_calibration.add_argument("--checkpoint", required=True)
    field_calibration.add_argument("--data", required=True)
    field_calibration.add_argument("--output", required=True)
    field_calibration.add_argument("--device", default="cpu")
    field_calibration.add_argument("--weighting", choices=["uniform", "balanced"], default="uniform")
    field_calibration.add_argument("--train-samples", type=int, default=128)
    field_calibration.add_argument("--observation-points", type=int, default=2048)
    field_calibration.add_argument("--seed", type=int, default=20260513)
    field_calibration.add_argument("--threads", type=int, default=4)
    cache = commands.add_parser("cache-tokens", help="Cache binary LFQ fields without changing sample splits")
    cache.add_argument("--checkpoint", required=True)
    cache.add_argument("--data", default="data/samples/processed")
    cache.add_argument("--output", default="data/tokens")
    cache.add_argument("--device", default="cpu")
    cache.add_argument("--batch-size", type=int, default=1)
    cache.add_argument("--seed", type=int, default=2026)
    cache.add_argument("--workers", type=int, default=0)
    cache.add_argument("--threads", type=int, default=4)
    cache.add_argument("--sampling-protocol", choices=["portable", "reference"], default="portable",
                       help="Reference replays the original per-cell selection algorithm with a recorded local seed; not its unrecorded RNG stream")
    evaluate = commands.add_parser("evaluate", help="Independent stability, reconstruction, and completion evaluation")
    evaluators = evaluate.add_subparsers(dest="evaluation", required=True)
    stability = evaluators.add_parser("stability")
    _evaluation_arguments(stability)
    stability.add_argument("--jitter", type=float, default=0.0, help="Independent Gaussian jitter; see evaluation report for coordinate units")
    reconstruction = evaluators.add_parser("reconstruction")
    _evaluation_arguments(reconstruction)
    reconstruction.add_argument("--checkpoint", required=True)
    reconstruction.add_argument("--resolution", type=int, default=32)
    reconstruction.add_argument("--bounds", type=float, nargs=2, default=(-0.5, 0.5), metavar=("MIN", "MAX"))
    reconstruction.add_argument("--save-meshes", action="store_true")
    completion = evaluators.add_parser("completion")
    _evaluation_arguments(completion)
    completion.add_argument("--prior", required=True)
    completion.add_argument("--tokenizer", required=True)
    completion.add_argument("--tokens", default="data/tokens")
    completion.add_argument("--mask-ratio", type=float, default=0.75)
    completion.add_argument("--resolution", type=int, default=32)
    completion.add_argument("--bounds", type=float, nargs=2, default=(-0.5, 0.5), metavar=("MIN", "MAX"))
    completion.add_argument("--save-meshes", action="store_true")
    completion.add_argument("--allow-unverified-prior", action="store_true", help="Explicit legacy-only override for a missing prior binding; known mismatches still fail")
    completion.add_argument("--allow-unverified-geometry", action="store_true", help="Explicit legacy-only override for a cache with no geometry fingerprint")
    tokens = evaluators.add_parser("tokens", help="Fixed-Bernoulli bit/index prior evaluation; LFQ-state diagnostics for bit priors")
    tokens.add_argument("--prior", required=True)
    tokens.add_argument("--tokenizer")
    token_source = tokens.add_mutually_exclusive_group(required=True)
    token_source.add_argument("--tokens", help="Release-format token cache")
    token_source.add_argument("--indices", help="External integer NHW/flattened token-index NPY")
    tokens.add_argument("--indices-metadata", help="Required metadata JSON for external indices")
    tokens.add_argument("--allow-unverified-indices", action="store_true")
    tokens.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    tokens.add_argument("--output", default="outputs/token_prior")
    tokens.add_argument("--device", default="cpu")
    tokens.add_argument("--mask-device", default="cpu")
    tokens.add_argument("--seed", type=int, default=20260511)
    tokens.add_argument("--batch-size", type=int, help="Default: 16 for bit priors, 1 for categorical priors")
    tokens.add_argument("--max-samples", type=int)
    tokens.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75, 0.9])
    tokens.add_argument("--uncertain-threshold", type=float, default=0.9)
    tokens.add_argument("--threads", type=int, default=4)
    benchmark = commands.add_parser("benchmark", help="Paper metric protocols, separate from general diagnostics")
    benchmarks = benchmark.add_subparsers(dest="benchmark", required=True)
    fidelity = benchmarks.add_parser("fidelity", help="Paper observed-cloud reconstruction, CD/EMD and signed-field fidelity")
    fidelity.add_argument("--checkpoint", required=True)
    fidelity.add_argument("--data", required=True)
    fidelity.add_argument("--output", required=True)
    fidelity.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    fidelity.add_argument("--device", default="cpu")
    fidelity.add_argument("--max-samples", type=int)
    fidelity.add_argument("--resolution", type=int, default=64)
    fidelity.add_argument("--metric-points", type=int, default=2048)
    fidelity.add_argument("--emd-points", type=int, default=512)
    fidelity.add_argument("--encoder-points", type=int, help="Fix the initial observation draw independently of training config")
    fidelity.add_argument("--query-points", type=int, help="Fix the signed-query budget independently of training config")
    fidelity.add_argument("--uniform-points", type=int, help="Fix the uniform-query sampling budget independently of training config")
    fidelity.add_argument("--seed", type=int, default=20260513)
    fidelity.add_argument("--sampling-protocol", choices=["portable", "reference"], default="reference")
    fidelity.add_argument("--threads", type=int, default=4)
    meshes = benchmarks.add_parser("geometry", help="Normalized mesh CD, drift failure, F-score and oracle boundary distance")
    meshes.add_argument("--decode-dir", required=True)
    meshes.add_argument("--output", required=True)
    meshes.add_argument("--sample-points", type=int, default=1024)
    meshes.add_argument("--seed", type=int, default=20260518)
    meshes.add_argument("--expected-samples", type=int)
    meshes.add_argument("--edge-percentile", type=float, default=80)
    meshes.add_argument("--drift-threshold", type=float, default=.005)
    decoded = benchmarks.add_parser("decode", help="Same-mask frozen-decoder cohort, followed by paper geometry metrics")
    decoded.add_argument("--tokenizer", required=True)
    decoded.add_argument("--bit-prior")
    decoded.add_argument("--categorical-prior")
    source = decoded.add_mutually_exclusive_group(required=True)
    source.add_argument("--tokens")
    source.add_argument("--indices")
    decoded.add_argument("--indices-metadata")
    decoded.add_argument("--allow-unverified-indices", action="store_true")
    decoded.add_argument("--allow-unverified-prior", action="store_true")
    decoded.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    selection = decoded.add_mutually_exclusive_group()
    selection.add_argument("--max-samples", type=int)
    selection.add_argument("--sample-indices", help="JSON list of explicit token-cache row indices")
    decoded.add_argument("--mask-npy", help="Optional original NHW masks, in SELECTED sample order")
    decoded.add_argument("--mask-ratio", type=float, default=.75)
    decoded.add_argument("--seed", type=int, default=20260516)
    decoded.add_argument("--resolution", type=int, default=56)
    decoded.add_argument("--device", default="cpu")
    decoded.add_argument("--output", required=True)
    decoded.add_argument("--threads", type=int, default=4)
    stress = benchmarks.add_parser("stability", help="Dataset adapters and matched fixed-observation/noise stress")
    stress.add_argument("--data", required=True)
    stress.add_argument("--format", choices=["observations", "meshes", "shapenet", "scanobjectnn"], default="observations")
    stress.add_argument("--split-file")
    stress.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    stress.add_argument("--limit", type=int)
    stress.add_argument("--expected-samples", type=int)
    stress.add_argument("--points", type=int, default=2048)
    stress.add_argument("--categories", nargs="+")
    stress.add_argument("--max-per-category", type=int)
    stress.add_argument("--seed", type=int, default=20260704)
    stress.add_argument("--jitter-sigma", type=float, default=.005, help="UNIT-RADIUS sigma; model sigma is half this value")
    stress.add_argument("--device", default="cpu")
    stress.add_argument("--threads", type=int, default=4)
    stress.add_argument("--output", required=True)
    support_compare = benchmarks.add_parser("support", help="Fresh matched VQ/LFQ cache comparison")
    support_compare.add_argument("--vq-tokens", required=True)
    support_compare.add_argument("--lfq-tokens", required=True)
    support_compare.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    support_compare.add_argument("--vocabulary", type=int, default=8192)
    support_compare.add_argument("--output", required=True)
    visualize = commands.add_parser("visualize", help="Render observed and folded point-cloud PNG/GIF comparisons")
    visualize.add_argument("--data", default="data/samples/processed")
    visualize.add_argument("--output", default="outputs/visualization")
    visualize.add_argument("--sample-id")
    visualize.add_argument("--frames", type=int, default=48)
    visualize.add_argument("--points", type=int, default=8192)
    visualize.add_argument("--renderer", choices=["software", "open3d"], default="software")
    visualize.add_argument("--width", type=int, default=1440)
    visualize.add_argument("--height", type=int, default=560)
    visualize.add_argument("--supersample", type=int, default=2)
    visualize.add_argument("--fps", type=int, default=12)
    visualize.add_argument("--max-bytes", type=int, default=16000000)
    monitor = commands.add_parser("plot-training", help="Plot real training logs and optional checkpoint diagnostics")
    monitor.add_argument("--run", required=True)
    monitor.add_argument("--output", help="Curve PNG, defaults to RUN/training.png")
    monitor.add_argument("--data", help="Processed geometry or token cache, enables a checkpoint preview")
    monitor.add_argument("--split", choices=["train", "val", "test"], default="val")
    monitor.add_argument("--sample-id")
    monitor.add_argument("--device", default="cpu")
    support = commands.add_parser("analyze-support", help="Plot codebook concentration, nominal usage and entropy support")
    source = support.add_mutually_exclusive_group(required=True)
    source.add_argument("--stats", help="JSON containing integer count histograms, including the paper reference")
    source.add_argument("--tokens", help="Release-format LFQ bit or VQ index cache")
    support.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    support.add_argument("--output", default="outputs/token_support")
    encode = commands.add_parser("encode", help="Encode a mesh or point observation without SDF labels")
    encode.add_argument("--checkpoint", required=True)
    encode.add_argument("--input", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--observation-points", type=int, default=8192)
    normalization = encode.add_mutually_exclusive_group()
    normalization.add_argument("--model-space", action="store_true", help="Input is already normalized; do not apply a second scale")
    normalization.add_argument("--normalization", choices=["centroid", "legacy-bbox"], help="Default: centroid; legacy-bbox preserves the earlier input convention")
    encode.add_argument("--sampling-frame", choices=["representative", "observation"], help="Spatial sampling coordinates; default representative, or observation with legacy-bbox")
    decode = commands.add_parser("decode", help="Decode encoded observations, optionally after masked completion")
    decode.add_argument("--checkpoint", required=True)
    decode.add_argument("--tokens", required=True)
    decode.add_argument("--output", required=True)
    decode.add_argument("--prior")
    decode.add_argument("--mask-ratio", type=float, default=0.5)
    decode.add_argument("--state", choices=["oracle", "hard", "relaxed"])
    decode.add_argument("--space", choices=["input", "representative"], default="input")
    decode.add_argument("--resolution", type=int, default=96)
    decode.add_argument("--bounds", type=float, nargs=2, default=(-0.5, 0.5), metavar=("MIN", "MAX"))
    decode.add_argument("--component-min-fraction", type=float, default=0.)
    decode.add_argument("--component-min-triangles", type=int, default=0)
    decode.add_argument("--component-min-extent-fraction", type=float, default=0.)
    decode.add_argument("--query-chunk-size", type=int, default=8192)
    decode.add_argument("--allow-unverified-prior", action="store_true")
    for command in (encode, decode):
        command.add_argument("--device", default="cpu")
        command.add_argument("--seed", type=int, default=2026)
        command.add_argument("--threads", type=int, default=4)
    weights = commands.add_parser("verify-weights", help="Verify pretrained pair checksums and tokenizer/prior bindings")
    weights.add_argument("--directory", required=True)
    convert = commands.add_parser("convert-checkpoint", help="Strictly migrate trusted research weights to the portable format")
    convert.add_argument("--source", required=True)
    convert.add_argument("--config", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--kind", choices=["tokenizer", "prior"], required=True)
    convert.add_argument("--trust-original", action="store_true", help="Allow unsafe pickle ONLY for a source checkpoint you independently trust")
    return root


def _config(path):
    result = yaml.safe_load(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "preprocess":
        from .preprocessing import preprocess
        result = preprocess(args.input_dir, args.output, args.surface_points, args.query_points,
                            args.uniform_points, args.seed, args.split_file,
                            normalization=args.normalization, axis_lock=args.axis_lock)
        print(f"Prepared {len(result['samples'])} samples in {args.output}")
    elif args.command == "import-data":
        from .preprocessing import import_preprocessed
        result = import_preprocessed(args.input_dir, args.split_file, args.output, args.input_scale,
                                     axis_lock=args.axis_lock, legacy_uniform_start=args.legacy_uniform_start,
                                     frame_caches=args.frame_cache, storage=args.storage)
        print(f"Imported {len(result['samples'])} samples in {args.output}")
    elif args.command == "verify-data":
        path = Path(args.manifest)
        manifest = json.loads(path.read_text())
        entries = manifest["assets"] + list(manifest["license_texts"].values())
        for item in entries:
            artifact = (path.parent / item["path"]).resolve()
            if not artifact.is_relative_to(path.parent.resolve()):
                raise ValueError("Manifest path escapes its directory")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Checksum mismatch: {item['path']}")
        print(f"Verified {len(manifest['assets'])} meshes and {len(manifest['license_texts'])} license texts")
    elif args.command.startswith("train-"):
        from .training import train
        config = _config(args.config)
        if config.get("Task") != args.command.removeprefix("train-"):
            raise ValueError("Configuration Task does not match this training command")
        print(train(config, args.data, args.output, device=args.device, epochs=args.epochs,
                    batch_size=args.batch_size, max_steps=args.max_steps, workers=args.workers,
                    resume=args.resume, init_checkpoint=args.init_checkpoint, seed=args.seed, threads=args.threads))
    elif args.command == "cache-tokens":
        import torch
        from torch.utils.data import DataLoader
        from .checkpoints import load_model
        from .token_cache import export_tokens
        from .training import geometry_dataset
        model, config = load_model(args.checkpoint, args.device)
        if config["Task"] != "tokenizer":
            raise ValueError("Token export needs a tokenizer checkpoint")
        if args.workers < 0:
            raise ValueError("workers must be nonnegative")
        torch.set_num_threads(max(1, args.threads))
        dataset = geometry_dataset(args.data, config, "all", args.seed, points_only=True,
                                   sampling_protocol=args.sampling_protocol)
        result = export_tokens(model, DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=torch.device(args.device).type == "cuda"), args.output, args.device)
        print(json.dumps({"samples": result["num_samples"], "splits": result["split_counts"]}, indent=2))
    elif args.command == "evaluate":
        if args.evaluation == "tokens":
            import torch
            from .paper_evaluation import evaluate_token_prior

            torch.set_num_threads(max(1, args.threads))
            result = evaluate_token_prior(args.prior, token_root=args.tokens, indices=args.indices,
                indices_metadata=args.indices_metadata, tokenizer_checkpoint=args.tokenizer,
                split=args.split, output=args.output, device=args.device, mask_device=args.mask_device,
                seed=args.seed, batch_size=args.batch_size, max_samples=args.max_samples,
                ratios=args.ratios, uncertain_threshold=args.uncertain_threshold,
                allow_unverified_indices=args.allow_unverified_indices)
            print(json.dumps({key: result[key] for key in ("evaluation", "output", "samples", "rows")}, indent=2))
            return
        from . import evaluation
        common = dict(data_root=args.data, split=args.split, output=args.output,
                      device=args.device, max_samples=args.max_samples, seed=args.seed)
        if args.evaluation == "stability":
            result = evaluation.eval_stability(**common, jitter=args.jitter)
        elif args.evaluation == "reconstruction":
            result = evaluation.eval_reconstruction(**common, checkpoint=args.checkpoint, resolution=args.resolution,
                                                     save_meshes=args.save_meshes, bounds=args.bounds)
        else:
            result = evaluation.eval_completion(**common, prior_checkpoint=args.prior,
                       tokenizer_checkpoint=args.tokenizer, token_root=args.tokens,
                       ratio=args.mask_ratio, resolution=args.resolution, save_meshes=args.save_meshes, bounds=args.bounds,
                       allow_unverified_prior=args.allow_unverified_prior,
                       allow_unverified_geometry=args.allow_unverified_geometry)
        print(json.dumps({key: result[key] for key in ["evaluation", "output", "attempted_samples", "successful_samples"]
                          if key in result}, indent=2))
    elif args.command == "calibrate-field":
        import torch
        from .calibration import calibrate_field

        torch.set_num_threads(max(1, args.threads))
        print(json.dumps(calibrate_field(args.checkpoint, args.data, args.output, device=args.device,
            weighting=args.weighting, train_samples=args.train_samples,
            observation_points=args.observation_points, seed=args.seed), indent=2))
    elif args.command == "calibrate-prior":
        import torch
        from .calibration import calibrate_prior

        torch.set_num_threads(max(1, args.threads))
        print(json.dumps(calibrate_prior(args.checkpoint, args.tokens, args.output, device=args.device,
            mask_ratio=args.mask_ratio, seed=args.seed, batch_size=args.batch_size), indent=2))
    elif args.command == "benchmark":
        if args.benchmark == "fidelity":
            import torch
            from .fidelity_benchmark import evaluate_fidelity

            torch.set_num_threads(max(1, args.threads))
            result = evaluate_fidelity(args.checkpoint, args.data, args.output, split=args.split,
                device=args.device, max_samples=args.max_samples, resolution=args.resolution,
                metric_points=args.metric_points, emd_points=args.emd_points, seed=args.seed,
                sampling_protocol=args.sampling_protocol, encoder_points=args.encoder_points,
                query_points=args.query_points, uniform_points=args.uniform_points)
        elif args.benchmark == "geometry":
            from .mesh_benchmark import evaluate_mesh_directory
            result = evaluate_mesh_directory(args.decode_dir, args.output, sample_points=args.sample_points,
                seed=args.seed, edge_percentile=args.edge_percentile, drift_threshold=args.drift_threshold,
                expected_samples=args.expected_samples)
        elif args.benchmark == "decode":
            import torch
            from .benchmark_decode import decode_benchmark
            from .mesh_benchmark import evaluate_mesh_directory
            torch.set_num_threads(max(1, args.threads))
            decode_benchmark(args.tokenizer, args.output, bit_prior=args.bit_prior,
                categorical_prior=args.categorical_prior, token_root=args.tokens, indices=args.indices,
                indices_metadata=args.indices_metadata, split=args.split, device=args.device,
                max_samples=args.max_samples, sample_indices=json.loads(Path(args.sample_indices).read_text()) if args.sample_indices else None,
                ratio=args.mask_ratio, seed=args.seed, mask_npy=args.mask_npy, resolution=args.resolution,
                allow_unverified_indices=args.allow_unverified_indices, allow_unverified_prior=args.allow_unverified_prior)
            result = evaluate_mesh_directory(args.output, Path(args.output) / "metrics")
        elif args.benchmark == "stability":
            import torch
            from .stress_benchmark import evaluate_stress
            torch.set_num_threads(max(1, args.threads))
            result = evaluate_stress(args.data, args.output, kind=args.format, split=args.split,
                split_file=args.split_file, limit=args.limit, expected_samples=args.expected_samples,
                points=args.points, seed=args.seed, jitter_sigma_unit_radius=args.jitter_sigma, device=args.device,
                categories=args.categories, max_per_category=args.max_per_category)
        else:
            from .token_support import compare_caches
            result = compare_caches(args.vq_tokens, args.lfq_tokens, args.output, split=args.split, vocabulary=args.vocabulary)
        print(json.dumps({key: result[key] for key in ["format", "samples", "requested_samples", "summary", "output"] if key in result}, indent=2))
    elif args.command == "visualize":
        from .visualization import render_alignment
        print(render_alignment(args.data, args.output, sample_id=args.sample_id, frames=args.frames,
                               points=args.points, renderer=args.renderer, width=args.width, height=args.height,
                               supersample=args.supersample, fps=args.fps, max_bytes=args.max_bytes))
    elif args.command == "plot-training":
        from .monitoring import plot_training, preview_checkpoint
        print(f"Training curves: {plot_training(args.run, args.output)}")
        if args.data:
            print(f"Checkpoint preview: {preview_checkpoint(args.run, args.data, split=args.split, sample_id=args.sample_id, device=args.device)}")
    elif args.command == "analyze-support":
        from .token_support import cache_histogram, plot_support
        stats = args.stats if args.stats else cache_histogram(args.tokens, args.split)
        report = plot_support(stats, args.output)
        print(json.dumps({"output": args.output, "series": [
            {key: row[key] for key in ("name", "nominal_usage", "effective_support", "codes_for_50")}
            for row in report["series"]]}, indent=2))
    elif args.command in {"encode", "decode"}:
        import torch
        from .inference import encode_observation, decode_observation

        torch.set_num_threads(max(1, args.threads))
        if args.command == "encode":
            result = encode_observation(args.checkpoint, args.input, args.output, device=args.device,
                                        seed=args.seed, observation_points=args.observation_points,
                                        model_space=args.model_space, normalization=args.normalization,
                                        sampling_frame=args.sampling_frame)
        else:
            result = decode_observation(args.checkpoint, args.tokens, args.output, device=args.device,
                seed=args.seed, resolution=args.resolution, space=args.space, prior_checkpoint=args.prior,
                mask_ratio=args.mask_ratio, state=args.state, allow_unverified_prior=args.allow_unverified_prior,
                bounds=args.bounds, component_min_fraction=args.component_min_fraction,
                component_min_triangles=args.component_min_triangles,
                component_min_extent_fraction=args.component_min_extent_fraction,
                query_chunk_size=args.query_chunk_size)
        print(json.dumps(result, indent=2))
        if args.command == "decode" and not result["mesh"]["valid"]:
            raise SystemExit(2)
    elif args.command == "verify-weights":
        from .weights import verify_weights
        manifest = verify_weights(args.directory)
        print(json.dumps({"verified_pairs": [pair["id"] for pair in manifest["pairs"]]}, indent=2))
    elif args.command == "convert-checkpoint":
        from .checkpoints import convert_original_checkpoint
        convert_original_checkpoint(args.source, _config(args.config), args.output,
                                    args.kind, trusted=args.trust_original)
        print(f"Converted checkpoint: {args.output}")
