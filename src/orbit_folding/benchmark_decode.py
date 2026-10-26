# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Same-mask frozen-decoder mesh cohort for the paper boundary benchmark."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_model, verify_prior_binding
from .evaluation import _decode, _grid, _limit
from .metrics import extract_mesh
from .meshing import MODEL_BOUNDS, clean_components
from .paper_evaluation import _token_source
from .provenance import file_sha256
from .token_cache import tokenizer_hash


@torch.random.fork_rng(devices=[])
@torch.no_grad()
def decode_benchmark(tokenizer_checkpoint, output, *, bit_prior=None, categorical_prior=None,
                     token_root=None, indices=None, indices_metadata=None, split="test", device="cpu",
                     max_samples=None, sample_indices=None, ratio=.75, seed=20260516, mask_npy=None,
                     resolution=56, chunk_size=8192, bounds=MODEL_BOUNDS, min_fraction=.02, min_triangles=64,
                     allow_unverified_indices=False, allow_unverified_prior=False):
    import open3d as o3d

    if (not np.isfinite(ratio) or not 0 <= ratio <= 1 or chunk_size < 1 or min_triangles < 0
            or not 0 <= min_fraction <= 1 or (sample_indices is not None and max_samples is not None)):
        raise ValueError("Invalid decode protocol or conflicting sample selectors")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Benchmark decoding requires an empty output directory")
    tokenizer, config = load_model(tokenizer_checkpoint, device)
    if config["Task"] != "tokenizer" or not (bit_prior or categorical_prior):
        raise ValueError("Supply a tokenizer and at least one completion prior")
    identity = tokenizer_hash(tokenizer)
    priors, bindings = {}, {}
    for kind, path in (("bit_field", bit_prior), ("categorical", categorical_prior)):
        if path is None:
            continue
        model, specs = load_model(path, device)
        if specs["Task"] != "prior" or model.prior_type != kind or model.codebook_size != tokenizer.tokenizer.codebook_size:
            raise ValueError("Prior type/vocabulary does not match the requested decoder")
        bindings[kind] = verify_prior_binding(model, identity, allow_unverified=allow_unverified_prior)
        priors[kind] = model
    lfq = tokenizer.quantizer_type == "lfq"
    if not lfq and bit_prior is not None:
        raise ValueError("Bit/relaxed decoding requires LFQ, not VQ indices")
    count, load_batch, provenance = _token_source(token_root, indices, indices_metadata, split,
        getattr(tokenizer.tokenizer, "num_bits", None), identity, allow_unverified_indices,
        categorical=not lfq, codebook_size=tokenizer.tokenizer.codebook_size)
    selected = np.arange(count)[:_limit(count, max_samples)] if sample_indices is None else np.asarray(sample_indices)
    if (selected.ndim != 1 or selected.dtype.kind not in "iu" or not len(selected)
            or selected.min() < 0 or selected.max() >= count or len(np.unique(selected)) != len(selected)):
        raise ValueError("Sample indices must be nonempty, unique integers inside the token array")
    mask_source = None
    if mask_npy is not None:
        mask_source = np.load(mask_npy, mmap_mode="r", allow_pickle=False)
        if mask_source.ndim == 4 and mask_source.shape[1] == 1:
            mask_source = mask_source[:, 0]
        if (mask_source.ndim != 3 or len(mask_source) != len(selected)
                or mask_source.dtype.kind not in "bui" or not np.all((mask_source == 0) | (mask_source == 1))):
            raise ValueError("Saved masks must be binary NHW with one row per SELECTED sample")
    generator = torch.Generator().manual_seed(seed + round(ratio * 10000))
    grid = _grid(resolution, bounds, device)
    modes = (["relaxed", "hard"] if bit_prior else []) + (["categorical"] if categorical_prior else [])
    root.mkdir(parents=True, exist_ok=True)
    for name in ["oracle", *modes]:
        (root / name).mkdir()
    rows, masks = [], []
    for position, source_index in enumerate(selected):
        values = load_batch([int(source_index)]).to(device)
        height, width = values.shape[-2:]
        mask = torch.rand((1, 1, height, width), generator=generator) < ratio if mask_source is None else torch.from_numpy(np.array(mask_source[position], dtype=bool))[None, None]
        if mask.shape[-2:] != (height, width):
            raise ValueError("Saved mask spatial shape differs from tokens")
        masks.append(mask.numpy()[0, 0])
        mask = mask.to(device)
        if lfq:
            from .models.categorical import batch_indices
            oracle = batch_indices({"bits": values}, tokenizer.tokenizer.codebook_size)
        else:
            oracle = values
        states = {"oracle": (oracle, False)}
        if bit_prior:
            probabilities = priors["bit_field"].model(torch.where(mask, .5, values), ~mask).sigmoid()
            states["relaxed"] = (torch.where(mask, probabilities, values), True)
            states["hard"] = (torch.where(mask, (probabilities >= .5).float(), values), True)
        if categorical_prior:
            model = priors["categorical"]
            logits = model.model(torch.where(mask[:, 0], model.model.mask_token_id, oracle))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite categorical logits")
            states["categorical"] = (torch.where(mask[:, 0], logits.argmax(-1), oracle), False)
        row = {"id": f"sample_{position:06d}", "source_index": int(source_index), "mesh_status": {}}
        for name, (state, relaxed) in states.items():
            planes = tokenizer.decode_probabilities(state) if relaxed else tokenizer.decode_indices(state)
            field = _decode(tokenizer, planes, grid, chunk_size).reshape((resolution,) * 3)
            status, vertices, faces = extract_mesh(field, bounds)
            relative = f"{name}/{row['id']}.ply"
            row[name], row["mesh_status"][name] = relative, status["status"]
            if status["valid"]:
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
                mesh.compute_vertex_normals()
                mesh = clean_components(mesh, min_fraction, min_triangles)
                if not o3d.io.write_triangle_mesh(str(root / relative), mesh):
                    raise OSError("Mesh export failed")
        rows.append(row)
        if (position + 1) % 32 == 0:
            print(f"Decoded {position + 1}/{len(selected)} token fields", flush=True)
    masks = np.stack(masks)
    np.save(root / "masks.npy", masks)
    report = {"format": "orbit_folding.decoded_set.v1", "modes": modes, "samples": rows,
        "tokenizer_hash": identity, "source": provenance, "prior_bindings_verified": bindings,
        "prior_sha256": {kind: file_sha256(path) for kind, path in (("bit_field", bit_prior), ("categorical", categorical_prior)) if path},
        "protocol": {"resolution": resolution, "bounds": list(bounds), "ratio": ratio, "seed": seed,
            "selection_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
            "mask_sha256": file_sha256(root / "masks.npy"), "supplied_mask_sha256": file_sha256(mask_npy) if mask_npy else None,
            "mask_rule": "Supplied mask" if mask_npy else "CPU Bernoulli; seed + round(ratio * 10000)",
            "component_min_fraction": min_fraction, "component_min_triangles": min_triangles},
        "scope": "Frozen decoder comparison, not ground-truth reconstruction or unconditional generation. Missing surfaces stay in the cohort."}
    (root / "decoded_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
