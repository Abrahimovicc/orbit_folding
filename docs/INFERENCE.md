<!-- Geometric Orbit Representatives for Unaligned 3D Tokenization -->
<!-- Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig -->
<!-- Contact: Xiang Chen <xiang.chen@uni-leipzig.de> | SPDX-License-Identifier: MIT -->

# Observation Inference

[Back to Main README](../README.md)

`orbit encode` accepts a mesh, NPY point cloud, or NPZ with `points`. Unlike
training preprocessing, it requires neither watertightness nor SDF targets.
This interface does not establish reliable SDF reconstruction on raw scans.

```bash
orbit encode --checkpoint checkpoints/official/reconstructive/tokenizer.pt \
  --input data/samples/raw/teacup.ply --output outputs/cup.npz
orbit decode --checkpoint checkpoints/official/reconstructive/tokenizer.pt \
  --tokens outputs/cup.npz --output outputs/cup_mesh.ply --resolution 96
```

Meshes use seeded area sampling; point clouds retain their full observed
point set for frame estimation. The default mesh observation has 8,192 points.
The encoder then samples its configured point budget.

## Python Integration

The same file-based interface is available without a CLI subprocess:

```python
from orbit_folding.inference import encode_observation, decode_observation

tokenizer = "checkpoints/official/reconstructive/tokenizer.pt"
encode_observation(tokenizer, "data/samples/raw/teacup.ply", "outputs/cup.npz",
                   device="cuda", normalization="legacy-bbox")
report = decode_observation(tokenizer, "outputs/cup.npz", "outputs/cup_mesh.ply",
                            device="cuda", resolution=96, space="input")
if not report["mesh"]["valid"]:
    raise RuntimeError(report["mesh"]["status"])
```

Use `device="cpu"` for CPU inference. Encoding returns source, normalization
and tokenizer metadata; decoding returns a mesh-status dictionary and writes
the PLY only when extraction succeeds. Invalid inputs, stale weights and output
collisions raise exceptions. Each function also writes a JSON sidecar.
`space="input"` restores the original input coordinates, while
`space="representative"` retains the selected frame.

For masked completion, pass a matched `prior_checkpoint`, `mask_ratio` and
`state="hard"` or `state="relaxed"` to `decode_observation`. SDF labels and
category labels are not required for these observation-inference calls.
For batch asset processing, use the
[catalog example](../applications/README.md#asset-token-caching), which loads
the tokenizer once and records the cache's sampling and source identities.

## Input Protocol

By default, raw inputs are centered at their centroid and divided by twice the
maximum radius. Both operations commute with rotation. Spatial subsampling is
performed in the selected representative frame so its grid does not turn with
the observation axes. Point ordering and the seed are held fixed in the
rotation tests; resampled or differently ordered clouds need not have identical
tokens. Frame ambiguity and floating-point threshold effects still apply.

`--model-space` skips normalization for already normalized inputs. Do not
normalize twice. For the original pretrained input/sampling convention, use:

```bash
orbit encode --checkpoint checkpoints/official/reconstructive/tokenizer.pt \
  --input data/samples/raw/teacup.ply --output outputs/cup_legacy.npz \
  --normalization legacy-bbox
```

`legacy-bbox` uses the axis-aligned bounding-box center and observation-frame
spatial sampling. It preserves the earlier code path but is not rotation
equivariant when independently applied to rotated raw files. For already
normalized research observations, `--model-space --sampling-frame observation`
preserves the original encoder sampling. Normalize a benchmark object once
before applying its test rotations. The new default does not change any
pretrained weights, but it is a different input protocol, not a claim of
bitwise reproduction of the original experiment.

Encoded NPZs store binary CHW bits, frame, center, scale, tokenizer hash and
source provenance, including normalization and sampling-frame choices.
`--space input` restores the input coordinates as
`representative @ frame.T * scale + center`; `--space representative` retains
the selected frame. Centering/scaling and the frame are not semantic pose labels.

## Paper Mesh Export

Decoded fields use the model-space cube `[-0.5, 0.5]`, matching the original
SDF evaluator and token-map decoder. Bounds control query locations, not input
normalization. Use `--bounds -0.6 0.6` to replay the earlier release's wider cube.
The bounds, query chunk size and component-filter settings are stored in the
output JSON.

For the original high-resolution reconstruction-gallery preset:

```bash
orbit decode --checkpoint checkpoints/official/reconstructive/tokenizer.pt \
  --tokens outputs/cup.npz --output outputs/cup_paper.ply \
  --resolution 160 --query-chunk-size 65536 \
  --component-min-fraction 0.02 --component-min-triangles 128 \
  --component-min-extent-fraction 0.08
```

The extent rule retains long thin components even when their triangle count
is small. Filtering is vectorized and does not move vertices or apply smoothing.
All three component thresholds default to zero for raw inference export;
`orbit benchmark decode` retains its original 2%/64-triangle quantitative rule.
Filtering reports both pre-filter and exported topology counts. Presentation
settings are separate from quantitative metrics.

For paper reconstruction CD/EMD and signed-field fidelity, use
[`orbit benchmark fidelity`](BENCHMARKS.md#decoded-fidelity). It estimates the
frame from the subsampled observed cloud, as the paper's reconstruction
evaluator does; this is distinct from the full-observation inference contract.

## Masked Completion

```bash
orbit decode --checkpoint checkpoints/official/reconstructive/tokenizer.pt \
  --prior checkpoints/official/reconstructive/prior.pt \
  --tokens outputs/cup.npz --mask-ratio 0.5 --state relaxed \
  --output outputs/cup_completed.ply
```

`--state hard` thresholds predicted bits; `--state relaxed` keeps probabilities.
Both preserve observed cells. `--state oracle` uses all original encoded bits,
not ground-truth geometry. No `--prior` means oracle reconstruction by default.
The prior must name the exact tokenizer hash, not merely have the same bit count.

All inference outputs require fresh filenames.
Use different stems for the encoded NPZ and decoded PLY; each has a JSON sidecar.
If the field has no valid isosurface, the command writes a JSON failure report, creates no substitute
mesh, and exits with status 2. Successful mesh export does not imply correct
topology, a calibrated SDF, or reliable missing-surface completion.

For a trusted legacy prior with no binding, `--allow-unverified-prior` is an
explicit diagnostic override. Known mismatches always fail; reports distinguish
verified and unverified pairs. Prefer the verified [pretrained bundles](PRETRAINED.md).

[Back to Main README](../README.md#inference)
