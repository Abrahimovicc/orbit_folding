<!-- Geometric Orbit Representatives for Unaligned 3D Tokenization -->
<!-- Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig -->
<!-- Contact: Xiang Chen <xiang.chen@uni-leipzig.de> -->
<!-- SPDX-License-Identifier: MIT -->

# Independent Evaluation and Asset Examples

[Back to Main README](../README.md)

Run from the release directory with package dependencies installed. Both entry
scripts use CPU and require no training. Project code is MIT; assets retain
their own licenses and credits in
[`data/attribution.md`](../data/attribution.md) and
[`data/manifest.json`](../data/manifest.json). Generated views are derivatives
of those assets, not MIT replacements for their source licensing.

## Orientation Inspection

```bash
python applications/inspection.py --data-root data --output artifacts/inspection
python applications/inspection.py --data-root data --jitter-unit-radius 0.005 --output artifacts/noisy_inspection
```

This object-orientation repeatability example is not a validated industrial
inspection system. The animation uses the bundled teacup if present, otherwise
the first actual asset: never private paper data or generated geometry. Rotated
inputs A/B (paper blue/copper), their complementary folded halves and a jade
reference representative share a fixed-scale orbiting camera. There is no
fitted display registration. Metadata records the full folded pointwise error.

`render_alignment(data_root, output, sample_id=None, frames=48, points=8192)`
returns paths to `alignment.png`, `alignment.gif`, and source/license metadata.
The default GIF budget is 16,000,000 bytes; any reduction in frame count or
resolution is recorded. `renderer='open3d', supersample=3` uses the paper's
point/halo materials; `renderer='software'` is the portable default.
Use `--render-only` or `--sample-id fox` as needed. Preserve source credits and
licenses when placing the rotating GIF in a README.

`eval_stability(data_root, split='test', output=..., device='cpu', seed=2026,
jitter=0.0, max_samples=None, **kwargs)` estimates frames from two independently
rotated, independently jittered observations. Both frames act on the **same clean
reference**, rotated into the corresponding observation coordinates. It does
not measure Chamfer between two different noisy geometries. The report compares
PCA-only and locked median/p95 `CD x 1e4`, using half the sum of the two mean
**squared** nearest-point distances, matching the paper stress evaluator's
numeric convention. This is not a reproduction of its datasets or full stress
protocol; no symmetry-aware matching is added here.

All public jitter values are per-coordinate Gaussian sigma in radius-0.5 model
coordinates. Paper/unit-radius sigma **0.005 corresponds to model sigma 0.0025**.
`--jitter-unit-radius` performs that division; `--jitter` and the Python API use
model units directly. Independent Gaussian noise is clipped per coordinate to
`+/-3 sigma`: paper sigma 0.005 clips at 0.015, or model sigma 0.0025 at 0.0075.
Reports record both unit conventions and clipping bounds. Processed points/SDF
are not rescaled again; raw assets are radius-normalized for illustration only.

Stability options: `rotation_pairs=8`, `mesh_points=8192`,
`reference_points=1024`, `chunk_size=512`. Frames use the full observed cloud;
only the common metric reference is subsampled. Invalid shapes, empty/nonfinite
or zero-extent inputs are reported as failures. Repeated PCA eigenvalues,
near-symmetry and lock-score ties can make orientation nonunique or unstable.
Small set Chamfer does not imply small orientation-angle error. Raw assets with
no explicit split are **unassigned illustrations**, never a held-out benchmark.

## Asset Token Caching

```bash
python applications/catalog.py --checkpoint checkpoints/tokenizer.pt --data-root data/samples/processed --output artifacts/catalog_tokens
```

Existing processed geometry and a release-format checkpoint are required.
Export uses the checkpoint's point-sampling strategy and representative-frame
policy, preserves sample IDs/categories/splits, and records a tokenizer hash.
Compatible full-cloud frames are reused; a different axis-lock policy
recomputes frames from the full observation. `--pc-size` explicitly overrides
the encoder point budget and is recorded in the cache metadata. LFQ NPZs store
uint8 bits; VQ NPZs store int64 indices. These are not a bit-packed
industrial compression format; reported bytes are actual files.

This avoids re-encoding unchanged assets, not semantic deduplication or a
rotation-invariant cache-key guarantee. Invalidate after geometry, sampling,
normalization, frame-policy, or tokenizer changes. Export records actual source
file fingerprints and encoder-sampling settings; completion evaluation checks
the geometry fingerprint and tokenizer/prior binding before decoding. These
checks do not detect differently encoded copies of the same shape or prove
training-set independence. Preserve original asset licenses/provenance.

## Field Evaluation

The package exposes these exact CLI integration functions:

```python
from orbit_folding.evaluation import eval_reconstruction, eval_completion

eval_reconstruction("checkpoints/tokenizer.pt", "data/samples/processed",
                    output="artifacts/reconstruction.json", resolution=32)
eval_completion("checkpoints/prior.pt", "checkpoints/tokenizer.pt",
                "artifacts/catalog_tokens", "data/samples/processed",
                output="artifacts/completion.json", resolution=32, ratio=0.5)
```

Both default to `split='test'`, `device='cpu'`, and accept `max_samples`.
Completion also accepts the CLI alias `mask_ratio` for `ratio`.
Reconstruction evaluates `raw_planes`, `continuous_planes`, and
`quantized_planes` from `encode(points, frame=frame)`. Queries use the same
effective full-observation frame; grid/mesh comparisons use representative coordinates. SDF remains
in model units.

Common keyword options: `pc_size=None` (checkpoint `PCsize`, otherwise 8192),
`samples_per_mesh=2048`, `uniform_samples=1024`, `query_chunk_size=4096`,
`chunk_size=512`, `mesh_samples=2048`, `seed=2026`, `sign_eps=0.0005`,
`bounds=(-0.5, 0.5)`, `save_meshes=False`. Query and nearest-neighbor work is
chunked. Optional PLY exports use Open3D; extraction uses scikit-image, never
trimesh. Query MAE and sign accuracy are count-pooled, with balanced and uniform
query sets reported separately. Target values within `sign_eps` are sign ties.
Finite/nonfinite, inside/outside, and surface-tie support counts remain visible.
Encoder/query sampling strategies and the axis-lock setting come from the
checkpoint configuration through the same data factory as training and token
export. Explicit evaluation counts override only the point/query budgets.
`options.sampling` records the effective settings and whether frames came from
a matching cache or full-observation recomputation. A PCA-only configuration
does not silently reuse a locked frame.
Completion rejects a known mismatch with the cached frame policy and records
`frame_policy_verified`; legacy caches without that policy remain explicitly
unverified on this check.

Completion uses exactly one seeded whole-cell hidden mask per sample. It
compares hard completion, relaxed probabilities, and the unmasked cached-token
oracle through the **same frozen tokenizer decoder** on identical queries.
Observed bits are checked for preservation. Hidden bit/token accuracies are
count-pooled; a token is correct only if all of its bits match. `ratio=0` has no
hidden accuracy support and returns null, not perfect accuracy. Oracle means the
unmasked tokenizer field, **not** ground-truth shape. Field and grid-gradient
drift are measured against that oracle; finite-difference gradient drift is not
a geometric sharp-edge or grasp-quality score.

Reports retain every failed sample and every no-isosurface/nonfinite-field
failure. Missing meshes have **null Chamfer, never zero**. A valid mesh means
finite, nonempty, nondegenerate extracted triangles; boundary/nonmanifold edges
and watertightness are reported separately. Chamfer compares area-sampled
predicted triangles with stored surface points, only for valid meshes. Valid-only
CD summaries are selection-biased, so always read the mesh-success counts.

Field evaluation requires suitable validated mesh-derived SDF targets. No claim
is made about reliable raw-scan SDF, full geometry from partial views, or grasp
quality. Tiny demos, untrained weights, and GPU overfit outputs are pipeline
checks, not paper-quality results. Splits/hashes do not prove training-set
independence.

[Back to Main README](../README.md#applications)
