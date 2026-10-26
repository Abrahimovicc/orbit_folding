<!--
Geometric Orbit Representatives for Unaligned 3D Tokenization
Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
SPDX-License-Identifier: MIT
-->

# Method

[Back to Main README](../README.md)

## Geometry Is Fixed; Representation Is Learned

The representative rule is geometric, not a learned pose predictor. Given an
observed point cloud, `representative_frame` centers it to compute covariance,
orders its PCA eigenvectors by decreasing eigenvalue, and corrects handedness.
It then scores the 24 proper signed axis permutations using fixed projection
probes. Both the eigensystem and scoring use float64. The selected rotation is
proper, so the representative is a rotated copy of the observation.

The learned modules are a local point encoder, a reconstructive spatial LFQ
tokenizer, and a shared plane-aware signed-distance decoder. No learned pose
network, alternate legacy decoder, variational bottleneck, or mesh refiner is
part of this model.

## Observations and Frames

Points are row vectors. For a frame `F`, representative coordinates are
`points @ F`; observation coordinates are recovered with `representative @ F.T`.
Centering selects the frame but is not an additional translation applied by
the model. The preprocessing contract supplies object-centered coordinates.
New raw-input preprocessing uses rotation-equivariant centroid/radius
normalization; the explicit `legacy-bbox` option preserves the earlier input
convention. Research NPZ import preserves its existing centering. These are
input protocols, not changes to the learned weights or representative rule.

Compute and cache the frame from **all points in that observation before
encoder subsampling**, not from a semantic template, a global reference object,
or a randomly selected encoder subset. Two different observations carry their
own independently estimated frames. A rotated copy of the same point set
should select the same representative when its geometry identifies the frame.
The tests estimate both frames independently for this check.

For the synthetic training partner, sample a uniform rotation `R`, form
`rotated_points = points @ R.T`, and propagate the clean observation's cached
frame as `rotated_frame = R @ F`. Consequently,
`rotated_points @ rotated_frame` selects the same orbit element. This known
synthetic transformation is not a pose annotation. Independent evaluation
observations must estimate their frames from their own geometry instead.

Tied PCA eigenvalues or tied probe scores do not identify a unique orientation.
Near-symmetry, noise, changed sampling, and partial observations can therefore
change the selected frame. A one-point cloud can have a valid proper frame
without an identifiable or rotation-stable orientation. Nonfloating, nonfinite,
empty, and incorrectly shaped point tensors are rejected explicitly.

## Exact Axis-Lock Rule

The normalized probe vectors, in order, are `(1,2,3)`, `(-2,1,0.5)`,
`(0.5,-1.5,2.5)`, `(1.75,0.25,-1.25)`, `(-1.25,-2.25,0.75)`, and
`(2.5,-0.75,-1.5)`. Their stored precision matches the original implementation.
For each candidate, project the centered representative onto every probe. With
`k = min(16,N)`, each probe contributes

```text
term = mean(top_k) - mean(bottom_k)/8 + mean(projection^3)/32 + mean(projection^5)/128
score = 0; for each probe in order: score = 16*score + term
```

Select the maximum score. With `V` the proper PCA frame and `A` the selected
matrix from the implementation's rotation bank, the returned row-vector frame
is `V @ A.T`. All 24 candidates are scored together. The rule is shared across
categories and does not choose a human-defined upright or front direction.

## Units and Data Contract

The model consumes already-normalized model-space tensors with object radius
approximately `0.5`. The dataset/import pipeline is responsible for applying
the configured input scale exactly once. `ReconstructiveTokenizer` does not
apply `InputScaleFactor` again. If coordinates are divided by a scale factor,
SDF distances must be divided by the same factor. Queries, SDF values, shell
offsets, truncation thresholds, and free-space margins use the same units.
Rotations preserve signed distance and rotate normals without scaling them.

`forward(batch, step)` expects `points`, `query_points`, `query_sdf`, and a cached
`frame`. Optional `normals` provide shell supervision; optional `uniform_points`
and `uniform_sdf` provide field-wide supervision. `category` is a list of
strings for the training auxiliary only. The data pipeline repairs normal
orientation before these normals are used as signed shell anchors.

## Learned Architecture

`configs/tokenizer_paper.yaml` records the selected model and active objective.
The small CPU configuration uses the same design with reduced widths and grids.

| Component | Paper Configuration |
| --- | --- |
| Point input | Seven coordinate features: xyz, xyz squared, and radius |
| Local encoder | Hidden width 128; three relative-feature/position aggregation blocks; 12 neighbors; 2048 reference points |
| Tri-planes | xz, xy, yz; 256 channels each; 128 by 128 cells; normalized bilinear splatting and three shared refinement blocks; output scale 0.025 |
| Sampling | Bilinear `grid_sample`, border padding, `align_corners=False`, coordinate padding 0.1; concatenate three query states |
| Reconstructive tokenizer | 768 to 512 channels; two residual blocks; 256-dimensional latent with GroupNorm; matching plane decoder |
| LFQ | Installed library; one 4096-state, 12-bit hypercube; learned 256-to-12 and 12-to-256 projections; soft clamp 5; straight-through signs |
| Plane-aware field | Shared LayerNorm/plane projection to width 128; coordinate-conditioned gate across planes; residual concatenation; xyz, xyz squared, and radius |
| SDF head | Four hidden layers of width 384; Softplus beta 20; scalar output scaled by 0.1 |

The local encoder is PointNeXt-like; it is not a dependency on the official
PointNeXt package. LFQ is provided by `vector-quantize-pytorch`, not newly
implemented or vendored here. The separate vocabulary-support study should not
be conflated with this completion tokenizer's 12-bit vocabulary.

## Reconstruction and Auxiliary Losses

The raw encoder planes, continuous tokenizer planes, and quantized tokenizer
planes all use the **same** field head. The continuous path decodes the
pre-quantization 256-dimensional latent; the quantized path decodes LFQ output.
They are not interchangeable with relaxed bit probabilities.

The default active objective reconstructs the clean representative. Signed SDF
regression uses a truncated target, Smooth L1 beta 0.003, near-surface weights,
and per-shape inside/outside balancing. Sign supervision uses balanced BCE on
`-predicted_sdf / temperature`, excluding the numerical surface band. Exterior
queries receive a squared positive-distance margin. Shell samples enforce zero
distance on the surface and signed distances at plus/minus normal offsets.
Uniform queries use the same signed field losses. Exact weights and thresholds
remain in the configuration rather than being replaced with a simplified loss.

The tokenizer input and feature-reconstruction targets are detached from the
raw encoder when configured. Field distillation detaches raw predictions as
targets. The encoder still learns through raw reconstruction and consistency;
the shared field head receives gradients through all reconstruction paths.
LFQ adds its configured entropy/diversity and commitment auxiliary terms.

The consistency term compares representative feature fields. Only Table
(`04379243`) and Lamp (`03636649`) labels select finite auxiliary yaw sets, of
orders four and eight respectively. This feature-closest yaw matching is a
training convention, not a pose label or inference input. The frame selector,
encoder, tokenizer, decoder, and completion prior do not require category labels
at inference. Geometric Chamfer and covariance auxiliaries check the fixed
representative; they do not train an otherwise hidden frame predictor.

Curricula use optimizer steps, not batches. All three reconstruction paths are
evaluated even at zero curriculum weight; zero-weight terms remain in the
autograd graph. This avoids unused parameters under DDP. Training-mode LFQ may
synchronize entropy statistics across ranks, whereas evaluation-mode LFQ
returns zero auxiliary loss without those entropy collectives. Use `.eval()`
for validation and token caching, and restore the previous mode afterward.

## LFQ State Interface

`encode(points, frame=None)` returns `rotation`, `representative`, `raw_planes`,
`continuous_planes`, `quantized_planes`, spatial integer `indices`, MSB-first
`bits`, and `aux_loss`. Planes have shape `(B,3C,H,W)`, indices `(B,H,W)`, and
bits `(B,D,H,W)`, where `D=log2(K)`. Omitting the frame estimates one from the
supplied cloud; callers using subsampled inputs should supply the full-cloud
frame instead. `decode(planes, queries_in_representative)` returns `(B,Q)` SDFs.
The data manifest records whether its cached frame uses the axis lock. A
configured policy mismatch triggers full-observation frame estimation before
subsampling, rather than silently reusing a locked frame for a PCA-only run.

`decode_indices` restores hard LFQ states. `decode_probabilities` maps
channel-first probabilities through `2*p-1`, the same LFQ output projection,
and the same plane decoder. Binary probabilities recover hard decoding exactly
in the CPU tests. Interior probabilities retain a differentiable path to the
predicted bits; this does not guarantee good geometry at arbitrary uncertain
states or severe missing-observation ratios.
Index decoding requires integer values in `[0,K)`; negative, fractional and
out-of-vocabulary indices are rejected. Probability decoding requires finite
values in `[0,1]`, without silently clipping invalid inputs.

## Scope

The contract is for normalized **object observations** and rotations of a
fixed observed geometry. It is not a global scene coordinate system, a
cross-view scene registration method, a semantic pose estimator, or a guarantee
that differently occluded scans have identical representatives. Tensor tests
exercise correctness and gradients; they do not establish reconstruction
quality, robustness to every symmetry, or reproduce paper-level results.

[Back to Main README](../README.md#overview)
