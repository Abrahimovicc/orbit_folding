<!--
Geometric Orbit Representatives for Unaligned 3D Tokenization
Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
SPDX-License-Identifier: MIT
-->

# Checkpoints

[Back to Main README](../README.md)

## Portable Format

Release checkpoints are dictionaries, not pickled model instances. They contain
the format marker `orbit_folding.checkpoint.v1`, `model_type`, explicit `config`,
`state_dict`, `epoch`, `step`, optional optimizer/scheduler state dictionaries,
and safe metadata in `extra`. `Task` is either `tokenizer` or `prior` and must
agree with the selected constructor. Loading never takes an import path from
the configuration.

Within those tasks, `Tokenizer.type` selects `lfq` (default) or `standard_vq`;
`Prior.type` selects `bit_field` (default) or `categorical`. The defaults preserve
existing LFQ checkpoints. Original categorical checkpoints need an explicit
`Prior: {type: categorical}` in their conversion config. Model state and all
active shapes are still checked strictly; no architecture is guessed from a
filename. See [baseline presets](BASELINES.md).

Verified pretrained tokenizer/prior pairs and their checksums are documented in
[PRETRAINED.md](PRETRAINED.md). `last.pt` contains the latest resumable state;
`best.pt` contains the best held-out state, not an average of checkpoints.

`save_checkpoint` writes atomically. `load_checkpoint` uses
`torch.load(..., weights_only=True)` and validates the envelope;
`load_model` additionally constructs the allowlisted model and strictly checks
all state names and shapes. An explicit model configuration is necessary:
tensor shapes alone do not encode geometry conventions, feature scale, decoder
aggregation, or loss settings.

## Compatibility Adapter

`convert_original_checkpoint` accepts a tensor state dictionary or a dictionary
containing `state_dict`. Supply the matching configuration explicitly. The
compatibility adapter **preserves active weights**; it does not retrain,
reinterpret mismatched tensors, or initialize missing learned components.

```python
from orbit_folding.checkpoints import convert_original_checkpoint, load_model

convert_original_checkpoint(
    "weights/source.ckpt",
    specs,
    "weights/tokenizer.pt",
    kind="tokenizer",
)
model, configuration = load_model("weights/tokenizer.pt", device="cpu")
```

The tokenizer keeps the trained names `pointnet.*`, `tokenizer.*`, and
`field_head.*`. For the fixed-geometric, shared-field configuration, the only
discardable original prefixes are:

| Prefix | Why It Can Be Removed |
| --- | --- |
| `pose_predictor.*` | The evaluated representative is computed geometrically; this learned predictor is unused |
| `token_field_head.*` | `ShareTokenFieldHead=True` selects `field_head` for every reconstruction path; the duplicate head is unused |

The adapter requires the corresponding configuration conditions. Unshared
heads or learned-frame configurations are not supported by the release model.
Unknown extra keys, missing active tensors, and incorrect active shapes fail
conversion. Similarly named prefixes such as `pose_predictor_extra.*` are not
allowlisted. The caller's configuration and source checkpoint are not modified.

Conversion strips local dataset, split, cache, and output paths from the stored
configuration. Provenance records a source SHA-256 hash, source format, and the
exact list of dropped keys, not the source filename or private filesystem path.
Optimizer/scheduler state is not migrated by the original-weight adapter;
converting a model is not equivalent to resuming the original training run.
Use the converted file with [`--init-checkpoint`](TRAINING.md#initialize-from-weights)
for fresh training, or `load_model` for inference. Never invent optimizer or RNG
state to make a weights-only file appear resumable.

Restricted deserialization is the default. `trusted=True` explicitly permits
an unsafe-pickle fallback for an independently verified original checkpoint.
Do not use it for untrusted files. A restricted-load failure can also indicate
corruption, so enabling pickle is not a generic repair.

## Validation Evidence and Limits

The release's CPU tests build small models and synthetic source checkpoints.
They verify exact preservation of active tensor values, strict removal of only
the documented unused prefixes, and rejection of missing or mismatched active
weights. No test requires a private checkpoint, original repository import,
external dataset, or downloaded example asset.

Separately, a private implementation audit checked the active paper objective
against its original reference on a fixed CPU batch with identical active
weights and supervision. The observed total-loss difference was zero, and no
compared clean-branch loss term differed beyond `atol=1e-6, rtol=1e-5`. A
full-size trained state strictly matched all 153 active tensors after dropping
the unused prefixes. These are compatibility checks, not a new training run or
a public reproduction of the paper's quantitative evaluation. The separate
[reproduction audit](REPRODUCTION.md) now also reports full-size real-data
CUDA comparisons and a new fixed-mask prior evaluation.

The CPU checks also observed zero frame difference against the original
full-cloud geometric implementation on the audited nondegenerate batch, and
zero hard-index versus binary-probability decoding difference after using
consistent contiguous tensor layouts. Relaxed decoding has finite nonzero
gradients. Small floating-point differences can remain across devices, library
versions, layouts, and precision settings; bitwise equality is not a promised
cross-platform numerical contract.

The audited LFQ implementations installed for the reference and release had
identical implementation files despite different package versions. This is
narrow source-level evidence for that LFQ implementation, not a claim that all
package versions, unrelated modules, or future releases are interchangeable.
Use the dependency pin in `pyproject.toml` for the public environment.

## Representation Compatibility

Weights, configuration, geometry conventions, bit ordering, and token caches
form one compatibility unit. Changing the tokenizer invalidates cached tokens
and the completion priors trained on them until those artifacts are regenerated
and evaluated. A different vocabulary size changes the LFQ bit dimension and
projection shapes, so it cannot be fixed by silently dropping checkpoint keys.

Training stores the source tokenizer hash in every prior checkpoint. Completion
evaluation and direct decoding require that binding to match the loaded
tokenizer, even when dimensions agree. For old priors with no binding, the
explicit `--allow-unverified-prior` flag records unverified provenance; it never
overrides a known mismatch. Token exports also record geometry content and
encoder-sampling provenance. Completion evaluation rejects stale geometry;
`--allow-unverified-geometry` permits only missing legacy provenance, never a
known mismatch. These checks establish artifact identity, not training-set
independence or generalization.

Inputs and signed distances are already in model-space units; checkpoints do
not authorize applying the input scale twice. See [Method](METHOD.md) for the
row-vector frame convention, full-observation frame cache, and the distinction
between geometric selection and learned representation modules.

## Third-Party Notices

LFQ remains an external `vector-quantize-pytorch` dependency. Its upstream MIT
notice is `Copyright (c) 2020 Phil Wang`; retain the full upstream notice with
any redistributed source, wheel, or substantial copied portion. No LFQ code is
vendored in the model package. Cite LFQ/MAGVIT-v2 and PointNeXt for their research
ideas; those citations are distinct from software-license notices.

The audited source lineage also includes the MIT notice
`Copyright (c) 2022 Gene Chou`. Retain that full notice for redistributed
derived portions; a new project MIT header does not replace upstream notices.
The release does not import the legacy ConvONet adapter, embedded U-Net,
structural-loss extensions, or legacy SDF decoder. Reintroducing copied code
from those components requires checking and carrying its own upstream notices,
not assuming one top-level license covers every dependency. Example data have
separate attribution and redistribution terms from the model source code.

[Back to Main README](../README.md#training)
