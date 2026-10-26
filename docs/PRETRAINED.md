<!-- Geometric Orbit Representatives for Unaligned 3D Tokenization -->
<!-- Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig -->
<!-- Contact: Xiang Chen <xiang.chen@uni-leipzig.de> | SPDX-License-Identifier: MIT -->

# Pretrained Pairs And Reproduction

[Back to Main README](../README.md)

Two matched pairs have been prepared and verified locally. Publication on
Hugging Face is deferred; there is currently no public weight download.
No placeholder URL is presented as a working download. File checksums and
pair identities are recorded in [pretrained_manifest.json](../data/pretrained_manifest.json).
Weights are separate from the source checkout and the four licensed examples.

## Result Mapping

| Measurement | Configuration / Pair | Data And Command | Scope |
| --- | --- | --- | --- |
| Representative repeatability | Fixed geometric rule, no learned weights | `orbit evaluate stability` on your observed point sets | The paper stress protocol uses two rotations of a fixed 2,048-point observation, independent clipped jitter and a common reference. The portable evaluator is diagnostic; match sampling and rotation counts before comparing numbers. |
| Full-shape reconstructive interface | `configs/tokenizer_paper.yaml`; `reconstructive` weights | [Full original train/test IDs](../data/splits/reference_shapes.json); `orbit encode` / `orbit decode` | 4,096 codes. This is the frozen reconstructive branch, not the 8,192-code support experiment. |
| Primary masked completion | `configs/tokenizer_matched.yaml`, `configs/prior_paper.yaml`; `matched_completion` weights | [Matched control IDs](../data/splits/matched_completion.json); [`orbit evaluate tokens`](REPRODUCTION.md#fixed-mask-token-and-state-evaluation) | 4,096 codes, eight categories. A fresh prior evaluation on all 512 original test token maps recovers 97.23% bit accuracy and 78.01% exact tokens. All five mask ratios agree at reported precision. Newly extracted tokens can differ through sampling or numerical effects. |
| VQ/LFQ effective support | Saved `data/paper_token_support.json` | `orbit analyze-support --stats data/paper_token_support.json` | 8,192 codes, 5,216 test shapes. Redraw of actual archived histograms, not training a VQ baseline. |

The split files contain identifiers only, not ShapeNet geometry. Obtain the
source dataset under its terms. The public training engine preserves the active
architecture/objective but adds explicit validation and safe content-bound
resume; it is not a bitwise recreation of the original trainer. The archived
tokenizer splits have no validation partition. To retrain with `best.pt`, define
and record a disjoint validation split; to retain the original train/test IDs,
set `Validation.enabled: false`. Prior weights came from their original
training-cache protocol, which used an internal held-out partition.
See the [reproduction audit](REPRODUCTION.md) for commands, protocol details
and which benchmark results were rerun versus reaggregated from archives.

## Download And Verify

Once the model repository is published, set its actual ID and immutable commit:

```bash
hf download "$HF_REPO_ID" --revision "$HF_REVISION" --local-dir checkpoints/official
orbit verify-weights --directory checkpoints/official
```

Verification checks SHA256 for all weight files, strict parameter compatibility,
bit dimensions and the prior's tokenizer hash. Never mix files from different
pairs. The `reconstructive` pair is the default inference example; use
`matched_completion` only for its stated control protocol.
For local use, place the verified bundle at `checkpoints/official`. The README
pretrained example uses `--normalization legacy-bbox` for the original input
convention. The new centroid default is a separate protocol; see
[input normalization and sampling](INFERENCE.md#input-protocol).

## Release Preparation

[tools/prepare_hf_release.py](../tools/prepare_hf_release.py) reads a private
JSON specification of source checkpoints, resolved configs and original token
metadata. It rejects incompatible source pairings, migrates only active weights,
records provenance hashes, emits model cards/licenses and verifies the result.
It never uploads or creates a Hub repository. Do not commit private source paths.

```bash
python tools/prepare_hf_release.py --sources outputs/pretrained_sources.json \
  --output outputs/hf_release --trust-original
```

`--trust-original` permits pickle only for independently trusted research
checkpoints. Public release checkpoints use restricted loading. After review,
an authorized maintainer may publish the prepared folder with:

```bash
hf upload "$HF_REPO_ID" outputs/hf_release --type model
```

[Back to Main README](../README.md#inference)
