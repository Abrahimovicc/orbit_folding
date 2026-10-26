<!-- Geometric Orbit Representatives for Unaligned 3D Tokenization -->
<!-- Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig -->
<!-- Contact: Xiang Chen <xiang.chen@uni-leipzig.de> | SPDX-License-Identifier: MIT -->

# Third-Party Notices

[Back to Main README](README.md)

The root MIT license covers original project code and documentation. It does
not relicense third-party data, documentation, library dependencies or trademarks.
The inherited research-code MIT notice is preserved in
[licenses/research-lineage-MIT.txt](licenses/research-lineage-MIT.txt).

## Example Models

The four models in `data/samples/raw` are derived from existing models distributed
by KhronosGroup/glTF-Sample-Assets. Their original licenses, creators, source
revisions, checksums and geometry-only modifications are documented in
[data/attribution.md](data/attribution.md) and [data/manifest.json](data/manifest.json).
Full CC0 and CC-BY license texts and unmodified upstream model notices are
retained under `data/licenses/`. Preserve those notices on redistribution.
Fox's mixed components are not a choice between alternative licenses.

## Software Dependencies

LFQ and standard VQ use the separately installed `vector-quantize-pytorch` package by
Phil Wang (lucidrains), distributed under MIT. No vendored copy is included.
Its full notice is retained in
[licenses/vector-quantize-pytorch-MIT.txt](licenses/vector-quantize-pytorch-MIT.txt)
for provenance.
The other separately installed dependencies, including PyTorch, Open3D, NumPy,
SciPy, scikit-image, Matplotlib, Pillow, einops and PyYAML, retain their own
copyright and licensing terms in their distributions.

The PointNeXt-style local encoder is the project's compact implementation of
local aggregation, not the official PointNeXt model or a packaged pretrained
PointNeXt network. All core architecture and objective modules were refactored
from this project's research implementation; unrelated external generators,
model repositories, and historical experiments are not bundled.

The animation is rendered from the licensed Fox geometry. Its data attribution
continues to apply to the derived visualization. No institutional logo or
third-party trademark is licensed by the code's MIT grant.

Affiliation logos are unmodified institutional marks, used for attribution.
Their source links and ownership scope are in [assets/logos/](assets/logos/README.md).

[Back to Main README](README.md#license-and-contact)
