<!-- Geometric Orbit Representatives for Unaligned 3D Tokenization -->
<!-- Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig -->
<!-- Contact: Xiang Chen <xiang.chen@uni-leipzig.de> | SPDX-License-Identifier: MIT -->

# Example Asset Attribution

[Back to Main README](../README.md)

Four existing, externally authored model geometries are bundled in
`samples/raw/`. None is a generated surrogate or a procedural replacement.
The PLY files are self-contained and need no texture files or network access.

**License scope:** original project metadata and this attribution document use
the [MIT license](licenses/MIT.txt). Third-party meshes retain their licenses
below; they are **not** relicensed under MIT. Preserve the applicable author,
source, license, warranty, and modification notices when redistributing them.

All model downloads and license checks use the official
[KhronosGroup/glTF-Sample-Assets repository](https://github.com/KhronosGroup/glTF-Sample-Assets),
pinned to commit `8f492390ddc7a2ae221a1369ef560589cf8c70fb` and verified on
2026-10-25. [manifest.json](manifest.json) records immutable download URLs,
source/output SHA-256 checksums, all selected node transforms, authors, full
license-text paths, modifications, and machine-readable geometry checks.
Paths in the manifest are relative to this `data` directory.

## Conversion And Verification

Conversion used Python 3.11.14, trimesh 4.9.0, and Open3D 0.19.0, on CPU only.
Each selected complete object node was transformed by its existing static
scene-node world transform. Only exactly coincident vertex positions were
welded, without a distance tolerance. Output is binary little-endian PLY with
float64 vertex positions and triangle indices. Textures, materials, UVs, vertex
colors, normals, cameras, lights, skinning, and animations were omitted.

There was **no normalization, centering, rescaling, simplification, remeshing,
hole filling, or surface repair**. Source scale and shape of the selected
objects are preserved. Unique transformed coordinates and triangle counts
match the sources; PLY coordinates and indices survive an exact round trip.
Fox is the static unanimated bind-pose geometry, not a sampled animation pose.
Its roughly 154.72-unit maximum extent is intentionally not corrected.

All four exported meshes have one connected component, zero boundary or
nonmanifold edges, no self-intersections, consistent outward winding, positive
signed volume, and `Open3D TriangleMesh.is_watertight() == True`. CPU signed
distance smoke tests returned negative just inside a face and positive just
outside it and beyond the bounding box. These are topology and local SDF
checks, not a claim that downstream normalization or SDF preprocessing has
already run.

| File | Vertices | Triangles | Bytes | Watertight |
| --- | ---: | ---: | ---: | --- |
| `samples/raw/mandarin_orange.ply` | 20,418 | 40,836 | 1,021,351 | Yes |
| `samples/raw/fox.ply` | 290 | 576 | 14,895 | Yes |
| `samples/raw/glass_hurricane_candle_holder.ply` | 1,218 | 2,432 | 61,297 | Yes |
| `samples/raw/teacup.ply` | 8,324 | 16,648 | 416,650 | Yes |

Total geometry size: 1,514,193 bytes (about 1.44 MiB).

## Mandarin Orange

- Source and legal notices: [official model directory](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/MandarinOrange), [retained README](licenses/upstream/Models/MandarinOrange/README.md), [retained model license](licenses/upstream/Models/MandarinOrange/LICENSE.md).
- Original creation: zamdreamer, (c) 2015 zamdreamer. The upstream descriptive prose spells this name `zandreamer`; its legal notices and metadata spell it `zamdreamer`. Both spellings are retained without resolving that upstream inconsistency.
- glTF PBR material: Eric Chadwick, (c) 2023 Wayfair; original material work is credited even though appearance data is omitted.
- Original photogrammetry source identified by Khronos: [Orange on Sketchfab](https://sketchfab.com/3d-models/orange-774bc66d11c24855bf722e35249fddf7).
- SPDX: `CC-BY-4.0`. [Full license text](licenses/CC-BY-4.0.txt); [official license](https://creativecommons.org/licenses/by/4.0/).
- Changes: geometry-only conversion described above; retain the complete `MandarinOrange` node and its source transform. No geometry nodes were omitted.
- Downloaded geometry: [glTF](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/MandarinOrange/glTF/MandarinOrange.gltf) plus [binary buffer](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/MandarinOrange/glTF/MandarinOrange.bin). Texture images were not downloaded.
- Source glTF SHA-256: `148a260776eb8258f1d90c1623bad1bdca17b1bec42da597536ef09e816c2091`.
- Source buffer SHA-256: `15bc58559ab4b60e6e1a94449940911e1c5ebc4c021004b7f4fac54323063cfd`.
- Output PLY SHA-256: `3a1bf51b90326fda29ce7ca9d052e0acbcaa99b3d315553f7bf78892743a4203`.

## Fox

- Source and legal notices: [official model directory](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/Fox), [retained README](licenses/upstream/Models/Fox/README.md), [retained model license](licenses/upstream/Models/Fox/LICENSE.md).
- Model: PixelMannen, 2014, public-domain dedication under `CC0-1.0`.
- Rigging and animation: tomkranis, (c) 2014 tomkranis, `CC-BY-4.0`.
- Conversion to glTF: @AsoboStudio and @scurest, (c) 2017 @AsoboStudio and @scurest, `CC-BY-4.0`.
- Conservative SPDX expression: `CC0-1.0 AND CC-BY-4.0`. [Full CC0 text](licenses/CC0-1.0.txt) and [full CC-BY text](licenses/CC-BY-4.0.txt); [official CC0](https://creativecommons.org/publicdomain/zero/1.0/) and [official CC-BY](https://creativecommons.org/licenses/by/4.0/). These are component licenses, not a choice between two licenses. All credits remain despite removal of rigging and animation.
- Changes: geometry-only conversion described above; retain the complete `fox` node, static bind-pose vertex geometry, and original scale. Skin deformation and animation were not evaluated. No geometry nodes were omitted.
- Downloaded model: [Fox.glb](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/Fox/glTF-Binary/Fox.glb).
- Source GLB SHA-256: `d97044e701822bac5a62696459b27d7b375aada5de8574ed4362edbba94771f7`.
- Output PLY SHA-256: `1df63c8dcf9517d100a89a30053401f7687bb5cc54862554af4c63c1c6c6dd04`.

## Glass Hurricane Candle Holder

- Source and legal notices: [official model directory](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/GlassHurricaneCandleHolder), [retained README](licenses/upstream/Models/GlassHurricaneCandleHolder/README.md), [retained model license](licenses/upstream/Models/GlassHurricaneCandleHolder/LICENSE.md).
- Model and textures: Eric Chadwick, (c) 2021 Wayfair, LLC.
- SPDX for extracted geometry: `CC-BY-4.0`. [Full license text](licenses/CC-BY-4.0.txt); [official license](https://creativecommons.org/licenses/by/4.0/).
- Changes: geometry-only conversion described above; select the complete `GlassHurricaneCandleHolder-glass` node at its original scene transform. Omit the separate `GlassHurricaneCandleHolder-opaque` candle/wick/scene-prop node. The complete glass shell is retained without filling or reshaping.
- Upstream additionally credits Khronos Group for its Khronos logo (2015) and glTF logo (2017). These appear in source appearance data, which is omitted. No logo image, texture, or trademark license is bundled with the PLY. The [full original trademark notice](licenses/upstream/LICENSES/LicenseRef-LegalMark-Khronos.txt) is retained for provenance, not as a grant of trademark rights.
- Downloaded model: [GlassHurricaneCandleHolder.glb](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/GlassHurricaneCandleHolder/glTF-Binary/GlassHurricaneCandleHolder.glb).
- Source GLB SHA-256: `add23791ee6f5f550b94a21232a323720c92c20d280f4f7d19cf954234f0b7b8`.
- Output PLY SHA-256: `bdda31c3e21bcb9a6f49c3bd2d3c74a0a43fe46efc5e21bc8e476d005bf000b8`.

## Diffuse Transmission Teacup

- Source and legal notices: [official model directory](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/DiffuseTransmissionTeacup), [retained README](licenses/upstream/Models/DiffuseTransmissionTeacup/README.md), [retained model license](licenses/upstream/Models/DiffuseTransmissionTeacup/LICENSE.md).
- Models and textures: Polyhaven.com and Eric Chadwick, 2023, public domain.
- SPDX: `CC0-1.0`. [Full license text](licenses/CC0-1.0.txt); [official license](https://creativecommons.org/publicdomain/zero/1.0/).
- Changes: geometry-only conversion described above; select the complete `tea_cup` object node, including its handle, at its original scene transform. Omit the distinct `tea_saucer` node. No cup surface is removed or repaired.
- Downloaded model: [DiffuseTransmissionTeacup.glb](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/DiffuseTransmissionTeacup/glTF-Binary/DiffuseTransmissionTeacup.glb).
- Source GLB SHA-256: `d4f567186fea262819aac6664cb9d7cef7f4600406038046771a69e10363935b`.
- Output PLY SHA-256: `4c52ddd54599e404373105661ddda9e6e5c48e7002915858319a260ba485c6ad`.

## Upstream Documentation And License Texts

The model-specific README, LICENSE, and metadata files are copied verbatim
under `licenses/upstream/Models/`, with their original legal and author
notices. Credit: KhronosGroup/glTF-Sample-Assets contributors. These copied
metadocumentation files are retained under `CC-BY-4.0`, as specified by each
model's LICENSE file. No project header was inserted into third-party files.
The manifest records each document's original URL and SHA-256 checksum.

The complete license texts were downloaded without alteration from their
publishers: [CC0 legal code](https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt)
and [CC-BY legal code](https://creativecommons.org/licenses/by/4.0/legalcode.txt).
They include the applicable as-is/warranty disclaimers. The CC licenses do
not grant trademark or patent rights; no endorsement or broader rights
clearance is implied.

## Selection Notes

The initially proposed Avocado and Water Bottle have open boundaries;
Barramundi Fish has self-intersections. None passed Open3D's strict
`is_watertight()` test after exact-coordinate welding. Duck was excluded
because its own license is Sony's SCEA Shared Source License, not CC0 or
CC-BY. Other complete-scene candidates also failed; their measured results
are recorded in the manifest.

The full candle-holder scene and the full cup-and-saucer scene also fail the
strict test. Their selected complete glass and cup objects pass on their
own; omitted scene nodes are explicitly listed above and in the manifest.
Only the attributed geometry and license records are redistributed here;
the complete original scenes are available from the pinned upstream URLs.

[Back to Main README](../README.md#quick-start)
