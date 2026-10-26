# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Supersampled paper-style point rendering with explicit provenance."""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
import torch

from .evaluation import (_options, _positive, _rotation_pair, load_observation,
                         observation_entries)
from .geometry import representative_frame

# Figure 4: deep paper blue, muted copper, jade, with pale point halos.
COLORS = np.array([[0.12, 0.33, 0.66], [0.74, 0.36, 0.12], [0.02, 0.50, 0.46]])
HALOS = np.array([[0.78, 0.86, 0.96], [0.97, 0.80, 0.64], [0.74, 0.90, 0.87]])
BACKGROUND = (250, 249, 245)
VIEW = np.array([1.55, -2.05, 1.05], dtype=float)
VIEW /= np.linalg.norm(VIEW)
RIGHT = np.cross(-VIEW, [0.0, 0.0, 1.0])
RIGHT /= np.linalg.norm(RIGHT)
UP = np.cross(RIGHT, -VIEW)


def _turn(cloud, angle):
    c, s = np.cos(angle), np.sin(angle)
    return cloud @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T


def _font(size, bold=False):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


class _PaperRenderer:
    def __init__(self, cell, supersample, backend, bound):
        self.cell, self.supersample = cell, supersample
        self.size = cell * supersample
        self.distance = bound * 4.5
        self.renderer = None
        if backend == "open3d":
            os.environ.setdefault("EGL_PLATFORM", "surfaceless")
            from open3d.visualization import rendering

            self.renderer = rendering.OffscreenRenderer(self.size, self.size)
            self.renderer.scene.set_background([*(np.array(BACKGROUND) / 255), 1.0])
            self.renderer.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, [0, 0, 0])

    def _projection(self, cloud):
        depth = cloud @ VIEW
        scale = self.size / (2 * np.tan(np.deg2rad(15))) / (self.distance - depth)
        return np.column_stack([self.size / 2 + (cloud @ RIGHT) * scale,
                                self.size / 2 - (cloud @ UP) * scale, depth])

    def panel(self, layers, angle):
        from PIL import Image, ImageDraw

        layers = [(_turn(points, angle), color) for points, color in layers]
        if self.renderer is not None:
            import open3d as o3d
            from open3d.visualization import rendering

            scene = self.renderer.scene
            scene.clear_geometry()
            for index, (points, color) in enumerate(layers):
                if len(points) == 0:
                    continue
                for halo in (True, False):
                    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
                    cloud.paint_uniform_color(HALOS[color] if halo else COLORS[color])
                    material = rendering.MaterialRecord()
                    material.shader = "defaultUnlit"
                    material.point_size = 0.92 * self.supersample * (1.42 if halo else 1)
                    scene.add_geometry(f"{index}_{halo}", cloud, material)
            self.renderer.setup_camera(30.0, np.zeros(3), VIEW * self.distance, [0, 0, 1])
            result = Image.fromarray(np.asarray(self.renderer.render_to_image())[..., :3]).convert("RGB")
        else:
            result = Image.new("RGB", (self.size, self.size), BACKGROUND)
            draw = ImageDraw.Draw(result)
            projected = np.concatenate([self._projection(points) for points, _ in layers])
            colors = np.concatenate([np.repeat(color, len(points)) for points, color in layers])
            for index in projected[:, 2].argsort():
                x, y, _ = projected[index]
                for radius, palette in ((0.66, HALOS), (0.46, COLORS)):
                    r = radius * self.supersample
                    color = tuple(np.rint(255 * palette[colors[index]]).astype(int))
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        # Actual projected world axes are composited last, in the foreground.
        draw = ImageDraw.Draw(result)
        origin, *ends = self._projection(np.vstack([np.zeros(3), np.eye(3) * self.distance * 0.045]))
        for end, color in zip(ends, [(199, 54, 49), (38, 153, 78), (45, 82, 180)]):
            draw.line((*origin[:2], *end[:2]), fill=color, width=max(1, round(self.supersample * 1.4)))
            x, y = end[:2]
            r = self.supersample * 1.9
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        return result.resize((self.cell, self.cell), Image.Resampling.LANCZOS)

    def close(self):
        if self.renderer is not None:
            self.renderer.scene.clear_geometry()
            self.renderer = None


@torch.no_grad()
def render_alignment(data_root, output, sample_id=None, frames=48, points=8192, **kwargs):
    """Render A/B, complementary folded halves, and an independent reference.

    Open3D uses Figure 4's point/halo materials and 30-degree camera. Software
    is a portable CPU alternative. Neither performs fitted registration.
    Frames see the full observation before display sampling. The folded panel
    is a canonical-x half-composite, not two differently registered surfaces.
    """
    from PIL import Image, ImageDraw

    options = _options(kwargs, dict(seed=2026, split="all", mesh_points=16384, jitter=0.0,
        lock=True, width=1440, height=560, fps=12, max_bytes=16000000,
        renderer="software", supersample=2))
    frames, points = _positive(frames, "frames", 2), _positive(points, "points")
    width, height = _positive(options["width"], "width", 480), _positive(options["height"], "height", 240)
    max_bytes = _positive(options["max_bytes"], "max_bytes", 1024)
    fps, ss = _positive(options["fps"], "fps"), _positive(options["supersample"], "supersample")
    options["mesh_points"] = _positive(options["mesh_points"], "mesh_points", 3)
    if options["renderer"] not in {"software", "open3d"}:
        raise ValueError("renderer must be software or open3d")
    jitter = float(options["jitter"])
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be finite and nonnegative in model-space units")
    entries = observation_entries(data_root, options["split"])
    entry = entries[0] if sample_id is None else next((entry for entry in entries if entry["id"] == sample_id), None)
    if entry is None:
        raise ValueError(f"No real observation with sample_id={sample_id!r}")
    reference = load_observation(data_root, entry, options["mesh_points"], options["seed"])
    _, observed, _ = _rotation_pair(reference, jitter, options["seed"])
    folded = observed @ representative_frame(observed, lock=bool(options["lock"]))
    ref_rep = reference @ representative_frame(reference, lock=bool(options["lock"]))
    selection = np.random.default_rng(options["seed"]).choice(len(reference), min(points, len(reference)), replace=False)
    a, b = observed[:, selection].numpy()
    fa, fb = folded[:, selection].numpy()
    ref = ref_rep[selection].numpy()
    threshold = float(np.median(ref[:, 0]))
    # Take each spatial half from its own independently estimated frame.
    half_a, half_b = fa[:, 0] <= threshold, fb[:, 0] > threshold
    panels = [([(a, 0)], "A input"), ([(b, 1)], "B input"),
              ([(fa[half_a], 0), (fb[half_b], 1)], "Folded A / B"), ([(ref, 2)], "Reference rep.")]
    bound = max(float(np.linalg.norm(cloud, axis=-1).max()) for cloud in (a, b, fa, fb, ref))
    if bound <= 0:
        raise ValueError("Cannot render a zero-radius observation")
    margin, gap = max(10, width // 80), max(4, width // 240)
    panel_width = (width - margin * 2 - gap * 3) // 4
    cell = min(panel_width, height - 110)
    font_size = max(11, min(22, width // 85))
    font, heading = _font(font_size), _font(font_size + 3, True)
    renderer = _PaperRenderer(cell, ss, options["renderer"], bound)
    images = []
    try:
        for index in range(frames):
            canvas = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(canvas)
            title = entry["id"].replace("_", " ").title() + " | Orbit representative consistency"
            while len(title) > 1 and draw.textlength(title, font=heading) > width - margin * 2:
                title = title[:-1]
            draw.text((margin, 16), title, font=heading, fill=(24, 32, 42))
            top = 70 + max(0, height - 110 - cell) // 2
            for column, (layers, label) in enumerate(panels):
                left = margin + column * (panel_width + gap)
                center = left + panel_width / 2
                draw.text((center, top - 27), label, anchor="mt", font=font, fill=(55, 62, 72))
                panel = renderer.panel(layers, index * 2 * np.pi / frames)
                canvas.paste(panel, (int(center - cell / 2), top))
            footer = "Blue: A   Copper: B   Jade: reference   |   Folded panel: spatial halves, no fitted alignment."
            footer_font = _font(max(7, font_size - 2))
            if draw.textlength(footer, font=footer_font) > width - 2 * margin:
                footer = "A: blue | B: copper | Reference: jade | No fitted alignment."
            draw.text((margin, height - 29), footer, font=footer_font, fill=(76, 84, 98))
            images.append(canvas)
            if options["renderer"] == "open3d" and (index == 0 or (index + 1) % 12 == 0):
                print(f"Render: {index + 1}/{frames} frames at {width}x{height}, {len(selection)} points", flush=True)
    finally:
        renderer.close()
    target = Path(output)
    if target.suffix.lower() in {".gif", ".png"}:
        gif_path, png_path, metadata_path = (target.with_suffix(suffix) for suffix in (".gif", ".png", ".json"))
    else:
        gif_path, png_path, metadata_path = (target / name for name in ("alignment.gif", "alignment.png", "alignment.json"))
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(png_path)
    selected, duration = images, max(10, round(1000 / fps))
    # A shared palette preserves halos and avoids frame-to-frame color flicker.
    palette = images[0].quantize(colors=128, dither=Image.Dither.NONE)
    while True:
        buffer = BytesIO()
        palette_frames = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in selected]
        palette_frames[0].save(buffer, format="GIF", save_all=True, append_images=palette_frames[1:],
                               duration=duration, loop=0, optimize=True, disposal=2)
        encoded = buffer.getvalue()
        if len(encoded) < max_bytes:
            break
        # Preserve spatial detail before temporal density.
        if len(selected) > 12:
            selected, duration = selected[::2], duration * 2
        elif selected[0].width > 480:
            selected = [image.resize((int(image.width * 0.8), int(image.height * 0.8)), Image.Resampling.LANCZOS)
                        for image in selected]
        elif len(selected) > 2:
            selected, duration = selected[::2], duration * 2
        else:
            raise ValueError("GIF byte budget is too small; increase max_bytes")
    gif_path.write_bytes(encoded)
    result = {"png": str(png_path), "gif": str(gif_path), "metadata": str(metadata_path),
        "sample_id": entry["id"], "source_path": entry["path"], "source_url": entry.get("source_url"),
        "license_spdx": entry.get("license_spdx"), "authors": entry.get("authors", []),
        "source_sha256": entry.get("sha256", entry.get("source_sha256")), "frames": len(selected),
        "gif_bytes": len(encoded), "gif_dimensions": list(selected[0].size), "requested_frames": frames,
        "seed": int(options["seed"]), "display_points": len(selection), "full_observation_points": len(reference),
        "renderer": options["renderer"], "supersample": ss, "palette": COLORS.tolist(),
        "jitter_sigma_model_units": jitter, "jitter_clip_bounds_model_units": [-3 * jitter, 3 * jitter],
        "folded_pointwise_max_abs_model_units": float((folded[0] - folded[1]).abs().max()),
        "overlay_markers": "Complementary canonical-x halves: A blue, B copper. No fitted display alignment.",
        "camera": "Shared fixed-scale perspective camera, 30-degree FOV; full 360-degree z orbit",
        "caveat": "Real licensed sampled geometry. Geometric repeatability, not learned reconstruction or generation."}
    metadata_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
