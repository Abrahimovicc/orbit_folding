# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Small generated fixtures, independent of redistributed meshes and samples."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import json

# Hide accelerators before importing torch, including during test collection.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import torch
import yaml

from orbit_folding.geometry import representative_frame


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def tokenizer_config(project_root):
    return yaml.safe_load((project_root / "configs/tokenizer_demo.yaml").read_text())


@pytest.fixture
def prior_config(project_root):
    return yaml.safe_load((project_root / "configs/prior_demo.yaml").read_text())


@pytest.fixture(scope="session")
def geometry_factory():
    def create(root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        samples = []
        for index, split in enumerate(("train", "train", "train", "train", "val", "test")):
            rng = np.random.default_rng(2026 + index)
            radius = np.float32(0.30 + index * 0.015)

            def directions(count):
                vectors = rng.normal(size=(count, 3))
                return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)

            normals = directions(96)
            points = normals * radius
            offsets = np.repeat(np.array([0, -0.01, 0.01, -0.08, 0.08], dtype=np.float32), 32)
            queries = directions(len(offsets)) * (radius + offsets[:, None])
            uniform = rng.uniform(-0.55, 0.55, (64, 3)).astype(np.float32)
            frame = representative_frame(torch.from_numpy(points)).numpy()
            sample_id = f"generated_{index}"
            filename = f"{sample_id}.npz"
            # Source-space metadata must not cause a second normalization on load.
            np.savez_compressed(
                root / filename, points=points, normals=normals, frame=frame,
                query_points=queries, query_sdf=np.linalg.norm(queries, axis=1) - radius,
                uniform_points=uniform, uniform_sdf=np.linalg.norm(uniform, axis=1) - radius,
                center=np.array([4.0, -2.0, 3.0]), scale=np.float64(7.5),
            )
            samples.append({"id": sample_id, "path": filename, "split": split,
                            "category": "generated_sphere", "radius": float(radius)})
        manifest = {"schema": "orbit-geometry-v1", "units": "model coordinates",
                    "sdf_convention": "negative inside; same distance units as points",
                    "seed": 2026, "samples": samples}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return root

    return create


@pytest.fixture
def geometry_root(tmp_path, geometry_factory):
    return geometry_factory(tmp_path / "geometry")


@pytest.fixture
def rewrite_npz():
    def rewrite(path, **updates):
        with np.load(path, allow_pickle=False) as record:
            arrays = {key: record[key].copy() for key in record.files}
        arrays.update(updates)
        np.savez_compressed(path, **arrays)

    return rewrite


@pytest.fixture(scope="session")
def cpu_env(project_root):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED="2026", PYTHONUNBUFFERED="1")
    # A parent torchrun session must not turn an ordinary subprocess into a rank.
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
                "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        env.pop(key, None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(project_root / "src"), env.get("PYTHONPATH"))))
    return env


@pytest.fixture(scope="session")
def run_python(project_root, cpu_env):
    def run(*arguments, check=True, timeout=120):
        command = [sys.executable, *map(str, arguments)]
        try:
            result = subprocess.run(command, cwd=project_root, env=cpu_env,
                                    capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            pytest.fail(f"CPU subprocess timed out: {command!r}\n{error.stdout}\n{error.stderr}")
        if check and result.returncode:
            pytest.fail(f"CPU subprocess failed ({result.returncode}): {command!r}\n"
                        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        return result

    return run


@pytest.fixture(scope="session")
def run_cli(run_python):
    def run(*arguments, **kwargs):
        return run_python("-m", "orbit_folding", *arguments, **kwargs)

    return run
