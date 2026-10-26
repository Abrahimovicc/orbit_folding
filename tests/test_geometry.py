# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Geometry contracts for actual observed point sets, not a global reference frame."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from orbit_folding.data import GeometryDataset
from orbit_folding.geometry import AXIS_ROTATIONS, random_rotations, representative_frame


@pytest.fixture(scope="module", autouse=True)
def _bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


def _observed_cloud(dtype=torch.float64):
    generator = torch.Generator().manual_seed(173)
    points = torch.randn(96, 3, generator=generator, dtype=dtype)
    points *= torch.tensor([0.18, 0.09, 0.04], dtype=dtype)
    points[:, 0] += points[:, 1].square()
    return points


def _assert_proper(rotation, atol):
    identity = torch.eye(3, dtype=rotation.dtype).expand(rotation.shape)
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, identity, rtol=0, atol=atol)
    torch.testing.assert_close(torch.linalg.det(rotation),
                               torch.ones(rotation.shape[:-2], dtype=rotation.dtype), rtol=0, atol=atol)


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-6), (torch.float64, 2e-14)])
def test_random_rotations_are_reproducible_and_proper(dtype, atol):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        first = random_rotations(32, device="cpu", dtype=dtype)
        torch.manual_seed(91)
        second = random_rotations(32, device="cpu", dtype=dtype)
    assert first.shape == (32, 3, 3)
    assert first.dtype == dtype
    assert torch.equal(first, second)
    _assert_proper(first, atol)


def test_axis_lock_bank_has_exactly_24_proper_signed_permutations():
    assert AXIS_ROTATIONS.shape == (24, 3, 3)
    assert torch.unique(AXIS_ROTATIONS.flatten(1), dim=0).shape[0] == 24
    assert torch.all((AXIS_ROTATIONS == 0) | (AXIS_ROTATIONS.abs() == 1))
    assert torch.all(AXIS_ROTATIONS.abs().sum(dim=1) == 1)
    assert torch.all(AXIS_ROTATIONS.abs().sum(dim=2) == 1)
    _assert_proper(AXIS_ROTATIONS, atol=0)


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 2e-6), (torch.float64, 2e-12)])
def test_independently_estimated_frames_are_stable_on_the_same_orbit(dtype, atol):
    observed = _observed_cloud(dtype)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        rotations = random_rotations(6, device="cpu", dtype=dtype)
    rotated_observations = observed[None] @ rotations.transpose(1, 2)
    # Both estimates use their own full observed cloud. No known pose or cached
    # anchor frame is passed into the independently rotated observation.
    frame = representative_frame(observed)
    independent_frames = representative_frame(rotated_observations)
    expected = (observed @ frame)[None].expand_as(rotated_observations)
    torch.testing.assert_close(rotated_observations @ independent_frames, expected, rtol=0, atol=atol)
    _assert_proper(independent_frames, atol)


@pytest.mark.parametrize("lock", [False, True])
def test_representative_is_a_rotation_of_the_observation_not_a_translation(lock):
    points = _observed_cloud() + torch.tensor([0.11, -0.04, 0.02], dtype=torch.float64)
    frame = representative_frame(points, lock=lock)
    representative = points @ frame
    _assert_proper(frame, 2e-14)
    torch.testing.assert_close(representative @ frame.T, points, rtol=0, atol=2e-14)
    torch.testing.assert_close(representative.norm(dim=-1), points.norm(dim=-1), rtol=0, atol=2e-14)
    torch.testing.assert_close(torch.pdist(representative), torch.pdist(points), rtol=0, atol=2e-14)
    assert representative.mean(dim=0).norm() > 0.01


def test_synthetic_partner_propagates_the_same_full_observation_frame():
    full_observation = _observed_cloud()
    frame = representative_frame(full_observation)
    sampled = full_observation[::3]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        rotation = random_rotations(1, "cpu", torch.float64)[0]
    partner = sampled @ rotation.T
    propagated = rotation @ frame
    torch.testing.assert_close(partner @ propagated, sampled @ frame, rtol=0, atol=2e-14)


def test_dataset_keeps_each_full_observation_frame_when_encoder_points_resample(tmp_path):
    observed = _observed_cloud(torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(62)
        rotation = random_rotations(1, "cpu")[0]
    observations = (observed, observed @ rotation.T)
    frames = []
    manifest = {"schema": "orbit-geometry-v1", "samples": []}
    for index, points in enumerate(observations):
        frame = representative_frame(points)
        frames.append(frame)
        name = f"observation_{index}.npz"
        np.savez(tmp_path / name, points=points.numpy(), normals=np.zeros_like(points.numpy()),
                 frame=frame.numpy(), query_points=points[:24].numpy(),
                 query_sdf=np.linspace(-0.08, 0.08, 24, dtype=np.float32),
                 uniform_points=points[24:48].numpy(),
                 uniform_sdf=np.linspace(-0.1, 0.1, 24, dtype=np.float32))
        manifest["samples"].append({"id": f"observation_{index}", "path": name, "split": "train"})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = GeometryDataset(tmp_path, pc_size=24, samples_per_mesh=16, uniform_samples=8, seed=31)
    for index, expected_frame in enumerate(frames):
        dataset.set_epoch(0)
        first = dataset[index]
        dataset.set_epoch(1)
        second = dataset[index]
        assert not torch.equal(first["points"], second["points"])
        assert first["points"].shape == (24, 3)
        assert torch.equal(first["frame"], expected_frame)
        assert torch.equal(second["frame"], expected_frame)
        matches = (first["points"][:, None] == observations[index][None]).all(dim=-1)
        assert matches.any(dim=1).all()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.uint8, torch.bool])
def test_frame_rejects_nonfloating_points(dtype):
    with pytest.raises(ValueError, match="floating-point"):
        representative_frame(torch.ones(4, 3, dtype=dtype))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_frame_rejects_nonfinite_points(invalid):
    points = _observed_cloud()
    points[2, 1] = invalid
    with pytest.raises(ValueError, match="finite"):
        representative_frame(points)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0, 3), (4, 2), (2, 4, 2)])
def test_frame_rejects_empty_or_invalid_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        representative_frame(torch.zeros(shape))


def test_one_point_has_a_valid_frame_without_claiming_identifiable_orientation():
    point = torch.tensor([[0.1, 0.2, -0.3]], dtype=torch.float64)
    frame = representative_frame(point)
    _assert_proper(frame, atol=0)
    torch.testing.assert_close((point @ frame) @ frame.T, point, rtol=0, atol=0)
