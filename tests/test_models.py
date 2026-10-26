# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Small, self-contained CPU contracts for reconstruction and weight conversion."""

from __future__ import annotations

import copy
import hashlib

import pytest
import torch
import torch.nn.functional as F

from orbit_folding.checkpoints import convert_original_checkpoint, load_checkpoint, load_model
from orbit_folding.geometry import representative_frame
from orbit_folding.models import ReconstructiveTokenizer


@pytest.fixture(scope="module", autouse=True)
def _bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _seed():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        yield


@pytest.fixture
def specs():
    return {
        "Task": "tokenizer", "Canonicalizer": "pca_representative",
        "InputScaleFactor": 2.0, "ShareTokenFieldHead": True,
        "TrainCleanBranchReconstruction": True, "TrainTargetBranchReconstruction": False,
        "SdfModelSpecs": {"latent_dim": 8, "pn_hidden_dim": 16},
        "Encoder": {"type": "pointnext_triplane", "hidden_dim": 16, "num_blocks": 2,
                    "knn_k": 6, "knn_reference_size": 16, "knn_chunk_size": 16,
                    "plane_refine_blocks": 1, "output_scale": 0.025},
        "plane_res": 4,
        "Tokenizer": {"token_dim": 16, "hidden_channels": 16, "codebook_size": 16,
                      "num_res_blocks": 1, "detach_input": True, "entropy_loss_weight": 0.5,
                      "commitment_loss_weight": 0.1, "diversity_gamma": 2,
                      "soft_clamp_input_value": 5.0},
        "FieldDecoder": {"type": "pointnext_plane_aware", "plane_feature_aggregation": "concat",
                         "hidden_dim": 32, "plane_state_dim": 16, "num_layers": 2},
        "SurfaceShell": {"enabled": True, "points": 8, "eps": 0.004,
                         "zero_weight": 1.0, "signed_weight": 1.0},
        "FreeSpace": {"enabled": True, "eps": 0.001, "margin": 0.004,
                      "near_band": 0.04, "near_weight": 2.0},
        "SignLossType": "bce", "EnableSymmetryAwarePose": True,
        "EnableAxisRepresentativeLock": True,
        "SymmetryOrders": {"04379243": 4, "03636649": 8},
        "LossWeights": {
            "raw_sdf": 150, "continuous_sdf": 75, "token_sdf": 150,
            "raw_sign": 150, "continuous_sign": 75, "token_sign": 150,
            "raw_free_space": 20, "continuous_free_space": 10, "token_free_space": 20,
            "feature_continuous": 5, "feature_token": 5, "field_distill": 2, "aux": 1,
            "pose": 10, "canonical_pc": 20, "pca": 0.02, "raw_feature_l2": 0.0001,
            "raw_surface_shell": 25, "continuous_surface_shell": 12.5, "token_surface_shell": 25,
            "raw_uniform_sdf": 30, "continuous_uniform_sdf": 15, "token_uniform_sdf": 30,
            "raw_uniform_sign": 30, "continuous_uniform_sign": 15, "token_uniform_sign": 30,
            "raw_uniform_free_space": 10, "continuous_uniform_free_space": 5,
            "token_uniform_free_space": 10,
        },
    }


@pytest.fixture
def batch():
    # The frame is computed on each full observation before encoder subsampling.
    observed = torch.randn(2, 64, 3) * torch.tensor([0.16, 0.1, 0.06])
    frame = representative_frame(observed)
    points = observed[:, ::2].clone()
    queries = torch.rand(2, 24, 3) * 0.6 - 0.3
    uniform = torch.rand(2, 16, 3) - 0.5
    return {"points": points, "normals": F.normalize(points, dim=-1), "frame": frame,
            "query_points": queries, "query_sdf": queries.norm(dim=-1) - 0.23,
            "uniform_points": uniform, "uniform_sdf": uniform.norm(dim=-1) - 0.23,
            "category": ["04379243", "03636649"]}


def test_encode_contract_uses_cached_frame_without_second_scale(specs, batch):
    model = ReconstructiveTokenizer(specs).eval()
    with torch.no_grad():
        encoded = model.encode(batch["points"], batch["frame"])
    expected_keys = {"rotation", "representative", "raw_planes", "continuous_planes",
                     "quantized_planes", "indices", "bits", "aux_loss"}
    assert set(encoded) == expected_keys
    assert torch.equal(encoded["rotation"], batch["frame"])
    assert torch.equal(encoded["representative"], batch["points"] @ batch["frame"])
    for name in ("raw_planes", "continuous_planes", "quantized_planes"):
        assert encoded[name].shape == (2, 24, 4, 4)
    assert encoded["indices"].shape == (2, 4, 4)
    assert encoded["bits"].shape == (2, 4, 4, 4)
    assert encoded["aux_loss"].shape == ()
    assert encoded["aux_loss"].item() == 0
    assert not hasattr(model, "pose_predictor")
    assert not hasattr(model, "token_field_head")
    assert all(key.startswith(("pointnet.", "tokenizer.", "field_head.")) for key in model.state_dict())


def test_lfq_bits_hard_indices_and_decoded_planes_round_trip_exactly(specs):
    model = ReconstructiveTokenizer(specs).eval()
    indices = torch.arange(16).reshape(1, 4, 4).repeat(2, 1, 1)
    bits = model.tokenizer.indices_to_bits(indices)
    assert torch.all((bits == 0) | (bits == 1))
    weights = torch.tensor([8, 4, 2, 1]).reshape(1, 4, 1, 1)
    recovered_indices = (bits.long() * weights).sum(dim=1)
    assert torch.equal(recovered_indices, indices)
    with torch.no_grad():
        hard = model.decode_indices(indices)
        round_trip = model.decode_indices(recovered_indices)
        binary_relaxation = model.decode_probabilities(bits)
    torch.testing.assert_close(hard, round_trip, rtol=0, atol=0)
    torch.testing.assert_close(hard, binary_relaxation, rtol=0, atol=0)


def test_relaxed_decoding_has_finite_nonzero_probability_gradients(specs):
    model = ReconstructiveTokenizer(specs).eval()
    logits = torch.randn(2, 4, 4, 4, requires_grad=True)
    decoded = model.decode_probabilities(logits.sigmoid())
    decoded.square().mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize("detach", [False, True])
def test_detach_setting_controls_only_the_raw_encoder_input(specs, batch, detach):
    specs["Tokenizer"]["detach_input"] = detach
    model = ReconstructiveTokenizer(specs)
    encoded = model.encode(batch["points"], batch["frame"])
    encoded["quantized_planes"].square().mean().backward()
    encoder_gradients = [parameter.grad for parameter in model.pointnet.parameters()]
    if detach:
        assert all(gradient is None for gradient in encoder_gradients)
    else:
        assert all(gradient is not None for gradient in encoder_gradients)
    assert any(parameter.grad is not None for parameter in model.tokenizer.parameters())


@pytest.mark.parametrize("step", [0, 2500])
def test_every_parameter_participates_even_at_zero_curriculum(specs, batch, step):
    model = ReconstructiveTokenizer(specs)
    loss, metrics = model(batch, step=step)
    assert torch.isfinite(loss)
    loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert not missing, f"Unused parameters break find_unused_parameters=False: {missing}"
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    for name in ("continuous_sdf", "feature", "token_sdf", "field_distill", "aux"):
        assert metrics["schedule/" + name].item() == (0 if step == 0 else 1)


@pytest.mark.parametrize("step", [0, 500, 2500])
def test_small_batched_training_objective_is_finite(specs, batch, step):
    model = ReconstructiveTokenizer(specs)
    loss, metrics = model(batch, step=step)
    assert loss.shape == () and loss.requires_grad
    assert metrics and all(not value.requires_grad for value in metrics.values())
    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert torch.equal(metrics["loss"], loss.detach())
    for branch in ("raw", "continuous", "token"):
        for term in ("sdf", "sign", "free_space", "surface_shell", "uniform_sdf", "uniform_sign",
                     "uniform_free_space"):
            assert "loss/clean_" + branch + "_" + term in metrics


def test_training_objective_requires_the_full_observation_frame(specs, batch):
    del batch["frame"]
    with pytest.raises(ValueError, match="full cloud"):
        ReconstructiveTokenizer(specs)(batch)


def test_auxiliary_yaw_categories_do_not_change_inference(specs, batch):
    model = ReconstructiveTokenizer(specs).eval()
    with torch.no_grad():
        first = model.encode(batch["points"], batch["frame"])
        # Inference has no category argument; auxiliary yaw settings cannot
        # impose semantic poses or change the geometric representative.
        model.specs["SymmetryOrders"] = {"04379243": 1, "03636649": 1}
        second = model.encode(batch["points"], batch["frame"])
    for key in first:
        assert torch.equal(first[key], second[key])


@pytest.mark.parametrize("wrapped", [False, True])
def test_original_weight_conversion_drops_only_documented_unused_prefixes(specs, tmp_path, wrapped):
    model = ReconstructiveTokenizer(specs)
    active = {name: value.detach().clone() for name, value in model.state_dict().items()}
    dropped = {"pose_predictor.conv1.weight": torch.randn(3, 3, 1),
               "token_field_head.sdf_head.0.weight": torch.randn(4, 4)}
    original = dict(active, **dropped)
    payload = {"state_dict": original, "epoch": 2, "global_step": 17} if wrapped else original
    source, output = tmp_path / "original.ckpt", tmp_path / "converted.pt"
    torch.save(payload, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    input_specs = copy.deepcopy(specs)
    input_specs.update(DataSource="data/samples/processed", TrainSplit="data/splits/train.json",
                       TestSplit="data/splits/test.json", CanonicalRotationCache="data/cache/frames.npz")
    converted = convert_original_checkpoint(source, input_specs, output, kind="tokenizer")
    loaded = load_checkpoint(output)
    assert input_specs["DataSource"] == "data/samples/processed"
    assert loaded["extra"]["conversion"]["dropped_keys"] == sorted(dropped)
    assert loaded["extra"]["conversion"]["source_sha256"] == digest
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert set(loaded["state_dict"]) == set(active)
    assert all(torch.equal(value, loaded["state_dict"][name]) for name, value in active.items())
    assert all(key not in loaded["config"] for key in (
        "DataSource", "TrainSplit", "TestSplit", "CanonicalRotationCache"))
    assert loaded["epoch"] == (2 if wrapped else 0)
    assert loaded["step"] == (17 if wrapped else 0)
    restored, restored_specs = load_model(output)
    assert isinstance(restored, ReconstructiveTokenizer)
    assert not restored.training
    assert restored_specs["Task"] == "tokenizer"
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in active.items())
    assert converted["format"] == loaded["format"]


@pytest.mark.parametrize("extra_key", ["unexpected.weight", "pose_predictor_extra.weight",
                                        "token_field_head_extra.weight"])
def test_conversion_rejects_unrecognized_keys(specs, tmp_path, extra_key):
    state = ReconstructiveTokenizer(specs).state_dict()
    state[extra_key] = torch.ones(1)
    source, output = tmp_path / "original.ckpt", tmp_path / "converted.pt"
    torch.save({"state_dict": state}, source)
    with pytest.raises(ValueError, match="unrecognized/used"):
        convert_original_checkpoint(source, specs, output, kind="tokenizer")
    assert not output.exists()


@pytest.mark.parametrize("failure", ["wrong_shape", "missing"])
def test_conversion_rejects_active_tensor_mismatches(specs, tmp_path, failure):
    state = ReconstructiveTokenizer(specs).state_dict()
    name = "pointnet.input.0.weight"
    if failure == "wrong_shape":
        state[name] = state[name][:-1]
    else:
        del state[name]
    source, output = tmp_path / "original.ckpt", tmp_path / "converted.pt"
    torch.save({"state_dict": state}, source)
    with pytest.raises(ValueError, match="Checkpoint mismatch") as error:
        convert_original_checkpoint(source, specs, output, kind="tokenizer")
    assert name in str(error.value)
    assert failure + "=" in str(error.value)
    assert not output.exists()


def test_conversion_does_not_silently_drop_an_unshared_head(specs, tmp_path):
    state = ReconstructiveTokenizer(specs).state_dict()
    state["token_field_head.sdf_head.0.weight"] = torch.ones(1)
    specs["ShareTokenFieldHead"] = False
    source, output = tmp_path / "original.ckpt", tmp_path / "converted.pt"
    torch.save(state, source)
    with pytest.raises(ValueError, match="shares one field head"):
        convert_original_checkpoint(source, specs, output, kind="tokenizer")
    assert not output.exists()
