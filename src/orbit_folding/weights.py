# Geometric Orbit Representatives for Unaligned 3D Tokenization
# Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig
# Contact: Xiang Chen <xiang.chen@uni-leipzig.de>
# SPDX-License-Identifier: MIT
"""Prepare and verify matched, checksum-bound weights for a Hugging Face release."""

import json
from pathlib import Path
import shutil

import torch
import yaml

from .checkpoints import convert_original_checkpoint, load_checkpoint, load_model, verify_prior_binding
from .provenance import file_sha256
from .token_cache import tokenizer_hash


WEIGHTS_FORMAT = "orbit_folding.pretrained.v1"


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Weight manifest path escapes its directory")
    return path


def verify_weights(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != WEIGHTS_FORMAT or not manifest.get("pairs"):
        raise ValueError("Unsupported or empty pretrained-weight manifest")
    seen = set()
    for pair in manifest["pairs"]:
        if pair["id"] in seen:
            raise ValueError("Duplicate weight-pair identity")
        seen.add(pair["id"])
        for kind in ("tokenizer", "prior"):
            item = pair[kind]
            if file_sha256(_inside(root, item["path"])) != item["sha256"]:
                raise ValueError(f"Weight checksum mismatch: {pair['id']} / {kind}")
        tokenizer, config = load_model(_inside(root, pair["tokenizer"]["path"]))
        prior, prior_config = load_model(_inside(root, pair["prior"]["path"]))
        if config["Task"] != "tokenizer" or prior_config["Task"] != "prior":
            raise ValueError("Weight roles are reversed")
        if tokenizer_hash(tokenizer) != pair["tokenizer_hash"]:
            raise ValueError("Tokenizer identity differs from the weight manifest")
        if (prior.codebook_size != tokenizer.tokenizer.codebook_size
                or (prior.prior_type == "bit_field" and tokenizer.quantizer_type != "lfq")):
            raise ValueError("Weight-pair vocabularies or quantizer semantics differ")
        verify_prior_binding(prior, pair["tokenizer_hash"])
        if not all(torch.isfinite(value).all() for model in (tokenizer, prior)
                   for value in model.state_dict().values() if value.is_floating_point()):
            raise ValueError("Non-finite model weights")
    return manifest


def prepare_hf_release(sources, output, *, trusted=False):
    """Migrate verified research pairs locally. Never creates or uploads a Hub repo.

    The source spec is private build input; only checksums and public labels are
    exported. A token-cache sidecar must identify the exact tokenizer source,
    and the prior's resolved config must identify that same token cache.
    """
    from .checkpoints import _original_checkpoint

    spec = json.loads(Path(sources).read_text()) if isinstance(sources, (str, Path)) else sources
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Prepare weights into a new, empty directory")
    if not spec.get("pairs"):
        raise ValueError("Source spec needs at least one matched pair")
    manifest = {"format": WEIGHTS_FORMAT, "hosting": "Hugging Face", "published": False, "pairs": []}
    identities = set()
    for pair in spec["pairs"]:
        identity = pair["id"]
        if (not isinstance(identity, str) or not identity or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in identity)
                or identity in identities):
            raise ValueError("Pair IDs must be unique lowercase filename components")
        identities.add(identity)
        source_root = Path(pair["source_root"]).resolve()

        def source(value):
            return (source_root / value).resolve()

        tokenizer_source, prior_source = source(pair["tokenizer_source"]), source(pair["prior_source"])
        token_metadata_path = source(pair["token_metadata"])
        metadata = json.loads(token_metadata_path.read_text())
        if source(metadata["stage1_ckpt"]) != tokenizer_source:
            raise ValueError("Token-cache metadata identifies another tokenizer checkpoint")
        tokenizer_config = yaml.safe_load(source(pair["tokenizer_config"]).read_text())
        prior_config = yaml.safe_load(source(pair["prior_config"]).read_text())
        if source(prior_config["token_path"]) != source(metadata["tokens_path"]):
            raise ValueError("Prior resolved configuration identifies another token cache")
        original_prior = _original_checkpoint(prior_source, trusted=trusted)
        hparams = original_prior.get("hyper_parameters", {})
        if hparams.get("token_path") and source(hparams["token_path"]) != source(metadata["tokens_path"]):
            raise ValueError("Prior checkpoint identifies another token cache")
        tokenizer_config.update(Task="tokenizer", Description="Reference reconstructive tokenizer")
        prior_config.update(Task="prior", Description="Matched masked bit-field prior")
        folder = root / identity
        folder.mkdir(parents=True)
        convert_original_checkpoint(tokenizer_source, tokenizer_config, folder / "tokenizer.pt", "tokenizer", trusted=trusted)
        convert_original_checkpoint(prior_source, prior_config, folder / "prior.pt", "prior", trusted=trusted)
        tokenizer, config = load_model(folder / "tokenizer.pt")
        prior, _ = load_model(folder / "prior.pt")
        digest = tokenizer_hash(tokenizer)
        if (int(metadata["codebook_size"]) != tokenizer.tokenizer.codebook_size
                or prior.num_bits != tokenizer.tokenizer.num_bits):
            raise ValueError("Source cache and migrated model dimensions disagree")
        payload = load_checkpoint(folder / "prior.pt")
        payload["extra"].update(tokenizer_hash=digest,
            binding_provenance={"token_metadata_sha256": file_sha256(token_metadata_path),
                "tokenizer_source_sha256": file_sha256(tokenizer_source),
                "prior_config_sha256": file_sha256(source(pair["prior_config"])),
                "method": "Original token-cache sidecar, prior resolved config and compatible active weights"})
        torch.save(payload, folder / "prior.pt")
        for kind in ("tokenizer", "prior"):
            clean_config = load_checkpoint(folder / f"{kind}.pt")["config"]
            (folder / f"{kind}_config.yaml").write_text(yaml.safe_dump(clean_config, sort_keys=False))
        manifest["pairs"].append({"id": identity, "scope": pair["scope"],
            "codebook_size": tokenizer.tokenizer.codebook_size,
            "num_bits": tokenizer.tokenizer.num_bits, "tokenizer_hash": digest,
            **{kind: {"path": f"{identity}/{kind}.pt", "sha256": file_sha256(folder / f"{kind}.pt"),
                       "bytes": (folder / f"{kind}.pt").stat().st_size} for kind in ("tokenizer", "prior")}})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    project = Path(__file__).resolve().parents[2]
    shutil.copy2(project / "LICENSE", root / "LICENSE")
    shutil.copytree(project / "licenses", root / "licenses")
    lines = ["---", "license: mit", "library_name: pytorch", "tags:", "  - 3d", "  - tokenization", "---", "",
        "# Geometric Orbit Representatives for Unaligned 3D Tokenization", "",
        "Chair of Multimodal Machine Learning, ScaDS.AI Dresden/Leipzig.",
        "Contact: Xiang Chen <xiang.chen@uni-leipzig.de>.", "",
        "Matched reconstructive tokenizers and masked bit-field priors. Frame selection is geometric, not semantic pose prediction.", "",
        "| Pair | Vocabulary | Scope |", "| --- | --- | --- |"]
    lines += [f"| {pair['id']} | {pair['codebook_size']} | {pair['scope']} |" for pair in manifest["pairs"]]
    lines += ["", "## Use", "", "Download a pinned Hub revision, then verify every file before inference:", "",
        "```bash", 'hf download "$HF_REPO_ID" --revision "$HF_REVISION" --local-dir checkpoints/official',
        "orbit verify-weights --directory checkpoints/official", "```", "",
        "The companion code exposes `orbit encode` and `orbit decode`; neither needs SDF labels at inference.",
        "Use `--normalization legacy-bbox` for the earlier raw-input convention, or `--model-space --sampling-frame observation` for already-normalized research observations. The centroid default is a separate rotation-equivariant input protocol, not the original benchmark protocol.",
        "Weights use the original active parameters and a strict tokenizer/prior hash binding. Optimizer states and private source paths are omitted.", "",
        "## Scope", "", "Object-level rotated observations and masked token completion; not unconditional generation or validated raw-scan SDF decoding.",
        "Near-symmetry can make the geometric frame ambiguous. Dataset licenses still govern access to the original training data.", "",
        "## Citation", "", "```bibtex", "@inproceedings{chen2026geometric,",
        "  title={Geometric Orbit Representatives for Unaligned 3D Tokenization},",
        "  author={Chen, Xiang and Mei, Jianhan and Binder, Alexander},",
        "  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},", "  year={2026}", "}", "```", ""]
    (root / "README.md").write_text("\n".join(lines))
    verify_weights(root)
    return manifest
