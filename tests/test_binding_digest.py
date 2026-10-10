"""Coverage for the per-pack binding digest, distinct from core integrity."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from openimcc import ImccDatapack, label_research_datapack, load_datapack
import openimcc.kernel as kernel
import openimcc.model as model


PACK_DIR = Path("src/openimcc/data/packs")
BASE_PACK = PACK_DIR / "imcc-sf04-v1.0.2.json"
EXT_PACK = PACK_DIR / "imcc-sf04-ext-v4.json"
RESEARCH_MODEL_ID = "IMCC-SF04-RESEARCH"


def _manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _research_pack(
    datapack: ImccDatapack,
    *,
    model_id: str = RESEARCH_MODEL_ID,
    coverage: str | dict[str, str] = "RE-research",
) -> ImccDatapack:
    return label_research_datapack(
        datapack,
        model_id=model_id,
        coverage=coverage,
    )


def test_published_packs_have_distinct_binding_and_same_core_hash() -> None:
    v1 = load_datapack(BASE_PACK)
    extension = load_datapack(EXT_PACK)
    assert v1.binding_digest
    assert extension.binding_digest
    assert v1.binding_digest != extension.binding_digest

    core_hashes = []
    for path in (BASE_PACK, EXT_PACK):
        data = _manifest(path)
        _, content_hash = model._validate_published_core(
            data,
            model_id=data.get("model_id", "IMCC-SF04"),
            version=data["imcc_sf04_datapack_version"],
        )
        core_hashes.append(content_hash)
    assert core_hashes == [kernel._PUBLISHED_DATAPACK_SHA256] * 2


def _research_sensitivity_variants(
    source: ImccDatapack,
) -> dict[str, tuple[ImccDatapack, str, str | dict[str, str]]]:
    changed_nu = np.array(source.nu, copy=True)
    changed_nu.flat[0] += 0.5
    changed_a = np.array(source.A, copy=True)
    changed_a.flat[0] += 0.5
    changed_b = np.array(source.B, copy=True)
    changed_b.flat[0] += 0.5
    changed_domains = list(source.domains)
    changed_domains[0] = (changed_domains[0][0] + 0.5, changed_domains[0][1])

    variants: dict[str, tuple[ImccDatapack, str, str | dict[str, str]]] = {
        "model_id": (source, f"{RESEARCH_MODEL_ID}-changed", "RE-research"),
        "version": (
            replace(source, version=f"{source.version}-changed"),
            RESEARCH_MODEL_ID,
            "RE-research",
        ),
        # Keep every coefficient array fixed: this isolates parent names/order.
        "parent_oxides": (
            replace(source, parent_oxides=tuple(reversed(source.parent_oxides))),
            RESEARCH_MODEL_ID,
            "RE-research",
        ),
        "complex_names": (
            replace(
                source,
                reactions=(f"{source.reactions[0]}-changed", *source.reactions[1:]),
            ),
            RESEARCH_MODEL_ID,
            "RE-research",
        ),
        "nu": (
            replace(source, nu=changed_nu),
            RESEARCH_MODEL_ID,
            "RE-research",
        ),
        "A": (
            replace(source, A=changed_a),
            RESEARCH_MODEL_ID,
            "RE-research",
        ),
        "B": (
            replace(source, B=changed_b),
            RESEARCH_MODEL_ID,
            "RE-research",
        ),
        "domains": (
            replace(
                source,
                domains=changed_domains,
            ),
            RESEARCH_MODEL_ID,
            "RE-research",
        ),
        "paper_domains": (
            replace(
                source,
                paper_domains=(
                    (source.paper_domains[0][0] + 0.5, source.paper_domains[0][1]),
                    *source.paper_domains[1:],
                ),
            ),
            RESEARCH_MODEL_ID,
            "RE-research",
        ),
        "coverage": (
            source,
            RESEARCH_MODEL_ID,
            "RE-research-changed",
        ),
    }
    # Build the comparison set independently of the mutation list, so adding
    # a payload key requires adding a sensitivity case here.
    payload_keys = set(
        kernel._kernel_datapack_binding_payload(
            source,
            model_id=RESEARCH_MODEL_ID,
            coverage={
                name: "RE-research"
                for name in (*source.parent_oxides, *source.reactions)
            },
        )
    )
    assert set(variants) == payload_keys
    return variants


def test_research_binding_digest_changes_for_every_payload_field() -> None:
    source = load_datapack(BASE_PACK).kernel_datapack
    original = _research_pack(source)
    assert original.binding_digest
    for key, (variant, model_id, coverage) in _research_sensitivity_variants(
        source
    ).items():
        labelled = _research_pack(variant, model_id=model_id, coverage=coverage)
        assert labelled.binding_digest, key
        assert labelled.binding_digest != original.binding_digest, key


def test_research_binding_digest_normalizes_signed_zero() -> None:
    source = load_datapack(BASE_PACK).kernel_datapack
    positive = np.array(source.A, copy=True)
    negative = np.array(source.A, copy=True)
    positive[0] = 0.0
    negative[0] = -0.0

    positive_digest = _research_pack(replace(source, A=positive)).binding_digest
    negative_digest = _research_pack(replace(source, A=negative)).binding_digest

    assert positive_digest
    assert negative_digest == positive_digest


def test_extension_manifest_content_changes_binding_digest(
    tmp_path: Path,
) -> None:
    changed = _manifest(EXT_PACK)
    changed["sp_extension"]["rows"][0]["A"] += 0.5
    altered_path = tmp_path / "changed-extension.json"
    altered_path.write_text(json.dumps(changed), encoding="utf-8")

    original = load_datapack(EXT_PACK)
    altered = load_datapack(altered_path)
    assert altered.binding_digest != original.binding_digest


def test_json_binding_digest_includes_effective_model_id_and_version_suffix(
    tmp_path: Path,
) -> None:
    manifest = _manifest(EXT_PACK)
    expected = model._published_datapack_manifest_hash(
        {
            **manifest,
            "model_id": "IMCC-SF04-EXT",
            "imcc_sf04_datapack_version": manifest["imcc_sf04_datapack_version"],
        }
    )
    loaded = load_datapack(EXT_PACK)
    assert loaded.binding_digest == expected

    changed_id = {**manifest, "model_id": "IMCC-SF04-EXT-OTHER"}
    assert model._published_datapack_manifest_hash(changed_id) != expected

    changed_version = dict(manifest)
    changed_version["imcc_sf04_datapack_version"] = (
        f"{manifest['imcc_sf04_datapack_version']}-changed"
    )
    changed_path = tmp_path / "changed-extension-version.json"
    changed_path.write_text(json.dumps(changed_version), encoding="utf-8")
    assert load_datapack(changed_path).binding_digest != loaded.binding_digest


def _reverse_mapping_keys(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _reverse_mapping_keys(value[key])
            for key in reversed(list(value))
        }
    if isinstance(value, list):
        return [_reverse_mapping_keys(member) for member in value]
    return value


def test_binding_digest_ignores_manifest_key_order_and_whitespace(
    tmp_path: Path,
) -> None:
    manifest = _manifest(BASE_PACK)
    compact_path = tmp_path / "compact.json"
    compact_path.write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )
    reordered_path = tmp_path / "reordered.json"
    reordered_path.write_text(
        json.dumps(_reverse_mapping_keys(manifest), indent=3), encoding="utf-8"
    )

    compact = load_datapack(compact_path)
    reordered = load_datapack(reordered_path)
    assert compact.binding_digest == reordered.binding_digest


def test_packaged_binding_digests_are_stable_across_hash_seeds() -> None:
    repository = Path(__file__).resolve().parents[1]
    code = """
import json
from pathlib import Path
from openimcc import load_datapack
packs = Path('src/openimcc/data/packs')
digests = [
    load_datapack(packs / 'imcc-sf04-v1.0.2.json').binding_digest,
    load_datapack(packs / 'imcc-sf04-ext-v4.json').binding_digest,
    load_datapack(packs / 'imcc-sf04-d066-v1.json').binding_digest,
    load_datapack(packs / 'imcc-sf04-d066-v1-kcaalsi2o7.json').binding_digest,
    load_datapack(packs / 'imcc-sf04-d066-ext-v1.json').binding_digest,
]
print(json.dumps(digests))
"""
    digests_by_seed = []
    for seed in ("1", "98765"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = str(repository / "src")
        output = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repository,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        digests_by_seed.append(json.loads(output))

    assert digests_by_seed[0] == digests_by_seed[1]
