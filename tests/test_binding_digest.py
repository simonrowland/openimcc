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
) -> ImccDatapack:
    return label_research_datapack(
        datapack,
        model_id=model_id,
        coverage="RE-research",
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


@pytest.mark.parametrize(
    "changed_field",
    ("A", "B", "nu", "domain", "model_id", "version", "parent_order"),
)
def test_research_binding_digest_changes_for_each_result_field(
    changed_field: str,
) -> None:
    source = load_datapack(BASE_PACK).kernel_datapack
    original = _research_pack(source)

    if changed_field in {"A", "B", "nu"}:
        values = np.array(getattr(source, changed_field), copy=True)
        values.flat[0] += 0.5
        variant = replace(source, **{changed_field: values})
    elif changed_field == "domain":
        domains = list(source.domains)
        domains[0] = (domains[0][0] + 0.5, domains[0][1])
        variant = replace(source, domains=domains)
    elif changed_field == "version":
        variant = replace(source, version=f"{source.version}-changed")
    elif changed_field == "parent_order":
        order = list(reversed(range(source.n_parents)))
        variant = replace(
            source,
            parent_oxides=tuple(source.parent_oxides[index] for index in order),
            nu=source.nu[order, :],
        )
    else:
        variant = source

    labelled = _research_pack(
        variant,
        model_id=(
            f"{RESEARCH_MODEL_ID}-changed"
            if changed_field == "model_id"
            else RESEARCH_MODEL_ID
        ),
    )
    assert original.binding_digest
    assert labelled.binding_digest
    assert labelled.binding_digest != original.binding_digest


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
