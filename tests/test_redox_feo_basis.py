from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
from numbers import Real
from pathlib import Path

import pytest

from openimcc import RedoxInputError, evaluate_redox
from openimcc.receipt import build_calculation_receipt


ROOT = Path(__file__).resolve().parents[1]
MELT_V1_PATH = ROOT / "src/openimcc/data/packs/imcc-sf04-d066-v1.json"


def _numeric_signature(value: object) -> str:
    entries: list[tuple[str, str, str]] = []

    def collect(member: object, path: str) -> None:
        if isinstance(member, bool):
            return
        if isinstance(member, Real):
            kind, encoded = (
                ("int", str(member))
                if isinstance(member, int)
                else ("float", float(member).hex())
            )
            entries.append((path, kind, encoded))
        elif is_dataclass(member):
            for item in fields(member):
                collect(getattr(member, item.name), f"{path}.{item.name}")
        elif isinstance(member, Mapping):
            for key in sorted(member, key=repr):
                collect(member[key], f"{path}[{key!r}]")
        elif isinstance(member, (tuple, list)):
            for index, item in enumerate(member):
                collect(item, f"{path}[{index}]")

    collect(value, "result")
    encoded = json.dumps(entries, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode()).hexdigest()


def test_default_pair_has_verified_basis_and_bit_identical_numeric_result() -> None:
    inventory = {"Fe": 1.0, "O": 1.25}
    result = evaluate_redox(inventory, 2200.0, "closed")

    assert result.feo_liquid_basis == {
        "id": "DG46-MASSON65",
        "delta_g_j_mol": "27778.965828 - 10.961554232*T",
        "relative_to": "JANAF Fe-019",
    }
    # Captured from the pre-change default solve; this signature covers every
    # numeric value in RedoxResult, nested ledgers, and its unchanged receipt.
    assert _numeric_signature(result) == (
        "c503960f5ff3f4ad1bf83ded06ed61f6d5334137e589e5efcfd2f5370f3b2c93"
    )

    expected_receipt = build_calculation_receipt(
        request={
            "inventory": inventory,
            "T_K": 2200.0,
            "mode": "closed",
            "lambda_imposed": None,
            "warm_start": None,
            "profile": None,
            "options": {
                "tol": 1.0e-12,
                "max_iter": 100,
                "redox_pack_path": None,
                "melt_pack_path": None,
            },
        },
        data_digests={
            "melt": result.melt_pack_binding_digest,
            "redox": result.redox_pack_digest,
        },
        solver={"tol": 1.0e-12, "max_iter": 100},
    )
    assert result.receipt is not None
    assert result.receipt.to_dict() == expected_receipt.to_dict()
    assert "feo_liquid_basis" not in result.receipt.to_dict()


def test_janaf_melt_pack_and_dg46_redox_pack_refuse_mixed_bases() -> None:
    with pytest.raises(RedoxInputError) as refused:
        evaluate_redox(
            {"Fe": 1.0, "O": 1.25},
            2200.0,
            "closed",
            melt_pack_path=MELT_V1_PATH,
        )

    message = str(refused.value)
    assert "imcc-sf04-d066-v1.json" in message
    assert "Fe-019" in message
    assert "openimcc-redox-v6.json" in message
    assert "DG46-MASSON65" in message
