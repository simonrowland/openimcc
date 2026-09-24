"""Behavior-focused coverage gates for the IMCC-SF04 kernel."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import openimcc.kernel as kernel
from openimcc import (
    ImccComponentOutsideDomainError,
    ImccCompositionIncompleteError,
    ImccDatapack,
    ImccFerricInputUnsupportedError,
    ImccNonconvergenceError,
    ImccTOutsideDatapackDomainError,
)
from openimcc.kernel import (
    _PUBLISHED_COVERAGE,
    _PUBLISHED_CORE_ROWS,
    _PUBLISHED_DATAPACK_SHA256,
    _PUBLISHED_MODEL_ID,
    _PUBLISHED_PARENT_OXIDES,
    _canonical_published_serialization,
    _datapack_with_identity,
    _solve_active,
    solve_imcc_sf04,
)


def _ab_pack(
    *,
    A: float = 0.0,
    B: float = 0.0,
    domains: list[tuple[float, float]] | None = None,
) -> ImccDatapack:
    return ImccDatapack(
        reactions=("AB",),
        nu=np.array([[1.0], [1.0]]),
        A=np.array([A]),
        B=np.array([B]),
        domains=domains or [(0.0, 1.0e6)],
        version="coverage-ab",
        parent_oxides=("A", "B"),
    )


def _pack_kwargs() -> dict[str, Any]:
    return {
        "reactions": ("AB",),
        "nu": np.array([[1.0], [1.0]]),
        "A": np.array([0.0]),
        "B": np.array([0.0]),
        "domains": [(0.0, 1.0e6)],
        "version": "coverage-validation",
        "parent_oxides": ("A", "B"),
    }


def test_datapack_rejects_invalid_shapes_and_values() -> None:
    cases = [
        ({"nu": np.array([1.0, 1.0])}, "nu must be a 2-D matrix"),
        ({"A": np.array([0.0, 1.0])}, "A shape"),
        ({"B": np.array([0.0, 1.0])}, "B shape"),
        ({"domains": []}, "domains length"),
        ({"reactions": ("AB", "CD")}, "reactions length"),
        ({"parent_oxides": ("A",)}, "parent_oxides length"),
        ({"nu": np.array([[1.0], [-1.0]])}, "nu must be non-negative"),
        ({"A": np.array([np.nan])}, "A and B must be finite"),
    ]
    for overrides, message in cases:
        kwargs = _pack_kwargs()
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=message):
            ImccDatapack(**kwargs)


def test_canonical_serialization_preserves_json_types_and_refuses_invalid_values() -> None:
    encoded = _canonical_published_serialization(
        {"b": [None, True, False, 1.2300, 0.0], "a": "café"}
    )
    assert encoded == b'{"a":"caf\\u00e9","b":[null,true,false,1.23,0]}'

    with pytest.raises(ValueError, match="must be finite"):
        _canonical_published_serialization(float("inf"))
    with pytest.raises(TypeError, match="keys must be strings"):
        _canonical_published_serialization({1: "not a JSON object key"})
    with pytest.raises(TypeError, match="unsupported canonical manifest value"):
        _canonical_published_serialization({"unsupported": {"set-value"}})


def test_identity_issuer_refuses_invalid_coverage_and_published_claims() -> None:
    pack = _ab_pack()
    species = (*pack.parent_oxides, *pack.reactions)

    duplicate = ImccDatapack(
        reactions=("A",),
        nu=np.array([[1.0], [1.0]]),
        A=np.array([0.0]),
        B=np.array([0.0]),
        domains=[(0.0, 1.0e6)],
        version="coverage-duplicate",
        parent_oxides=("A", "B"),
    )
    with pytest.raises(ValueError, match="globally unique"):
        _datapack_with_identity(
            duplicate,
            model_id="coverage",
            coverage={name: "research" for name in ("A", "B")},
        )

    with pytest.raises(ValueError, match="coverage must exactly match"):
        _datapack_with_identity(
            pack,
            model_id="coverage",
            coverage={"A": "research"},
        )
    with pytest.raises(ValueError, match="non-empty string"):
        _datapack_with_identity(
            pack,
            model_id="coverage",
            coverage={name: ("" if name == "A" else "research") for name in species},
        )
    with pytest.raises(ValueError, match="model_id must be a non-empty"):
        _datapack_with_identity(
            pack,
            model_id="",
            coverage="research",
        )
    with pytest.raises(ValueError, match="recognized IMCC model identity"):
        _datapack_with_identity(
            pack,
            model_id="unrecognized",
            coverage=_PUBLISHED_COVERAGE,
            published_manifest_sha256=_PUBLISHED_DATAPACK_SHA256,
        )

    published_coverage = {name: _PUBLISHED_COVERAGE for name in species}
    with pytest.raises(ValueError, match="exactly match the frozen core"):
        _datapack_with_identity(
            pack,
            model_id="IMCC-SF04-EXT",
            coverage=published_coverage,
            published_manifest_sha256=_PUBLISHED_DATAPACK_SHA256,
        )

    parents = _PUBLISHED_PARENT_OXIDES
    reactions = tuple(f"C{index}" for index in range(_PUBLISHED_CORE_ROWS + 1))
    nu = np.zeros((len(parents), len(reactions)))
    for index in range(len(reactions)):
        nu[index % len(parents), index] = 1.0
    extended = ImccDatapack(
        reactions=reactions,
        nu=nu,
        A=np.zeros(len(reactions)),
        B=np.zeros(len(reactions)),
        domains=[(0.0, 1.0e6)] * len(reactions),
        version="coverage-extended",
        parent_oxides=parents,
    )
    extended_coverage = {
        name: _PUBLISHED_COVERAGE
        for name in (*parents, *reactions[:_PUBLISHED_CORE_ROWS])
    }
    extended_coverage[reactions[-1]] = "research"
    with pytest.raises(ValueError, match="cannot include extension rows"):
        _datapack_with_identity(
            extended,
            model_id=_PUBLISHED_MODEL_ID,
            coverage=extended_coverage,
            published_manifest_sha256=_PUBLISHED_DATAPACK_SHA256,
        )


def test_solve_active_refuses_empty_active_parent_set() -> None:
    with pytest.raises(ValueError, match="active parent set is empty"):
        _solve_active(
            np.array([]),
            np.empty((0, 0)),
            np.array([]),
            np.array([]),
            tol=1.0e-12,
            max_iter=100,
        )


def test_solver_reports_a_nonconverged_continuation_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stalled_least_squares(fun: Any, y_init: np.ndarray, **kwargs: Any) -> Any:
        del fun, kwargs
        return SimpleNamespace(
            x=y_init.copy(),
            nfev=1,
            message="synthetic continuation stall",
        )

    monkeypatch.setattr(kernel, "least_squares", stalled_least_squares)
    tolerance = 1.0e-12
    with pytest.raises(ImccNonconvergenceError) as exc:
        _solve_active(
            np.array([0.5, 0.5]),
            np.array([[1.0], [1.0]]),
            np.array([5.0 * kernel.LOG10]),
            np.array([2.0]),
            tol=tolerance,
            max_iter=2,
        )

    continuation = exc.value.diagnostics["continuation"]
    assert len(continuation) == 2
    assert continuation[1]["log_lambda"] < 0.0
    assert continuation[1]["residual_inf"] > tolerance
    assert exc.value.diagnostics["scipy_message"] == "synthetic continuation stall"


def test_kernel_input_validation_reports_typed_refusals() -> None:
    pack = _ab_pack(domains=[(1000.0, 2000.0)])

    cases = [
        (
            np.array([[0.5, 0.5]]),
            {},
            ImccCompositionIncompleteError,
            "1-D",
        ),
        (
            np.array([0.5, 0.5, 0.0]),
            {},
            ImccComponentOutsideDomainError,
            "exceeds",
        ),
        (
            np.array([0.5]),
            {},
            ImccCompositionIncompleteError,
            "does not match",
        ),
        (
            np.array([np.nan, 0.5]),
            {},
            ImccCompositionIncompleteError,
            "non-finite",
        ),
        (
            np.array([-0.1, 1.1]),
            {},
            ImccCompositionIncompleteError,
            "negative",
        ),
        (
            np.array([0.0, 0.0]),
            {},
            ImccCompositionIncompleteError,
            "zero",
        ),
        (
            np.array([0.5, 0.5]),
            {"basis": 0.0},
            ImccCompositionIncompleteError,
            "positive",
        ),
        (
            np.array([0.5, 0.5]),
            {"basis": 2.0},
            ImccCompositionIncompleteError,
            "does not match",
        ),
        (
            np.array([0.5, 0.5]),
            {"T_K": 0.0},
            ImccTOutsideDatapackDomainError,
            "positive finite",
        ),
        (
            np.array([0.5, 0.5]),
            {"T_K": float("inf")},
            ImccTOutsideDatapackDomainError,
            "positive finite",
        ),
        (
            np.array([0.5, 0.5]),
            {"max_iter": "not-a-number"},
            ValueError,
            "max_iter must be a finite",
        ),
    ]
    for parent, kwargs, error, message in cases:
        with pytest.raises(error, match=message) as exc:
            solve_imcc_sf04(parent, kwargs.pop("T_K", 1500.0), pack, **kwargs)
        if hasattr(exc.value, "code"):
            assert exc.value.code.startswith("imcc_")


def test_kernel_refuses_negative_extra_components_with_stable_code() -> None:
    pack = _ab_pack()
    for extra in ({"Fe2O3": -0.1}, {"P2O5": -0.1}):
        with pytest.raises(ImccCompositionIncompleteError) as exc:
            solve_imcc_sf04([0.5, 0.5], 1000.0, pack, extra_mol=extra)
        assert exc.value.code == "imcc_composition_incomplete"

    with pytest.raises(ImccFerricInputUnsupportedError) as exc:
        solve_imcc_sf04([0.5, 0.5], 1000.0, pack, extra_mol={"Fe2O3": 0.1})
    assert exc.value.code == "imcc_ferric_input_unsupported"
