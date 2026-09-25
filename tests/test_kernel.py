"""
Rung-1 gates for the IMCC-SF04 melt-equilibrium kernel.

- Analytic binary systems (closed-form derivation in comments).
- Limiting cases (x -> 0, x -> 1, K -> 0, K -> infinity).
- Atom balance <= 1e-12 relative on every solve.
- Bit-determinism.
- Typed-refusal behaviour per class.
- Log-space stability on a stiff synthetic system.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# The simulator this engine was extracted from owns a "fidelity vocabulary"
# deciding which label strings deny authority. That predicate is deny-by-default
# -- it returns True for almost any string, including "inside" -- so porting it
# would assert close to nothing. Every call site here sat next to an assertion
# on the label's exact VALUE, which is strictly stronger, so the checks below
# assert the value directly. The trust policy itself stays upstream; this
# package has no view on which backends a host considers authoritative.
from openimcc import (
    ImccDatapack,
    ImccComponentOutsideDomainError,
    ImccCompositionIncompleteError,
    ImccFerricInputUnsupportedError,
    ImccNonconvergenceError,
    ImccRefusal,
    ImccTOutsideDatapackDomainError,
    evaluate,
    label_research_datapack,
    load_datapack,
)

# The raw kernel entry point is deliberately NOT package-exported. Kernel tests
# import it from the module directly and require unproven packs to stay denied.
from openimcc.kernel import solve_imcc_sf04
from openimcc.kernel import _active_residual, LOG10


# --------------------------------------------------------------------------- #
# Synthetic datapack helpers
# --------------------------------------------------------------------------- #


def make_ab_datapack(
    A: float = 0.0,
    B: float = 0.0,
    T_lo: float = 0.0,
    T_hi: float = 1.0e6,
    version: str = "synthetic-ab",
) -> ImccDatapack:
    """Single AB complex over two parents A and B."""
    return ImccDatapack(
        reactions=("AB",),
        nu=np.array([[1.0], [1.0]]),
        A=np.array([A]),
        B=np.array([B]),
        domains=[(T_lo, T_hi)],
        version=version,
        parent_oxides=("A", "B"),
    )


def make_8parent_datapack(complexes: list[dict]) -> ImccDatapack:
    """Build a synthetic datapack over the 8 placeholder parents."""
    parents = ("Si", "Mg", "Fe", "Ca", "Al", "Ti", "Na", "K")
    n = len(parents)
    reactions = []
    nu_cols = []
    A = []
    B = []
    domains = []
    for c in complexes:
        reactions.append(c["name"])
        col = np.zeros(n)
        for pname, coeff in c["nu"].items():
            col[parents.index(pname)] = coeff
        nu_cols.append(col)
        A.append(c["A"])
        B.append(c["B"])
        domains.append(c["domain"])
    nu = np.column_stack(nu_cols)
    return ImccDatapack(
        reactions=reactions,
        nu=nu,
        A=np.array(A),
        B=np.array(B),
        domains=domains,
        version="synthetic-8",
        parent_oxides=parents,
    )


# --------------------------------------------------------------------------- #
# Analytic binary solution (A + B -> AB)
# --------------------------------------------------------------------------- #
# For a single AB complex with analytical fractions xA and xB (xA + xB = 1),
# let w = x_AB. From the parent balances:
#     xA* = xA - xB * w
#     xB* = xB - xA * w
# and mass action w = K * xA* * xB*. Substituting:
#     w = K (xA - xB*w)(xB - xA*w)
#       = K [ xA*xB - (xA^2 + xB^2) * w + xA*xB * w^2 ]
# Rearranged to standard quadratic a w^2 + b w + c = 0:
#     a = K * xA * xB
#     b = -[ K * (xA^2 + xB^2) + 1 ]
#     c = K * xA * xB = a
# The physical root is the smaller positive root:
#     w = (-b - sqrt(b^2 - 4 a^2)) / (2 a)
# with a -> 0 limit w -> 0.
# Then gamma_i = x_i* / x_i.


def analytic_ab(xA: float, log10_K: float) -> tuple[float, float, float]:
    K = 10.0 ** log10_K
    xB = 1.0 - xA
    a = K * xA * xB
    if a == 0.0:
        return xA, xB, 0.0
    b = -(K * (xA * xA + xB * xB) + 1.0)
    disc = b * b - 4.0 * a * a
    # The physical root is the smaller positive root. The naive quadratic
    # formula cancels for small K; use the stable form w = (2a) / (-b + sqrtD).
    w = (2.0 * a) / (-b + np.sqrt(disc))
    xAs = xA - xB * w
    xBs = xB - xA * w
    return float(xAs), float(xBs), float(w)


def atom_balance_residual(result, datapack: ImccDatapack) -> float:
    """Maximum relative residual of the parent-oxide mass balances."""
    reconstructed = result.parent_x_star + datapack.nu @ result.complex_x
    expected = result.parent_x * result.D
    return float(np.max(np.abs(reconstructed - expected)) / result.D)


def assert_solve_ok(result, datapack: ImccDatapack) -> float:
    assert np.all(result.parent_x_star >= 0.0)
    assert np.all(result.complex_x >= -1.0e-15)
    assert np.all(np.isfinite(result.species_x))
    # Independent closure checks (not self-referential).
    assert np.isclose(
        np.sum(result.species_x), 1.0, atol=1.0e-12
    ), "species mole fractions do not sum to 1"
    S = datapack.nu.sum(axis=0)
    D_reconstructed = 1.0 + np.sum((S - 1.0) * result.complex_x)
    assert np.isclose(result.D, D_reconstructed, atol=1.0e-12), (
        f"reported D {result.D:.12g} does not match independent "
        f"reconstruction {D_reconstructed:.12g}"
    )
    rel = atom_balance_residual(result, datapack)
    assert rel <= 1.0e-12, f"atom balance relative residual {rel:.3e}"
    return rel


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("xA", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
@pytest.mark.parametrize("log10_K", [-10.0, -2.0, 0.0, 2.0, 10.0])
def test_analytic_binary_ab(xA: float, log10_K: float) -> None:
    pack = make_ab_datapack(A=log10_K, B=0.0)
    parent = np.array([xA, 1.0 - xA])
    res = solve_imcc_sf04(parent, 1000.0, pack)
    xAs_exp, xBs_exp, w_exp = analytic_ab(xA, log10_K)

    assert np.isclose(res.parent_x_star[0], xAs_exp, atol=1.0e-12)
    assert np.isclose(res.parent_x_star[1], xBs_exp, atol=1.0e-12)
    assert np.isclose(res.complex_x[0], w_exp, atol=1.0e-12)

    assert np.isclose(res.parent_activity[0], xAs_exp, atol=1.0e-12)
    assert np.isclose(res.parent_activity[1], xBs_exp, atol=1.0e-12)

    if xA > 0.0:
        assert np.isclose(res.parent_gamma[0], xAs_exp / xA, atol=1.0e-12)
    if 1.0 - xA > 0.0:
        assert np.isclose(res.parent_gamma[1], xBs_exp / (1.0 - xA), atol=1.0e-12)

    assert_solve_ok(res, pack)


def test_limiting_K_to_zero() -> None:
    # K -> 0: no complexing, x* = x, gamma = 1.
    pack = make_ab_datapack(A=-1000.0, B=0.0)
    parent = np.array([0.3, 0.7])
    res = solve_imcc_sf04(parent, 1000.0, pack)
    assert np.allclose(res.parent_x_star, parent, atol=1.0e-12)
    assert np.allclose(res.parent_gamma, 1.0, atol=1.0e-12)
    assert np.isclose(res.complex_x[0], 0.0, atol=1.0e-12)
    assert_solve_ok(res, pack)


def test_limiting_K_to_infinity() -> None:
    # K -> infinity: the minority parent is essentially fully complexed.
    # A=20 gives K=1e20, enough to push the minority parent to machine-zero
    # while staying inside the solver's physical basin.
    pack = make_ab_datapack(A=20.0, B=0.0)
    xA = 0.3
    parent = np.array([xA, 1.0 - xA])
    res = solve_imcc_sf04(parent, 1000.0, pack)
    # A is the minority component; its unbound fraction should be driven to ~0.
    assert res.parent_x_star[0] <= 1.0e-8
    # B remains as excess. With one AB complex, total species moles = 1 - w,
    # where w -> xA in the limit, so xB* = (xB - xA) / (1 - xA).
    assert np.isclose(res.parent_x_star[1], (1.0 - xA - xA) / (1.0 - xA), atol=1.0e-8)
    assert_solve_ok(res, pack)


def test_limiting_x_to_zero() -> None:
    # A -> 0, B -> 1: no AB can form because A is absent.
    pack = make_ab_datapack(A=2.0, B=0.0)
    res = solve_imcc_sf04(np.array([0.0, 1.0]), 1000.0, pack)
    assert np.isclose(res.parent_x_star[0], 0.0, atol=1.0e-12)
    assert np.isclose(res.parent_x_star[1], 1.0, atol=1.0e-12)
    assert np.isclose(res.complex_x[0], 0.0, atol=1.0e-12)
    assert_solve_ok(res, pack)


def test_limiting_x_to_one() -> None:
    # A -> 1, B -> 0: no AB can form because B is absent.
    pack = make_ab_datapack(A=2.0, B=0.0)
    res = solve_imcc_sf04(np.array([1.0, 0.0]), 1000.0, pack)
    assert np.isclose(res.parent_x_star[0], 1.0, atol=1.0e-12)
    assert np.isclose(res.parent_x_star[1], 0.0, atol=1.0e-12)
    assert np.isclose(res.complex_x[0], 0.0, atol=1.0e-12)
    assert_solve_ok(res, pack)


def test_atom_balance_on_all_solves() -> None:
    # Sweep many states and confirm the ledger closes on every solve.
    worst = 0.0
    pack = make_ab_datapack(A=1.0, B=2000.0)
    for xA in np.linspace(0.01, 0.99, 17):
        for T in np.linspace(800.0, 2200.0, 9):
            parent = np.array([xA, 1.0 - xA])
            res = solve_imcc_sf04(parent, T, pack)
            rel = assert_solve_ok(res, pack)
            worst = max(worst, rel)
    assert worst <= 1.0e-12


def test_bit_determinism() -> None:
    pack = make_ab_datapack(A=1.0, B=500.0)
    parent = np.array([0.3, 0.7])
    res1 = solve_imcc_sf04(parent, 1500.0, pack)
    res2 = solve_imcc_sf04(parent, 1500.0, pack)
    assert np.array_equal(res1.parent_x_star, res2.parent_x_star)
    assert np.array_equal(res1.complex_x, res2.complex_x)
    assert np.array_equal(res1.parent_gamma, res2.parent_gamma)
    assert np.array_equal(res1.species_x, res2.species_x)
    assert res1.D == res2.D
    assert res1.convergence.iterations == res2.convergence.iterations


def test_refusal_T_outside_domain() -> None:
    pack = make_ab_datapack(A=0.0, B=0.0, T_lo=1000.0, T_hi=2000.0)
    parent = np.array([0.5, 0.5])
    with pytest.raises(ImccTOutsideDatapackDomainError):
        solve_imcc_sf04(parent, 900.0, pack)
    with pytest.raises(ImccTOutsideDatapackDomainError):
        solve_imcc_sf04(parent, 2100.0, pack)
    res = solve_imcc_sf04(parent, 900.0, pack, allow_extrapolation=True)
    assert res.extrapolated is True
    assert_solve_ok(res, pack)


def test_refusal_composition_incomplete() -> None:
    pack = make_ab_datapack()
    parent = np.array([0.5, 0.5])
    with pytest.raises(ImccCompositionIncompleteError):
        solve_imcc_sf04(np.array([-0.1, 0.6]), 1000.0, pack)
    with pytest.raises(ImccCompositionIncompleteError):
        solve_imcc_sf04(np.array([0.5, 0.6]), 1000.0, pack, basis=1.0)
    with pytest.raises(ImccCompositionIncompleteError):
        solve_imcc_sf04(np.array([0.0, 0.0]), 1000.0, pack)
    with pytest.raises(ImccCompositionIncompleteError):
        solve_imcc_sf04(np.array([np.nan, 0.5]), 1000.0, pack)


@pytest.mark.parametrize("nu_value", [float("nan"), float("inf"), float("-inf")])
def test_datapack_refuses_nonfinite_nu(nu_value: float) -> None:
    with pytest.raises(ValueError) as exc:
        ImccDatapack(
            reactions=("AB",),
            nu=np.array([[nu_value], [1.0]]),
            A=np.array([0.0]),
            B=np.array([0.0]),
            domains=[(0.0, 1.0e6)],
            version="nonfinite-nu",
            parent_oxides=("A", "B"),
        )

    assert "nu must be finite" in str(exc.value)


def test_refusal_component_outside_domain() -> None:
    pack = make_ab_datapack()
    parent = np.array([0.5, 0.5])
    # Wrong vector length (extra positive component).
    with pytest.raises(ImccComponentOutsideDomainError):
        solve_imcc_sf04(np.array([0.5, 0.5, 0.0]), 1000.0, pack)
    # Unsupported positive extra component.
    with pytest.raises(ImccComponentOutsideDomainError):
        solve_imcc_sf04(parent, 1000.0, pack, extra_mol={"P2O5": 0.1})


def test_refusal_ferric_input() -> None:
    pack = make_ab_datapack()
    parent = np.array([0.5, 0.5])
    with pytest.raises(ImccFerricInputUnsupportedError):
        solve_imcc_sf04(parent, 1000.0, pack, extra_mol={"Fe2O3": 0.1})
    # Zero Fe2O3 is allowed and must not raise.
    res = solve_imcc_sf04(parent, 1000.0, pack, extra_mol={"Fe2O3": 0.0})
    assert_solve_ok(res, pack)


def test_refusal_nonconvergence() -> None:
    pack = make_ab_datapack(A=1.0, B=0.0)
    parent = np.array([0.5, 0.5])
    with pytest.raises(ImccNonconvergenceError) as exc:
        solve_imcc_sf04(parent, 1000.0, pack, max_iter=0)
    assert exc.value.diagnostics["iterations"] == 0


def test_log_space_stiff_synthetic() -> None:
    # Wide K span across ~13 orders of magnitude, chosen to stay inside the
    # finite range of double-precision exponentials during the solve.
    complexes = [
        {
            "name": "Mg2Si",
            "nu": {"Mg": 2.0, "Si": 1.0},
            "A": 5.0,
            "B": 0.0,
            "domain": (500.0, 2500.0),
        },
        {
            "name": "FeAl2",
            "nu": {"Fe": 1.0, "Al": 2.0},
            "A": -5.0,
            "B": 0.0,
            "domain": (500.0, 2500.0),
        },
        {
            "name": "CaTi",
            "nu": {"Ca": 1.0, "Ti": 1.0},
            "A": 3.0,
            "B": 0.0,
            "domain": (500.0, 2500.0),
        },
        {
            "name": "Na2K",
            "nu": {"Na": 2.0, "K": 1.0},
            "A": 8.0,
            "B": 0.0,
            "domain": (500.0, 2500.0),
        },
        {
            "name": "SiAl2",
            "nu": {"Si": 1.0, "Al": 2.0},
            "A": 0.0,
            "B": 0.0,
            "domain": (500.0, 2500.0),
        },
    ]
    pack = make_8parent_datapack(complexes)
    parent = np.array([0.15, 0.20, 0.10, 0.10, 0.15, 0.10, 0.10, 0.10])
    res = solve_imcc_sf04(parent, 1500.0, pack)
    rel = assert_solve_ok(res, pack)
    assert np.all(res.parent_activity >= 0.0)
    assert np.all(np.isfinite(res.parent_gamma))
    assert res.labels.model_id == "internal-analytical"
    assert res.labels.evidence_class == "internal-analytical"
    assert rel <= 1.0e-12


def test_raw_result_labels_default_to_denied_untrusted_identity() -> None:
    pack = make_ab_datapack(A=0.0, B=0.0, version="r2.1-test")
    parent = np.array([0.4, 0.6])
    res = solve_imcc_sf04(parent, 1200.0, pack)
    assert res.labels.model_id == "internal-analytical"
    assert res.labels.datapack_version == "r2.1-test"
    assert res.labels.evidence_class == "internal-analytical"
    assert res.labels.evidence_class == "internal-analytical"
    for name in res.species_names:
        assert res.labels.coverage[name] == "internal-analytical"
    assert res.extrapolated is False


# --------------------------------------------------------------------------- #
# Regression tests for review findings (P1-1, P2-1..P2-7)
# --------------------------------------------------------------------------- #


def make_3parent_inactive_datapack() -> ImccDatapack:
    """3-parent complex over 8 placeholder parents; complex is inactive when Fe=0."""
    parents = ("Si", "Mg", "Fe", "Ca", "Al", "Ti", "Na", "K")
    n = len(parents)
    nu = np.zeros((n, 1))
    nu[0, 0] = 1.0  # Si
    nu[1, 0] = 1.0  # Mg
    nu[2, 0] = 1.0  # Fe
    return ImccDatapack(
        reactions=("SiMgFe",),
        nu=nu,
        A=np.array([0.0]),
        B=np.array([0.0]),
        domains=[(1500.0, 2500.0)],
        version="synthetic-3parent-inactive",
        parent_oxides=parents,
    )


def test_p1_1_adv_4c_inactive_complex_domain_refusal() -> None:
    # 3-parent complex with Fe=0 cannot form; its [1500, 2500] K domain must not
    # cause refusal at 1000 K.
    pack = make_3parent_inactive_datapack()
    parent = np.array([0.2, 0.2, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1])
    res = solve_imcc_sf04(parent, 1000.0, pack)
    assert_solve_ok(res, pack)
    assert res.extrapolated is False


def test_p1_1_adv_4d_inactive_complex_extrapolated_false() -> None:
    # Same inactive-complex setup under allow_extrapolation: nothing is consumed
    # outside its domain, so extrapolated must remain False.
    pack = make_3parent_inactive_datapack()
    parent = np.array([0.2, 0.2, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1])
    res = solve_imcc_sf04(parent, 1000.0, pack, allow_extrapolation=True)
    assert_solve_ok(res, pack)
    assert res.extrapolated is False


def test_p2_1_jacobian_matches_finite_differences() -> None:
    # Verify the analytical Jacobian code against central finite differences.
    # The docstring must describe this form: x_i is the analytical target,
    # not x_i*.
    pack = make_ab_datapack(A=1.0, B=0.0)
    parent = np.array([0.3, 0.7])
    T = 1000.0
    log10_K = pack.A + pack.B / T
    lnK = LOG10 * log10_K
    S = pack.nu.sum(axis=0)
    x = parent / np.sum(parent)
    y0 = np.log(np.maximum(x, 1.0e-12))
    f0, _g, _D, J_analytic = _active_residual(y0, x, pack.nu, lnK, S)

    h = 1.0e-7
    J_fd = np.zeros_like(J_analytic)
    for i in range(len(y0)):
        y_plus = y0.copy()
        y_minus = y0.copy()
        y_plus[i] += h
        y_minus[i] -= h
        f_plus, *_ = _active_residual(y_plus, x, pack.nu, lnK, S)
        f_minus, *_ = _active_residual(y_minus, x, pack.nu, lnK, S)
        J_fd[:, i] = (f_plus - f_minus) / (2.0 * h)

    assert np.allclose(J_analytic, J_fd, atol=1.0e-6)


def test_p2_2_independent_atom_balance_closure() -> None:
    # Atom-balance closure must be checkable without relying on kernel-reported D.
    pack = make_ab_datapack(A=1.0, B=2000.0)
    parent = np.array([0.4, 0.6])
    res = solve_imcc_sf04(parent, 1500.0, pack)
    assert_solve_ok(res, pack)
    # Independent checks repeated explicitly (assert_solve_ok already includes them).
    assert np.isclose(np.sum(res.species_x), 1.0, atol=1.0e-12)
    S = pack.nu.sum(axis=0)
    D_reconstructed = 1.0 + np.sum((S - 1.0) * res.complex_x)
    assert np.isclose(res.D, D_reconstructed, atol=1.0e-12)


def test_p2_3_gamma_greater_than_one() -> None:
    # Inert A + B-dimer (B2). B strongly dimerizes, shrinking the total-species
    # denominator; A is weakly complexed and must show gamma_A > 1.
    pack = ImccDatapack(
        reactions=("B2",),
        nu=np.array([[0.0], [2.0]]),
        A=np.array([np.log10(10.0)]),
        B=np.array([0.0]),
        domains=[(0.0, 1.0e6)],
        version="synthetic-b2",
        parent_oxides=("A", "B"),
    )
    parent = np.array([0.4, 0.6])
    res = solve_imcc_sf04(parent, 1000.0, pack)
    assert_solve_ok(res, pack)
    assert res.parent_gamma[0] > 1.0, (
        f"gamma_A = {res.parent_gamma[0]:.6f} should exceed 1 for an inert "
        "parent in a strongly dimerized melt"
    )


def test_p2_4_negative_extra_mol_ferric() -> None:
    pack = make_ab_datapack()
    parent = np.array([0.5, 0.5])
    with pytest.raises(ImccCompositionIncompleteError):
        solve_imcc_sf04(parent, 1000.0, pack, extra_mol={"Fe2O3": -0.1})


def test_p2_4_negative_extra_mol_component() -> None:
    pack = make_ab_datapack()
    parent = np.array([0.5, 0.5])
    with pytest.raises(ImccCompositionIncompleteError):
        solve_imcc_sf04(parent, 1000.0, pack, extra_mol={"P2O5": -0.5})


def test_p2_5_scalar_input() -> None:
    pack = make_ab_datapack()
    with pytest.raises(ImccCompositionIncompleteError):
        solve_imcc_sf04(0.5, 1000.0, pack)


def test_p2_6_nonfinite_residual() -> None:
    # K=1e300 overflows the residual; must raise ImccNonconvergenceError, not a
    # raw scipy ValueError.
    pack = make_ab_datapack(A=1.0e300, B=0.0)
    parent = np.array([0.5, 0.5])
    with pytest.raises(ImccNonconvergenceError) as exc:
        solve_imcc_sf04(parent, 1000.0, pack)
    assert "non-finite" in str(exc.value).lower()


def test_extreme_extrapolated_evaluate_has_no_runtime_warning() -> None:
    pack = load_datapack("src/openimcc/data/packs/imcc-sf04-v1.0.2.json")
    composition = {"SiO2": 50.0, "CaO": 20.0, "MgO": 15.0, "Al2O3": 15.0}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        try:
            result = evaluate(
                composition,
                500.0,
                pack,
                basis_type="wt",
                allow_extrapolation=True,
                allow_out_of_envelope=True,
            )
        except ImccRefusal:
            result = None

    assert result is None or result.extrapolated is True
    assert not [warning for warning in caught if warning.category is RuntimeWarning]


def test_p2_7_total_displacement() -> None:
    pack = make_ab_datapack(A=1.0, B=0.0)
    parent = np.array([0.5, 0.5])
    res = solve_imcc_sf04(parent, 1000.0, pack)
    assert res.convergence.total_displacement > 0.0

    # Force a nonconvergence where the solver reached a y before failing; the
    # diagnostics must report the actual displacement, not a hardcoded 0.0.
    # The 8-parent stiff problem needs more than max_iter=1 evaluations, so
    # least_squares terminates before the residual is below tol.
    stiff_complexes = [
        {
            "name": "Mg2Si",
            "nu": {"Mg": 2.0, "Si": 1.0},
            "A": 5.0,
            "B": 0.0,
            "domain": (500.0, 2500.0),
        },
        {
            "name": "FeAl2",
            "nu": {"Fe": 1.0, "Al": 2.0},
            "A": -5.0,
            "B": 0.0,
            "domain": (500.0, 2500.0),
        },
        {
            "name": "CaTi",
            "nu": {"Ca": 1.0, "Ti": 1.0},
            "A": 3.0,
            "B": 0.0,
            "domain": (500.0, 2500.0),
        },
        {
            "name": "Na2K",
            "nu": {"Na": 2.0, "K": 1.0},
            "A": 8.0,
            "B": 0.0,
            "domain": (500.0, 2500.0),
        },
        {
            "name": "SiAl2",
            "nu": {"Si": 1.0, "Al": 2.0},
            "A": 0.0,
            "B": 0.0,
            "domain": (500.0, 2500.0),
        },
    ]
    stiff_pack = make_8parent_datapack(stiff_complexes)
    stiff_parent = np.array([0.15, 0.20, 0.10, 0.10, 0.15, 0.10, 0.10, 0.10])
    with pytest.raises(ImccNonconvergenceError) as exc:
        solve_imcc_sf04(stiff_parent, 1500.0, stiff_pack, tol=1.0e-12, max_iter=1)
    assert exc.value.diagnostics["total_displacement"] > 0.0


# --------------------------------------------------------------------------- #
# SC-130 production-sweep regressions (kernel-codex findings 1–3, 8)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "tol",
    [float("inf"), float("-inf"), float("nan"), 0.0, -1.0],
)
def test_sc130_tol_non_positive_or_nonfinite_refused(tol: float) -> None:
    pack = make_ab_datapack()
    parent = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="tol must be a positive finite"):
        solve_imcc_sf04(parent, 1000.0, pack, tol=tol)


@pytest.mark.parametrize("max_iter", [float("inf"), float("-inf"), float("nan")])
def test_sc130_max_iter_nonfinite_refused(max_iter: float) -> None:
    pack = make_ab_datapack()
    parent = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="max_iter must be a finite"):
        solve_imcc_sf04(parent, 1000.0, pack, max_iter=max_iter)


def test_sc130_max_iter_huge_int_refused() -> None:
    pack = make_ab_datapack()
    parent = np.array([0.5, 0.5])
    with pytest.raises(ValueError) as exc:
        solve_imcc_sf04(parent, 1000.0, pack, max_iter=10**400)

    assert "max_iter must be a finite number" in str(exc.value)


@pytest.mark.parametrize(
    "window",
    [
        (float("nan"), float("nan")),
        (float("nan"), 2000.0),
        (1000.0, float("nan")),
        (float("inf"), float("inf")),
        (float("-inf"), float("inf")),
        (float("-inf"), 2000.0),
        (1000.0, float("inf")),
    ],
)
def test_sc130_nonfinite_domain_window_refused(
    window: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="finite Kelvin"):
        ImccDatapack(
            reactions=("AB",),
            nu=np.array([[1.0], [1.0]]),
            A=np.array([0.0]),
            B=np.array([0.0]),
            domains=[window],
            version="sc130-bad-domain",
            parent_oxides=("A", "B"),
        )


def test_sc130_domain_window_must_be_a_pair() -> None:
    with pytest.raises(ValueError, match="must be a \\(T_low, T_high\\) pair"):
        ImccDatapack(
            reactions=("AB",),
            nu=np.array([[1.0], [1.0]]),
            A=np.array([0.0]),
            B=np.array([0.0]),
            domains=[(0.0, 1.0e6, 2.0e6)],  # type: ignore[list-item]
            version="sc130-bad-domain-shape",
            parent_oxides=("A", "B"),
        )


def test_sc130_zero_stoichiometry_column_refused() -> None:
    with pytest.raises(ValueError, match="at least one positive coefficient"):
        ImccDatapack(
            reactions=("ghost",),
            nu=np.array([[0.0]]),
            A=np.array([-1.0]),
            B=np.array([0.0]),
            domains=[(0.0, 1.0e6)],
            version="sc130-ghost",
            parent_oxides=("A",),
        )
    with pytest.raises(ValueError, match="at least one positive coefficient"):
        ImccDatapack(
            reactions=("ghost", "AB"),
            nu=np.array([[0.0, 1.0], [0.0, 1.0]]),
            A=np.array([-1.0, 0.0]),
            B=np.array([0.0, 0.0]),
            domains=[(0.0, 1.0e6), (0.0, 1.0e6)],
            version="sc130-ghost-ab",
            parent_oxides=("A", "B"),
        )


# --------------------------------------------------------------------------- #
# Rung-3 regression: real workbook compositions that refused with the original
# ideal-fraction start (31/70 sheet-T melt solves).  Each vector is the 8-oxide
# wt% feed recorded in docs-private/.../rung3/workings.json, with Fe2O3 already
# folded into FeO per the rung-3 protocol.
# --------------------------------------------------------------------------- #

_RUNG3_PARENTS = (
    "SiO2",
    "MgO",
    "FeO",
    "CaO",
    "Al2O3",
    "TiO2",
    "Na2O",
    "K2O",
)

# Exact 38-row v1.0.1 kernel fixture used by the rung-3 run. Keeping the
# sparse stoichiometry and A+B/T coefficients here makes the regression run in
# a fresh checkout; docs-private evidence is intentionally gitignored.
_RUNG3_COMPLEXES = [
    ("Mg2SiO4", {"SiO2": 1, "MgO": 2}, -0.94, 7434, (2500, 3500)),
    ("MgSiO3", {"SiO2": 1, "MgO": 1}, 0.42, 2329, (2500, 3500)),
    ("MgAl2O4", {"MgO": 1, "Al2O3": 1}, 1.18, 464, (2500, 3500)),
    ("MgTiO3", {"MgO": 1, "TiO2": 1}, -0.13, 3246, (2500, 3500)),
    ("MgTi2O5", {"MgO": 1, "TiO2": 2}, 0.51, 2845, (2500, 3500)),
    ("Mg2TiO4", {"MgO": 2, "TiO2": 1}, 0.67, 3812, (2500, 3500)),
    ("Al6Si2O13", {"SiO2": 2, "Al2O3": 3}, -2.94, 9375, (2500, 3500)),
    ("CaAl2O4", {"CaO": 1, "Al2O3": 1}, -1.89, 10060, (2500, 3500)),
    ("CaAl4O7", {"CaO": 1, "Al2O3": 2}, -0.59, 9713, (2500, 3500)),
    ("Ca12Al14O33", {"CaO": 12, "Al2O3": 7}, -6.3, 72239, (2500, 3500)),
    ("CaSiO3", {"SiO2": 1, "CaO": 1}, 0.54, 5568, (2500, 3500)),
    ("CaAl2Si2O8", {"SiO2": 2, "CaO": 1, "Al2O3": 1}, 2.63, 5326, (2500, 3500)),
    ("CaMgSi2O6", {"SiO2": 2, "MgO": 1, "CaO": 1}, 1.46, 8485, (2500, 3500)),
    ("Ca2MgSi2O7", {"SiO2": 2, "MgO": 1, "CaO": 2}, 0.63, 15327, (2500, 3500)),
    ("Ca2Al2SiO7", {"SiO2": 1, "CaO": 2, "Al2O3": 1}, 2.01, 10710, (2500, 3500)),
    ("CaTiO3", {"CaO": 1, "TiO2": 1}, -0.08, 7055, (2500, 3500)),
    ("Ca2SiO4", {"SiO2": 1, "CaO": 2}, 0.63, 8416, (2500, 3500)),
    ("CaTiSiO5", {"SiO2": 1, "CaO": 1, "TiO2": 1}, -0.18, 10071, (2500, 3500)),
    ("FeTiO3", {"FeO": 1, "TiO2": 1}, -0.51, 3569, (2500, 3500)),
    ("Fe2SiO4", {"SiO2": 1, "FeO": 2}, -0.63, 3103, (2500, 3500)),
    ("FeAl2O4", {"FeO": 1, "Al2O3": 1}, -1.76, 5692, (2500, 3500)),
    ("CaAl12O19", {"CaO": 1, "Al2O3": 6}, -3.79, 22612, (2500, 3500)),
    ("Mg2Al4Si5O18", {"SiO2": 5, "MgO": 2, "Al2O3": 2}, 7.48, 0, (2500, 3500)),
    ("Na2SiO3", {"SiO2": 1, "Na2O": 1}, -1.33, 13870, (2500, 3500)),
    ("Na2Si2O5", {"SiO2": 2, "Na2O": 1}, -1.39, 15350, (2500, 3500)),
    ("NaAlSiO4", {"SiO2": 1, "Al2O3": 0.5, "Na2O": 0.5}, 0.65, 6997, (2500, 3500)),
    ("NaAlSi3O8", {"SiO2": 3, "Al2O3": 0.5, "Na2O": 0.5}, 1.29, 8788, (2500, 3500)),
    ("NaAlO2", {"Al2O3": 0.5, "Na2O": 0.5}, 0.55, 3058, (2500, 3500)),
    ("Na2TiO3", {"TiO2": 1, "Na2O": 1}, -1.38, 15445, (2500, 3500)),
    ("NaAlSi2O6", {"SiO2": 2, "Al2O3": 0.5, "Na2O": 0.5}, -1.02, 9607, (2500, 3500)),
    ("K2SiO3", {"SiO2": 1, "K2O": 1}, 0.27, 12735, (1700, 3000)),
    ("K2Si2O5", {"SiO2": 2, "K2O": 1}, 0.35, 14685, (1700, 3000)),
    ("KAlSiO4", {"SiO2": 1, "Al2O3": 0.5, "K2O": 0.5}, 0.97, 8675, (2500, 3500)),
    ("KAlSi3O8", {"SiO2": 3, "Al2O3": 0.5, "K2O": 0.5}, 1.11, 11229, (2500, 3500)),
    ("KAlO2", {"Al2O3": 0.5, "K2O": 0.5}, 0.72, 4679, (2500, 3500)),
    ("KAlSi2O6", {"SiO2": 2, "Al2O3": 0.5, "K2O": 0.5}, 1.53, 10125, (2500, 3500)),
    ("K2Si4O9", {"SiO2": 4, "K2O": 1}, -0.96, 17572, (1700, 3000)),
    ("KCaAlSi2O7", {"SiO2": 2, "CaO": 1, "Al2O3": 0.5, "K2O": 0.5}, 4.3, 17037, (1700, 3000)),
]


def make_rung3_datapack() -> ImccDatapack:
    """Build the tracked test copy of the source-verified v1.0.1 datapack."""
    nu = np.zeros((len(_RUNG3_PARENTS), len(_RUNG3_COMPLEXES)))
    for column, (_name, stoich, _A, _B, _domain) in enumerate(_RUNG3_COMPLEXES):
        for parent, coefficient in stoich.items():
            nu[_RUNG3_PARENTS.index(parent), column] = coefficient
    raw = ImccDatapack(
        reactions=tuple(row[0] for row in _RUNG3_COMPLEXES),
        nu=nu,
        A=np.array([row[2] for row in _RUNG3_COMPLEXES]),
        B=np.array([row[3] for row in _RUNG3_COMPLEXES]),
        domains=[row[4] for row in _RUNG3_COMPLEXES],
        version="rung3-regression-v1.0.1",
        parent_oxides=_RUNG3_PARENTS,
    )
    return label_research_datapack(
        raw,
        model_id="IMCC-SF04-RE",
        coverage="RE-regression",
    )


# 8-oxide wt% vectors (Fe2O3 already folded into FeO) for the 31 (sheet, T)
# melt solves that refused imcc_nonconvergence in the rung-3 workbook
# regression.  Transcribed from docs-private/.../rung3/workings.json.
_RUNG3_REGRESSION_CASES = [
    # (sheet, T_K, {oxide: wt%})
    ("tho", 1500.0, {"SiO2": 50.71, "MgO": 4.68, "FeO": 13.470072141126078, "CaO": 8.83, "Al2O3": 14.48, "TiO2": 1.7, "Na2O": 3.16, "K2O": 0.77}),
    ("aba", 1500.0, {"SiO2": 44.8, "MgO": 11.07, "FeO": 12.24844783858423, "CaO": 10.16, "Al2O3": 13.86, "TiO2": 1.96, "Na2O": 3.19, "K2O": 1.09}),
    ("aba", 1625.0, {"SiO2": 44.8, "MgO": 11.07, "FeO": 12.24844783858423, "CaO": 10.16, "Al2O3": 13.86, "TiO2": 1.96, "Na2O": 3.19, "K2O": 1.09}),
    ("aba", 1750.0, {"SiO2": 44.8, "MgO": 11.07, "FeO": 12.24844783858423, "CaO": 10.16, "Al2O3": 13.86, "TiO2": 1.96, "Na2O": 3.19, "K2O": 1.09}),
    ("aba", 1900.0, {"SiO2": 44.8, "MgO": 11.07, "FeO": 12.24844783858423, "CaO": 10.16, "Al2O3": 13.86, "TiO2": 1.96, "Na2O": 3.19, "K2O": 1.09}),
    ("aba", 2000.0, {"SiO2": 44.8, "MgO": 11.07, "FeO": 12.24844783858423, "CaO": 10.16, "Al2O3": 13.86, "TiO2": 1.96, "Na2O": 3.19, "K2O": 1.09}),
    ("aba", 2125.0, {"SiO2": 44.8, "MgO": 11.07, "FeO": 12.24844783858423, "CaO": 10.16, "Al2O3": 13.86, "TiO2": 1.96, "Na2O": 3.19, "K2O": 1.09}),
    ("kom", 1500.0, {"SiO2": 47.1, "MgO": 29.6, "FeO": 11.53, "CaO": 5.44, "Al2O3": 4.04, "TiO2": 0.24, "Na2O": 0.46, "K2O": 0.09}),
    ("kom", 1625.0, {"SiO2": 47.1, "MgO": 29.6, "FeO": 11.53, "CaO": 5.44, "Al2O3": 4.04, "TiO2": 0.24, "Na2O": 0.46, "K2O": 0.09}),
    ("kom", 1900.0, {"SiO2": 47.1, "MgO": 29.6, "FeO": 11.53, "CaO": 5.44, "Al2O3": 4.04, "TiO2": 0.24, "Na2O": 0.46, "K2O": 0.09}),
    ("kom", 2250.0, {"SiO2": 47.1, "MgO": 29.6, "FeO": 11.53, "CaO": 5.44, "Al2O3": 4.04, "TiO2": 0.24, "Na2O": 0.46, "K2O": 0.09}),
    ("kom", 2375.0, {"SiO2": 47.1, "MgO": 29.6, "FeO": 11.53, "CaO": 5.44, "Al2O3": 4.04, "TiO2": 0.24, "Na2O": 0.46, "K2O": 0.09}),
    ("kom", 2500.0, {"SiO2": 47.1, "MgO": 29.6, "FeO": 11.53, "CaO": 5.44, "Al2O3": 4.04, "TiO2": 0.24, "Na2O": 0.46, "K2O": 0.09}),
    ("dun", 1500.0, {"SiO2": 40.2, "MgO": 43.2, "FeO": 13.609639482237126, "CaO": 0.8, "Al2O3": 0.8, "TiO2": 0.2, "Na2O": 0.3, "K2O": 0.1}),
    ("dun", 1625.0, {"SiO2": 40.2, "MgO": 43.2, "FeO": 13.609639482237126, "CaO": 0.8, "Al2O3": 0.8, "TiO2": 0.2, "Na2O": 0.3, "K2O": 0.1}),
    ("dun", 1750.0, {"SiO2": 40.2, "MgO": 43.2, "FeO": 13.609639482237126, "CaO": 0.8, "Al2O3": 0.8, "TiO2": 0.2, "Na2O": 0.3, "K2O": 0.1}),
    ("dun", 1900.0, {"SiO2": 40.2, "MgO": 43.2, "FeO": 13.609639482237126, "CaO": 0.8, "Al2O3": 0.8, "TiO2": 0.2, "Na2O": 0.3, "K2O": 0.1}),
    ("dun", 2250.0, {"SiO2": 40.2, "MgO": 43.2, "FeO": 13.609639482237126, "CaO": 0.8, "Al2O3": 0.8, "TiO2": 0.2, "Na2O": 0.3, "K2O": 0.1}),
    ("bit", 1500.0, {"SiO2": 75.6, "MgO": 0.21, "FeO": 0.25, "CaO": 0.95, "Al2O3": 13.0, "TiO2": 1.1, "Na2O": 3.35, "K2O": 5.55}),
    ("bit", 1625.0, {"SiO2": 75.6, "MgO": 0.21, "FeO": 0.25, "CaO": 0.95, "Al2O3": 13.0, "TiO2": 1.1, "Na2O": 3.35, "K2O": 5.55}),
    ("bit", 1750.0, {"SiO2": 75.6, "MgO": 0.21, "FeO": 0.25, "CaO": 0.95, "Al2O3": 13.0, "TiO2": 1.1, "Na2O": 3.35, "K2O": 5.55}),
    ("bit", 1875.0, {"SiO2": 75.6, "MgO": 0.21, "FeO": 0.25, "CaO": 0.95, "Al2O3": 13.0, "TiO2": 1.1, "Na2O": 3.35, "K2O": 5.55}),
    ("bit", 2000.0, {"SiO2": 75.6, "MgO": 0.21, "FeO": 0.25, "CaO": 0.95, "Al2O3": 13.0, "TiO2": 1.1, "Na2O": 3.35, "K2O": 5.55}),
    ("bit", 2125.0, {"SiO2": 75.6, "MgO": 0.21, "FeO": 0.25, "CaO": 0.95, "Al2O3": 13.0, "TiO2": 1.1, "Na2O": 3.35, "K2O": 5.55}),
    ("bit", 2250.0, {"SiO2": 75.6, "MgO": 0.21, "FeO": 0.25, "CaO": 0.95, "Al2O3": 13.0, "TiO2": 1.1, "Na2O": 3.35, "K2O": 5.55}),
    ("bit", 2375.0, {"SiO2": 75.6, "MgO": 0.21, "FeO": 0.25, "CaO": 0.95, "Al2O3": 13.0, "TiO2": 1.1, "Na2O": 3.35, "K2O": 5.55}),
    ("bit", 2500.0, {"SiO2": 75.6, "MgO": 0.21, "FeO": 0.25, "CaO": 0.95, "Al2O3": 13.0, "TiO2": 1.1, "Na2O": 3.35, "K2O": 5.55}),
    ("cai", 1500.0, {"SiO2": 29.1, "MgO": 10.2, "FeO": 0.6, "CaO": 28.8, "Al2O3": 29.6, "TiO2": 1.3, "Na2O": 0.18, "K2O": 0.1}),
    ("cai", 1625.0, {"SiO2": 29.1, "MgO": 10.2, "FeO": 0.6, "CaO": 28.8, "Al2O3": 29.6, "TiO2": 1.3, "Na2O": 0.18, "K2O": 0.1}),
    ("cai", 1750.0, {"SiO2": 29.1, "MgO": 10.2, "FeO": 0.6, "CaO": 28.8, "Al2O3": 29.6, "TiO2": 1.3, "Na2O": 0.18, "K2O": 0.1}),
    ("cai", 2125.0, {"SiO2": 29.1, "MgO": 10.2, "FeO": 0.6, "CaO": 28.8, "Al2O3": 29.6, "TiO2": 1.3, "Na2O": 0.18, "K2O": 0.1}),
]


@pytest.mark.parametrize("sheet, T_K, comp_wt", _RUNG3_REGRESSION_CASES)
def test_rung3_refusing_compositions_converge(
    sheet: str, T_K: float, comp_wt: dict[str, float]
) -> None:
    """Previously refusing workbook compositions must now solve and close balance."""
    pack = make_rung3_datapack()
    res = evaluate(
        comp_wt,
        T_K=T_K,
        pack=pack,
        basis_type="wt",
        allow_extrapolation=True,
        tol=1.0e-12,
    )
    assert_solve_ok(res, pack)
    assert res.convergence.residual_inf <= 1.0e-12, f"{sheet}@{T_K:g} K"
    assert res.convergence.status == "converged"


def test_rung3_bit_1900_control_converges() -> None:
    """The requested bit@1900 K control was not one of the baseline refusals."""
    pack = make_rung3_datapack()
    bit_wt = {
        "SiO2": 75.6,
        "MgO": 0.21,
        "FeO": 0.25,
        "CaO": 0.95,
        "Al2O3": 13.0,
        "TiO2": 1.1,
        "Na2O": 3.35,
        "K2O": 5.55,
    }
    res = evaluate(
        bit_wt,
        T_K=1900.0,
        pack=pack,
        basis_type="wt",
        allow_extrapolation=True,
        tol=1.0e-12,
    )
    assert_solve_ok(res, pack)
    assert res.convergence.residual_inf <= 1.0e-12
