"""Experimental binary MgO–SiO2 modified-quasichemical candidate.

This module evaluates only MgO–SiO2 formula-unit mole amounts. It does not
embed the binary in multicomponent melts or replace the default evaluator.
Model and parameter citation: Wu, Eriksson, Pelton & Blander (1993),
*ISIJ International* 33(1), 26–35, DOI 10.2355/isijinternational.33.26.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Literal

from openimcc._constants import R_J_MOL_K
from openimcc.model import (
    ImccComponentOutsideDomainError,
    ImccCompositionIncompleteError,
    ImccTOutsideDatapackDomainError,
)
from openimcc.scalar_boundary import is_declared_real_scalar

_R = R_J_MOL_K
# Wu 1993, p. 27: b_SiO2=1.3774 and b_MgO=b_SiO2/2; store z=2b so
# Y_i=z_i*x_i/sum(z*x) and the pair count is sum(z*x)/2.
_Z_MGO = 1.37744375
_Z_SIO2 = 2.7548875
_T_MIN_K = 1200.0
_T_MAX_K = 3000.0
_ACTIVITY_STEP = 1.0e-5
_ACTIVITY_STENCIL_REL_TOL = 0.05
_BINODAL_STEP = 0.001

_SOURCE = (
    "Wu, P., Eriksson, G., Pelton, A.D. & Blander, M. (1993), "
    "ISIJ International 33(1), 26–35, DOI 10.2355/isijinternational.33.26; "
    "Eqs. (2)–(8), pp. 27, 30; liquid reference functions, Table 1, p. 28."
)
_CANDIDATE_ID = "wu-1993-mgo-sio2-mqm-openimcc-liquid-standards-v1"


@dataclass(frozen=True)
class MgOSiO2MQMResult:
    """Immutable output from the EXPERIMENTAL Wu 1993 binary candidate."""

    a_sio2: float
    a_mgo: float
    x_sio2: float
    temperature_K: float
    pair_fractions: tuple[float, float, float]
    stability: Literal["homogeneous", "inside_two_liquid_field"]
    two_liquid_field_x_bounds: tuple[float, float] | None
    candidate_id: str
    experimental_status: str
    source_citation: str
    standard_state: str


def _cp_integrals(
    temperature_K: float,
    *,
    reference_K: float,
    a: float,
    b: float = 0.0,
    c: float = 0.0,
    d: float = 0.0,
    e: float = 0.0,
) -> tuple[float, float]:
    """Return ∫Cp dT and ∫Cp/T dT for Wu Table 1's cal-based Cp form."""
    t = temperature_K
    t0 = reference_K
    h = (
        a * (t - t0)
        + b * 1.0e-3 / 2.0 * (t * t - t0 * t0)
        + c * 1.0e5 * (-1.0 / t + 1.0 / t0)
        + 2.0 * d * (math.sqrt(t) - math.sqrt(t0))
        + e * 1.0e8 * (-1.0 / (2.0 * t * t) + 1.0 / (2.0 * t0 * t0))
    )
    s = (
        a * math.log(t / t0)
        + b * 1.0e-3 * (t - t0)
        + c * 1.0e5 * (-1.0 / (2.0 * t * t) + 1.0 / (2.0 * t0 * t0))
        + 2.0 * d * (1.0 / math.sqrt(t0) - 1.0 / math.sqrt(t))
        + e * 1.0e8 * (-1.0 / (3.0 * t**3) + 1.0 / (3.0 * t0**3))
    )
    return h, s


def _wu_liquid_gibbs_j_mol(oxide: str, temperature_K: float) -> float:
    """Wu Table 1 liquid G° from H=A+∫Cp dT and S=B+∫Cp/T dT."""
    t0 = 298.15
    if oxide == "MgO":
        a_h, b_s = -130340.58, 6.4541207
        coefficients = (17.398557, -0.7510000, 1.2494063, -70.793260, 0.013968958)
    elif oxide == "SiO2" and temperature_K < 1996.0:
        a_h, b_s = -214339.36, 12.148448
        coefficients = (19.960229, 0.0, -5.8684512, -89.553776, 0.66938861)
    elif oxide == "SiO2":
        a_h, b_s = -221471.21, 2.3702523
        coefficients = (20.5, 0.0, 0.0, 0.0, 0.0)
    else:
        raise ValueError(f"no Wu liquid standard for {oxide!r}")
    h_integral, s_integral = _cp_integrals(
        temperature_K,
        reference_K=t0,
        a=coefficients[0], b=coefficients[1], c=coefficients[2],
        d=coefficients[3], e=coefficients[4],
    )
    # Table 1 reports A and B, Cp, and the conversion 1 cal = 4.184 J.
    # H−TS is in cal/mol until this final conversion to J/mol.
    return (a_h + h_integral - temperature_K * (b_s + s_integral)) * 4.184


def _openimcc_liquid_gibbs_j_mol(temperature_K: float) -> tuple[float, float, str]:
    """Read loaded OpenIMCC liquid references and their condensate identity."""
    # Reuse the loaded condensate standard implementation. Importing here
    # keeps the gas extra optional for normal ``import openimcc`` callers.
    from openimcc.gas import load_gas_datapack, species_thermo

    standards = load_gas_datapack()
    return (
        species_thermo("MgO", "l", temperature_K, standards).G_J_mol,
        species_thermo("SiO2", "l", temperature_K, standards).G_J_mol,
        standards.condensate_table_digest,
    )


def _pair_fractions(x_sio2: float, temperature_K: float) -> tuple[float, float, float, float, float]:
    """Return (X_MgMg, X_MgSi, X_SiSi, Y_Si, Δg) at pair equilibrium."""
    x_mgo = 1.0 - x_sio2
    coord_sum = _Z_MGO * x_mgo + _Z_SIO2 * x_sio2
    y_mgo = _Z_MGO * x_mgo / coord_sum
    y_sio2 = _Z_SIO2 * x_sio2 / coord_sum

    # Wu Eqs. (7)–(8), p. 30: omega and eta are functions of YSi and
    # Δg=omega−eta*T. omega is J/mol of pair reaction; eta is J/(mol K).
    omega = -86090.0 - 48974.0 * y_sio2 + 328109.0 * y_sio2**7
    eta = -37.656 * y_sio2 + 125.52 * y_sio2**7
    delta_g = omega - eta * temperature_K

    if x_mgo == 0.0 or x_sio2 == 0.0:
        q = 0.0
        x_mgm, x_sisi = y_mgo, y_sio2
    else:
        # Premise: Wu Eq. (2), p. 27, gives the pair reaction equilibrium;
        # with X_MgSi=2q, X_MgMg=YMg−q, X_SiSi=YSi−q and Z=2, it reduces to
        # q²/[(YMg−q)(YSi−q)] = exp(−Δg/RT). Log form avoids overflow and the
        # bisection is in the feasible interval [0,min(YMg,YSi)].
        target = -delta_g / (_R * temperature_K)
        lower, upper = 0.0, min(y_mgo, y_sio2)
        cancellation_collapsed = False
        for _ in range(100):
            q = (lower + upper) / 2.0
            if q <= 0.0:
                value = -math.inf
            else:
                mg_pairs = y_mgo - q
                si_pairs = y_sio2 - q
                if mg_pairs <= 0.0 or si_pairs <= 0.0:
                    cancellation_collapsed = True
                    break
                value = (
                    2.0 * math.log(q)
                    - math.log(mg_pairs)
                    - math.log(si_pairs)
                )
            if value < target:
                lower = q
            else:
                upper = q
        if cancellation_collapsed:
            # Let r=min(YMg,YSi)-q. The other pair remainder is
            # |YMg−YSi|+r, so the log quotient is
            # 2 ln(q)−ln(r)−ln(|YMg−YSi|+r). Its log arguments are
            # dimensionless, as is the target -Δg/(RT), since Δg and RT
            # are both J/mol. As r→0+, -ln(r)→+∞; as r→min(Y), q→0
            # and the quotient tends to -∞, keeping the root bracketed
            # inside 0<r<min(Y).
            smaller_y = min(y_mgo, y_sio2)
            larger_gap = abs(y_mgo - y_sio2)
            lower_remainder, upper_remainder = 0.0, smaller_y
            for _ in range(100):
                remainder = (lower_remainder + upper_remainder) / 2.0
                q = smaller_y - remainder
                if remainder <= 0.0:
                    value = math.inf
                elif q <= 0.0:
                    value = -math.inf
                else:
                    value = (
                        2.0 * math.log(q)
                        - math.log(remainder)
                        - math.log(larger_gap + remainder)
                    )
                if value > target:
                    lower_remainder = remainder
                else:
                    upper_remainder = remainder
            remainder = (lower_remainder + upper_remainder) / 2.0
            q = smaller_y - remainder
            if y_mgo <= y_sio2:
                x_mgm, x_sisi = remainder, larger_gap + remainder
            else:
                x_mgm, x_sisi = larger_gap + remainder, remainder
        else:
            q = (lower + upper) / 2.0
            x_mgm, x_sisi = y_mgo - q, y_sio2 - q

    x_mgsi = 2.0 * q
    return x_mgm, x_mgsi, x_sisi, y_sio2, delta_g


def _xlog_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if numerator <= 0.0 else numerator * math.log(numerator / denominator)


_EDGE_SERIES_SWITCH = 1.0e-8


def _mixing_excess_edge_series_j_mol(
    x_sio2: float, temperature_K: float
) -> tuple[float, float, float, float, float] | None:
    """Return (gE, dgE/dx, d2gE/dx2, muMgE, muSiE) near either edge."""
    if 0.0 < x_sio2 < _EDGE_SERIES_SWITCH:
        minority_oxide_fraction = x_sio2
        delta_0 = -86090.0
        d_delta_dy_0 = -48974.0 + 37.656 * temperature_K
        rt = _R * temperature_K
        pair_constant = math.exp(-delta_0 / rt)
        # With minority site fraction u, the pair equilibrium gives the
        # minority homopair fraction r=u^2/K+O(u^3), K=exp(-Delta/RT).
        # The pair excess entropy is u^2*(1-(1+ln K)/K)+O(u^3). Substituting
        # q=u-r into Npair*q*Delta plus RT*Npair*S_pair, then converting from
        # site fraction u to oxide fraction x (u=zSi*x/zMg+O(x^2)), gives
        # gE=A*x+B*x^2+O(x^3), where A=zSi*Delta0/2 and
        # B=zSi^2/(2*zMg)*(Delta'_0-Delta0/K+RT*(1-(1+ln K)/K)).
        # Delta and RT are J/mol, so A and B are J/mol. At x=1e-8 the
        # regular derivative and this expansion differ by less than 1 J/mol
        # over the candidate's 1200-3000 K interval. The sanity limits are
        # gE(0)=0 and dgE/dx(0)=zSi*Delta0/2, the insertion-potential limit.
        a = _Z_SIO2 * delta_0 / 2.0
        b = (_Z_SIO2 * _Z_SIO2 / (2.0 * _Z_MGO)) * (
            d_delta_dy_0 - delta_0 / pair_constant
            + rt * (1.0 - (1.0 + math.log(pair_constant)) / pair_constant)
        )
        g_excess = a * minority_oxide_fraction + b * minority_oxide_fraction**2
        dg_excess = a + 2.0 * b * minority_oxide_fraction
        mu_mgo = -b * minority_oxide_fraction**2
        mu_sio2 = dg_excess - b * minority_oxide_fraction**2
        return g_excess, dg_excess, 2.0 * b, mu_mgo, mu_sio2

    if 0.0 < 1.0 - x_sio2 < _EDGE_SERIES_SWITCH:
        minority_oxide_fraction = 1.0 - x_sio2
        delta_1 = 193045.0 - 87.864 * temperature_K
        d_delta_dy_1 = 2247789.0 - 840.984 * temperature_K
        rt = _R * temperature_K
        pair_constant = math.exp(-delta_1 / rt)
        # At the silica-rich edge the minority site fraction is MgO,
        # u=1-YSi=zMg*(1-x)/zSi+O((1-x)^2). The pair balance again gives
        # r=u^2/K+O(u^3); after conversion to oxide fraction t=1-x,
        # gE=A*t+B*t^2+O(t^3). Here A=zMg*Delta1/2 and
        # B=zMg^2/(2*zSi)*(-Delta'_1-Delta1/K+RT*(1-(1+ln K)/K));
        # the minus sign converts dDelta/dYSi to dDelta/du. Units remain
        # J/mol, and gE(0)=0 with the MgO insertion-potential limit preserved.
        a = _Z_MGO * delta_1 / 2.0
        b = (_Z_MGO * _Z_MGO / (2.0 * _Z_SIO2)) * (
            -d_delta_dy_1 - delta_1 / pair_constant
            + rt * (1.0 - (1.0 + math.log(pair_constant)) / pair_constant)
        )
        g_excess = a * minority_oxide_fraction + b * minority_oxide_fraction**2
        dg_excess = -a - 2.0 * b * minority_oxide_fraction
        mu_mgo = -dg_excess - b * minority_oxide_fraction**2
        mu_sio2 = -b * minority_oxide_fraction**2
        return g_excess, dg_excess, 2.0 * b, mu_mgo, mu_sio2

    return None


def _mixing_excess_gibbs_j_mol(
    x_sio2: float, temperature_K: float
) -> float:
    """Return minimized Wu excess Gmix in J per mole of oxide units."""
    edge_state = _mixing_excess_edge_series_j_mol(x_sio2, temperature_K)
    if edge_state is not None:
        return edge_state[0]
    if x_sio2 <= 0.0 or x_sio2 >= 1.0:
        return 0.0

    x_mgo = 1.0 - x_sio2
    x_mgm, x_mgsi, x_sisi, _y_sio2, delta_g = _pair_fractions(
        x_sio2, temperature_K
    )
    coord_sum = _Z_MGO * x_mgo + _Z_SIO2 * x_sio2
    y_mgo = _Z_MGO * x_mgo / coord_sum
    y_sio2 = _Z_SIO2 * x_sio2 / coord_sum
    pair_moles = coord_sum / 2.0
    pair_entropy = math.fsum((
        _xlog_ratio(x_mgm, y_mgo * y_mgo),
        _xlog_ratio(x_sisi, y_sio2 * y_sio2),
        _xlog_ratio(x_mgsi, 2.0 * y_mgo * y_sio2),
    ))
    # These are the excess terms already separated from the free-oxide ideal
    # baseline; forming total Gmix and subtracting that baseline loses low-y
    # precision. Wu Eqs. (3)–(5): Npair*q*Delta is in J/mol.
    return _R * temperature_K * pair_moles * pair_entropy + pair_moles * (
        x_mgsi / 2.0
    ) * delta_g


def _mixing_gibbs_j_mol(x_sio2: float, temperature_K: float) -> float:
    """Minimized Wu binary Gmix in J per mole of oxide formula units."""
    x_mgo = 1.0 - x_sio2
    ideal = _R * temperature_K * (
        (x_mgo * math.log(x_mgo) if x_mgo > 0.0 else 0.0)
        + (x_sio2 * math.log(x_sio2) if x_sio2 > 0.0 else 0.0)
    )
    return math.fsum((ideal, _mixing_excess_gibbs_j_mol(x_sio2, temperature_K)))


def _mixing_enthalpy_j_mol(x_sio2: float, temperature_K: float) -> float:
    """Return the Wu binary mixing enthalpy at fixed oxide composition."""
    x_mgo = 1.0 - x_sio2
    _x_mgm, x_mgsi, _x_sisi, y_sio2, _delta_g = _pair_fractions(
        x_sio2, temperature_K
    )
    pair_moles = (_Z_MGO * x_mgo + _Z_SIO2 * x_sio2) / 2.0
    omega = -86090.0 - 48974.0 * y_sio2 + 328109.0 * y_sio2**7
    # The pair configurational term is entropic. At the minimized pair state,
    # the envelope theorem leaves Hmix=Npair*q*omega, with q=X_MgSi/2;
    # units are J per mole of oxide formula units.
    return pair_moles * (x_mgsi / 2.0) * omega


def _mixing_gibbs_derivative_j_mol(
    x_sio2: float,
    temperature_K: float,
) -> float:
    """Return dGmix/dx from the analytic excess derivative and ideal slope."""
    if x_sio2 <= 0.0 or x_sio2 >= 1.0:
        raise ValueError("binary Gibbs derivative requires an interior composition")
    return _mixing_excess_gibbs_derivative_j_mol(
        x_sio2, temperature_K
    ) + _R * temperature_K * (
        math.log(x_sio2) - math.log1p(-x_sio2)
    )


def _mixing_excess_gibbs_derivative_j_mol(
    x_sio2: float,
    temperature_K: float,
) -> float:
    """Return the analytic derivative of Wu's excess mixing Gibbs energy."""
    if x_sio2 <= 0.0:
        return _Z_SIO2 * -86090.0 / 2.0
    if x_sio2 >= 1.0:
        delta_g_at_silica = 193045.0 - 87.864 * temperature_K
        return -_Z_MGO * delta_g_at_silica / 2.0
    edge_state = _mixing_excess_edge_series_j_mol(x_sio2, temperature_K)
    if edge_state is not None:
        return edge_state[1]

    p_mgm, p_mgsi, p_sisi, y_sio2, delta_g = _pair_fractions(
        x_sio2, temperature_K
    )
    x_mgo = 1.0 - x_sio2
    coord_sum = _Z_MGO * x_mgo + _Z_SIO2 * x_sio2
    dcoord_dx = _Z_SIO2 - _Z_MGO
    y_mgo = 1.0 - y_sio2
    dy_sio2_dx = _Z_MGO * _Z_SIO2 / (coord_sum * coord_sum)
    dy_mgo_dx = -dy_sio2_dx
    pair_moles = coord_sum / 2.0
    pair_entropy = math.fsum((
        _xlog_ratio(p_mgm, y_mgo * y_mgo),
        _xlog_ratio(p_sisi, y_sio2 * y_sio2),
        _xlog_ratio(p_mgsi, 2.0 * y_mgo * y_sio2),
    ))
    d_pair_entropy_dx = math.fsum((
        dy_mgo_dx * (math.log(p_mgm / (y_mgo * y_mgo)) + 1.0)
        - 2.0 * p_mgm * dy_mgo_dx / y_mgo,
        dy_sio2_dx * (math.log(p_sisi / (y_sio2 * y_sio2)) + 1.0)
        - 2.0 * p_sisi * dy_sio2_dx / y_sio2,
        -p_mgsi * (dy_mgo_dx / y_mgo + dy_sio2_dx / y_sio2),
    ))
    d_omega_dy = -48974.0 + 7.0 * 328109.0 * y_sio2**6
    d_eta_dy = -37.656 + 7.0 * 125.52 * y_sio2**6
    d_delta_dy = d_omega_dy - temperature_K * d_eta_dy
    q = p_mgsi / 2.0
    d_excess_dx = (
        _R * temperature_K * (
            dcoord_dx / 2.0 * pair_entropy
            + pair_moles * d_pair_entropy_dx
        )
        + dcoord_dx / 2.0 * q * delta_g
        + pair_moles * q * d_delta_dy * dy_sio2_dx
    )
    return d_excess_dx


def _lower_hull(points: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    hull: list[tuple[int, float, float]] = []
    for point in points:
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            cross = (
                (second[1] - first[1]) * (point[2] - first[2])
                - (second[2] - first[2]) * (point[1] - first[1])
            )
            if cross > 0.0:
                break
            hull.pop()
        hull.append(point)
    return hull


def _two_liquid_bounds(temperature_K: float) -> tuple[float, float] | None:
    count = round(1.0 / _BINODAL_STEP)
    points = [
        (index, index * _BINODAL_STEP,
         _mixing_gibbs_j_mol(index * _BINODAL_STEP, temperature_K))
        for index in range(count + 1)
    ]
    hull = _lower_hull(points)
    gaps = [
        (left[1], right[1])
        for left, right in zip(hull, hull[1:])
        if right[1] - left[1] > 2.0 * _BINODAL_STEP
    ]
    if not gaps:
        return None
    return min(bounds[0] for bounds in gaps), max(bounds[1] for bounds in gaps)


def evaluate_mgo_sio2_mqm(
    composition_moles: Mapping[str, float], T_K: float
) -> MgOSiO2MQMResult:
    """Evaluate the EXPERIMENTAL Wu 1993 MgO–SiO2 binary liquid candidate.

    ``composition_moles`` contains MgO and SiO2 formula-unit mole amounts.
    Positive amounts of any other oxide are refused. The supported temperature
    interval is 1200–3000 K: it is inside Wu Table 1's liquid standard ranges
    and the intersection where OpenIMCC's default MgO(l)/SiO2(l) references
    are available. The result reports activities on those OpenIMCC standards.

    Returned activities are derivatives of the single-phase Gibbs curve,
    including inside the estimated two-liquid gap. For example, at 1960 K and
    x_SiO2=0.7, a(SiO2) is about 1.115 versus about 0.985 at the binodal. The
    standard shift preserves Wu's chemical potential, so pure-end activities
    equal exp((G°_Wu−G°_open)/(RT)); pure MgO at 1873.15 K has a(MgO) about
    1.076 with the packaged standards. Below 2200 K the packaged MgO(l) row is
    the supercooled continuation Mg-009-SC-CP.

    This evaluator is binary-only. It does not model CaO, FeO, Al2O3, Na2O or
    other regolith/basalt components and does not imply an embedding rule.
    """
    if not isinstance(composition_moles, Mapping):
        raise ImccCompositionIncompleteError("composition_moles must be a mapping")
    if not is_declared_real_scalar(T_K, allow_numeric_str=True):
        raise ImccTOutsideDatapackDomainError(
            "temperature must be a real number"
        )
    try:
        temperature = float(T_K)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ImccTOutsideDatapackDomainError("temperature must be finite") from exc
    if not math.isfinite(temperature) or not _T_MIN_K <= temperature <= _T_MAX_K:
        raise ImccTOutsideDatapackDomainError(
            f"temperature {temperature!r} K is outside the candidate reference "
            f"range [{_T_MIN_K:g}, {_T_MAX_K:g}] K"
        )

    amounts: dict[str, float] = {}
    unknown_positive: list[str] = []
    for name, raw in composition_moles.items():
        if not isinstance(name, str):
            raise ImccCompositionIncompleteError("oxide names must be strings")
        if not is_declared_real_scalar(raw, allow_numeric_str=True):
            raise ImccCompositionIncompleteError(
                f"moles for {name!r} must be finite and nonnegative"
            )
        try:
            amount = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ImccCompositionIncompleteError(
                f"moles for {name!r} must be finite and nonnegative"
            ) from exc
        if not math.isfinite(amount) or amount < 0.0:
            raise ImccCompositionIncompleteError(
                f"moles for {name!r} must be finite and nonnegative"
            )
        if name not in {"MgO", "SiO2"}:
            if amount > 0.0:
                unknown_positive.append(name)
            continue
        amounts[name] = amount
    if unknown_positive:
        raise ImccComponentOutsideDomainError(
            f"MgO–SiO2 MQM candidate refuses other positive oxides: "
            f"{sorted(unknown_positive)}"
        )
    total = math.fsum(amounts.values())
    if total <= 0.0:
        raise ImccCompositionIncompleteError(
            "composition must contain positive MgO or SiO2 moles"
        )
    x_sio2 = amounts.get("SiO2", 0.0) / total
    x_mgo = 1.0 - x_sio2

    pair_mgm, pair_mgsi, pair_sisi, _y_sio2, _delta_g = _pair_fractions(
        x_sio2, temperature
    )
    mix = _mixing_gibbs_j_mol(x_sio2, temperature)
    if x_sio2 == 0.0:
        ln_a_sio2_wu, ln_a_mgo_wu = -math.inf, 0.0
    elif x_sio2 == 1.0:
        ln_a_sio2_wu, ln_a_mgo_wu = 0.0, -math.inf
    else:
        # Check that both oxide fractions retain enough local precision to
        # report activities, since 1−xSiO2 loses resolution near the endpoint.
        h = min(_ACTIVITY_STEP, x_sio2 / 2.0, x_mgo / 2.0)
        x_mgo_center = amounts.get("MgO", 0.0) / total
        x_sio2_plus = x_sio2 + h
        x_sio2_minus = x_sio2 - h
        x_mgo_plus = 1.0 - x_sio2_plus
        x_mgo_minus = 1.0 - x_sio2_minus
        recovered_steps = (
            x_mgo_center - x_mgo_plus,
            x_mgo_minus - x_mgo_center,
        )
        relative_step_error = max(
            abs(recovered_step - h) / h for recovered_step in recovered_steps
        )
        if relative_step_error > _ACTIVITY_STENCIL_REL_TOL:
            raise ImccComponentOutsideDomainError(
                "Wu 1993 MgO–SiO2 MQM activity evaluation refused; violated axis: "
                "composition_mole_fraction; local fraction perturbations are "
                f"distorted by {relative_step_error:.1%} by float rounding, "
                f"above the {_ACTIVITY_STENCIL_REL_TOL:.0%} tolerance"
            )
        derivative = _mixing_gibbs_derivative_j_mol(x_sio2, temperature)
        # Premise: Wu Eqs. (2)–(8), pp. 27, 30, minimize Gmix at fixed
        # analytical composition. For g=Gmix/(nMgO+nSiO2), Euler's relation
        # gives μSi,mix=g+(1−x)dg/dx and μMg,mix=g−x*dg/dx, both J/mol.
        # Dividing by RT (J/mol) yields dimensionless ln activities.
        ln_a_sio2_wu = (mix + x_mgo * derivative) / (_R * temperature)
        ln_a_mgo_wu = (mix - x_sio2 * derivative) / (_R * temperature)

    # Premise: Wu Table 1, p. 28 and the loaded OpenIMCC condensate table
    # both express G° relative to elements. Changing the reference from Wu to
    # OpenIMCC therefore gives ln(a_open)=ln(a_Wu)+(G°_Wu−G°_open)/(RT).
    # ΔG is J/mol and RT is J/mol, so the shift is dimensionless. At 1873 K
    # the SiO2 shift is about −0.00039 dex and the MgO shift about +0.0320 dex;
    # these are sanity checks against the sign and magnitude of the conversion.
    open_g_mgo, open_g_sio2, condensate_digest = _openimcc_liquid_gibbs_j_mol(
        temperature
    )
    shift_mgo = (
        _wu_liquid_gibbs_j_mol("MgO", temperature) - open_g_mgo
    ) / (_R * temperature)
    shift_sio2 = (
        _wu_liquid_gibbs_j_mol("SiO2", temperature) - open_g_sio2
    ) / (_R * temperature)
    a_sio2 = math.exp(ln_a_sio2_wu + shift_sio2) if math.isfinite(ln_a_sio2_wu) else 0.0
    a_mgo = math.exp(ln_a_mgo_wu + shift_mgo) if math.isfinite(ln_a_mgo_wu) else 0.0

    bounds = _two_liquid_bounds(temperature)
    in_gap = (
        bounds is not None
        and bounds[0] + _BINODAL_STEP < x_sio2 < bounds[1] - _BINODAL_STEP
    )
    return MgOSiO2MQMResult(
        a_sio2=a_sio2,
        a_mgo=a_mgo,
        x_sio2=x_sio2,
        temperature_K=temperature,
        pair_fractions=(pair_mgm, pair_mgsi, pair_sisi),
        stability="inside_two_liquid_field" if in_gap else "homogeneous",
        two_liquid_field_x_bounds=bounds,
        candidate_id=_CANDIDATE_ID,
        experimental_status="EXPERIMENTAL candidate; binary only",
        source_citation=_SOURCE,
        standard_state=(
            "OpenIMCC loaded pure-liquid MgO(l)/SiO2(l) standards; "
            f"condensate table canonical SHA-256 {condensate_digest}; "
            "converted from Wu Table 1, p. 28"
        ),
    )
