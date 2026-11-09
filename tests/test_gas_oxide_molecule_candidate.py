"""Source checks for the opt-in Si2O2 and AlSiO gas table."""

import csv
import hashlib
import math
from pathlib import Path

import pytest
import yaml

from openimcc import default_gas_channels, evaluate, evaluate_gas
from openimcc.gas import (
    ImccGasTemperatureOutsideDomainError,
    _janaf_gibbs,
    gas_channel_metadata,
    load_gas_datapack,
)
from openimcc.model import load_datapack
from tools import build_gas_tables


ROOT = Path(__file__).resolve().parents[1]
GAS_DIR = ROOT / "src/openimcc/data/gas"
BASE_GAS = GAS_DIR / "gas-shomate.csv"
BASE_OXIDE = GAS_DIR / "condensate.csv"
CANDIDATE_GAS = GAS_DIR / "gas-shomate-oxide-molecule-candidate.csv"
STANDARD_STATE_GAS = GAS_DIR / "gas-shomate-standard-state-candidate.csv"
DEFAULT_GAS = GAS_DIR / "gas-shomate-default-v1.csv"
PROVENANCE = GAS_DIR / "PROVENANCE.yaml"


def _candidate_rows() -> list[dict[str, str]]:
    with CANDIDATE_GAS.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_at(
    rows: list[dict[str, str]], species: str, temperature: float
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["species_name"] == f"{species}(g)"
        and float(row["T_min"]) <= temperature
    ]
    return max(matches, key=lambda row: float(row["T_min"]))


def _numeric_gas_row(row: dict[str, str]) -> dict[str, object]:
    return {
        key: float(value) if key in "ABCDEFGH" else value
        for key, value in row.items()
    }


def test_provenance_file_digests_match_packaged_tables() -> None:
    provenance = yaml.safe_load(PROVENANCE.read_text(encoding="utf-8"))
    file_digests: list[tuple[str, str]] = []

    def collect(value: object) -> None:
        if isinstance(value, dict):
            filename = value.get("file")
            digest = value.get("sha256")
            if isinstance(filename, str) and isinstance(digest, str):
                file_digests.append((filename, digest))
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    collect(provenance)
    assert file_digests
    for filename, expected in file_digests:
        path = GAS_DIR / filename
        assert path.is_file(), filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, filename


def test_candidate_builder_is_reproducible_and_keeps_default_bytes(
    tmp_path: Path,
) -> None:
    rebuilt = tmp_path / CANDIDATE_GAS.name
    assert build_gas_tables.build_oxide_molecule_candidate(BASE_GAS, rebuilt) == 7
    assert rebuilt.read_bytes() == CANDIDATE_GAS.read_bytes()
    assert CANDIDATE_GAS.read_bytes().startswith(BASE_GAS.read_bytes())
    # The default combination has its own raw-byte pins after promotion.
    assert hashlib.sha256(DEFAULT_GAS.read_bytes()).hexdigest() == (
        "7f6f03cfae5f818df95a4a03e4bf2b43b9b1da08795eff8f1550e18620ccef36"
    )
    assert hashlib.sha256((GAS_DIR / "condensate-default-v1.csv").read_bytes()).hexdigest() == (
        "1913f0c5d48b24f61576f75d4ae845f4ddf7943d40d517f62ce8dc1e9af5018b"
    )

    assert hashlib.sha256(BASE_GAS.read_bytes()).hexdigest() == (
        "e01c363ac6bb526b5e8ee6df300963652e0771585e9348e6456afef5cacb8058"
    )
    assert hashlib.sha256(BASE_OXIDE.read_bytes()).hexdigest() == (
        "1b9149eb9d53ffd7d96872b25b27144bca902a71b963fcec9b6bf9cec37127a0"
    )
    default_pack = load_gas_datapack()
    historical_default = load_gas_datapack(
        gas_path=BASE_GAS, oxide_path=BASE_OXIDE
    )
    assert default_pack.gas_table_digest == (
        "4d069a8951cd11e0d2edb37420e3aa04e4927c51c6eee00bbc53368b20d6c492"
    )
    assert historical_default.gas_table_digest == (
        "2efbe131d82a95725f00a725b7ab06ffbad004e223d4f1ff556adbf2153bfc71"
    )
    provenance = yaml.safe_load(PROVENANCE.read_text(encoding="utf-8"))
    candidate = next(
        item for item in provenance["candidate_gas_files"]
        if item["file"] == CANDIDATE_GAS.name
    )
    assert candidate["sha256"] == hashlib.sha256(
        CANDIDATE_GAS.read_bytes()
    ).hexdigest()


def test_default_combination_builder_applies_only_the_intended_gas_changes(
    tmp_path: Path,
) -> None:
    rebuilt = tmp_path / DEFAULT_GAS.name
    counts = build_gas_tables.build_default_combination_gas(
        BASE_GAS, STANDARD_STATE_GAS, CANDIDATE_GAS, rebuilt
    )

    assert counts == {
        "base_rows": 678,
        "corrected_rows": 54,
        "corrected_species": 27,
        "added_rows": 7,
    }
    assert rebuilt.read_bytes() == DEFAULT_GAS.read_bytes()
    assert DEFAULT_GAS.read_bytes() == CANDIDATE_GAS.read_bytes().replace(
        BASE_GAS.read_bytes(), STANDARD_STATE_GAS.read_bytes(), 1
    )


def test_published_functions_reproduce_printed_source_anchors() -> None:
    rows = _candidate_rows()
    gas_pack = load_gas_datapack(gas_path=CANDIDATE_GAS, oxide_path=BASE_OXIDE)
    melt_pack = load_datapack()
    gas_constant = 8.3144

    def molecular_gibbs(species: str, temperature: float) -> float:
        if species == "Si2O2":
            # NIST JPCRD Table 2: q=-(G-H°298)/(RT), with the printed
            # high-temperature coefficient -0.040 × 10^-12 K^-4.
            a, b, c, d, e = (33.340, 10.626e-3, -2.860e-6, 0.509e-9, -0.040e-12)
            q = a + b * temperature + c * temperature**2 + d * temperature**3 + e * temperature**4
            return (
                -43.22e3 * gas_constant
                - gas_constant * temperature * q
                - gas_constant * temperature * math.log(1.01325)
            )

        # NASA TN D-7123 Table VIII (2000 and 2100 K): q0 and H-H0 are
        # printed to 0.01 J/(mol K) and 0.01 kJ/mol. Table XIII gives
        # ΔfH°298=19.8 kJ/mol. The report's molecular partition function is
        # at 1 atm, so convert the molecule's entropy to the table's 1 bar.
        q0, h_at_t = {2000.0: (315.44, 104.35), 2100.0: (317.99, 110.11)}[temperature]
        h_at_298 = 11.058
        entropy = q0 + h_at_t * 1000.0 / temperature
        return (
            19.8e3 + (h_at_t - h_at_298) * 1000.0
            - temperature * entropy
            - gas_constant * temperature * math.log(1.01325)
        )

    def reaction_regression(
        species: str, temperature: float, composition: dict[str, float],
        reactants: tuple[str, ...],
    ) -> None:
        melt = evaluate(
            composition, temperature, melt_pack, basis_type="mol",
            allow_extrapolation=True, allow_out_of_envelope=True,
        )
        activities = {
            oxide: float(melt.parent_activity[index])
            for index, oxide in enumerate(melt_pack.parent_oxides)
            if oxide in composition
        }
        pressures = evaluate_gas(
            activities, temperature, 1.0e-10, gas_pack,
            parent_oxides=tuple(composition), gas_species=(species, *reactants),
            allow_extrapolation=True,
        )
        expected_gibbs = molecular_gibbs(species, temperature)
        product_row = _numeric_gas_row(_row_at(rows, species, temperature))
        # This matches the existing SiO/Al molecular-row convention. A row
        # fitted after subtracting elemental thermal functions fails here.
        assert _janaf_gibbs(temperature, product_row) == pytest.approx(
            expected_gibbs, abs=12.0
        )

        reactant_gibbs = 0.0
        for reactant in reactants:
            reactant_rows = gas_pack.gas_df.loc[f"{reactant}(g)"]
            interval = reactant_rows[
                (reactant_rows["T_min"] <= temperature)
                & (reactant_rows["T_max"] >= temperature)
            ].iloc[0]
            reactant_gibbs += _janaf_gibbs(temperature, interval)
        equilibrium_constant = math.exp(
            -(expected_gibbs - reactant_gibbs) / (gas_constant * temperature)
        )
        reactant_pressure = math.prod(float(pressures[name]) for name in reactants)
        expected_pressure = equilibrium_constant * reactant_pressure
        # The source constants are printed to 3–5 significant figures and the
        # Shomate fits add a small interpolation error; 0.05% covers both while
        # detecting a wrong reactant interval or molecular standard pressure.
        assert float(pressures[species]) == pytest.approx(expected_pressure, rel=0.0005)

    wollastonite = {"CaO": 0.5, "SiO2": 0.5}
    for temperature in (1900.0, 2121.0):
        reaction_regression("Si2O2", temperature, wollastonite, ("SiO", "SiO"))

    mullite = {"Al2O3": 0.6, "SiO2": 0.4}
    for temperature in (2000.0, 2100.0):
        reaction_regression("AlSiO", temperature, mullite, ("Al", "SiO"))


def test_default_table_selects_both_channels_and_flags_alsio_extrapolation() -> None:
    default_pack = load_gas_datapack()
    candidate_pack = load_gas_datapack(
        gas_path=CANDIDATE_GAS, oxide_path=BASE_OXIDE
    )
    historical_pack = load_gas_datapack(gas_path=BASE_GAS, oxide_path=BASE_OXIDE)
    parents = ("SiO2", "Al2O3")
    default_channels, default_omitted = default_gas_channels(parents, default_pack)
    candidate_channels, _candidate_omitted = default_gas_channels(
        parents, candidate_pack
    )
    historical_channels, _historical_omitted = default_gas_channels(
        parents, historical_pack
    )
    default_names = {name for name, _reaction in default_channels}
    candidate_names = {name for name, _reaction in candidate_channels}
    historical_names = {name for name, _reaction in historical_channels}
    assert {"Si2O2", "AlSiO"} <= default_names
    assert not {"Si2O2", "AlSiO"} & set(default_omitted)
    assert {"Si2O2", "AlSiO"} <= candidate_names
    assert not {"Si2O2", "AlSiO"} & historical_names

    metadata = gas_channel_metadata(
        {"Al2O3": 0.6, "SiO2": 0.4}, 1800.0, candidate_pack
    )["AlSiO"]
    assert metadata.available
    assert metadata.parent_oxygen_demand == pytest.approx(3.5)
    assert metadata.neutral_pO2_exponent == pytest.approx(-1.25)

    activities = {"SiO2": 0.4, "Al2O3": 0.6}
    result = evaluate_gas(
        activities, 1800.0, 1.0e-10, candidate_pack,
        parent_oxides=parents, gas_species=("Si2O2", "AlSiO"),
    )
    assert set(result) == {"Si2O2", "AlSiO"}
    assert all(math.isfinite(float(value)) and value > 0.0 for value in result.values())

    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        evaluate_gas(
            activities, 2473.0, 1.0e-10, candidate_pack,
            parent_oxides=parents, gas_species=("AlSiO",),
            allow_extrapolation=False,
        )
    extrapolated = evaluate_gas(
        activities, 2473.0, 1.0e-10, candidate_pack,
        parent_oxides=parents, gas_species=("AlSiO",),
        allow_extrapolation=True,
    )
    assert "AlSiO" in str(extrapolated.domain_flags)
    assert "2200" in str(extrapolated.domain_flags)


def test_kems_input_smoke_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Print matched-input predictions beside measurements; assert no agreement."""
    gas_pack = load_gas_datapack(gas_path=CANDIDATE_GAS, oxide_path=BASE_OXIDE)
    melt_pack = load_datapack()
    smoke_rows = (
        # Printed KEMS pressures: hatrurite Table 2 and mullite Table 2;
        # source reports atm. Convert only for the side-by-side display.
        ("Shornikov 2026 hatrurite", "Si2O2", 2121.0,
         {"CaO": 0.75, "SiO2": 0.25}, 1.68e-11),
        ("Shults et al. 1994 mullite", "AlSiO", 1933.0,
         {"Al2O3": 0.6, "SiO2": 0.4}, 0.341e-10),
        ("Shults et al. 1994 mullite", "AlSiO", 2033.0,
         {"Al2O3": 0.6, "SiO2": 0.4}, 5.34e-10),
    )
    rows = []
    for source, species, temperature, composition, printed_atm in smoke_rows:
        melt = evaluate(
            composition, temperature, melt_pack, basis_type="mol",
            allow_extrapolation=True, allow_out_of_envelope=True,
        )
        activities = {
            oxide: float(melt.parent_activity[index])
            for index, oxide in enumerate(melt_pack.parent_oxides)
            if oxide in composition
        }
        pressure = evaluate_gas(
            activities, temperature, 1.0e-10, gas_pack,
            parent_oxides=tuple(composition), gas_species=(species,),
            allow_extrapolation=True,
        )[species]
        assert math.isfinite(float(pressure)) and pressure > 0.0
        rows.append((source, species, temperature, composition, float(pressure), printed_atm * 1.01325))

    print("KEMS smoke rows; predicted at diagnostic pO2=1e-10 bar; printed p in atm converted to bar:")
    for source, species, temperature, composition, pressure, printed_bar in rows:
        print(
            f"{source}\t{species}\tT={temperature:g} K\t"
            f"x={composition}\tp_model={pressure:.6g} bar\t"
            f"p_printed={printed_bar:.6g} bar"
        )
    report = capsys.readouterr().out
    print(report, end="")
    assert "KEMS smoke rows" in report
