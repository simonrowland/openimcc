"""Named, immutable selections of the published SF04 and openimcc packs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from openimcc.kernel import ImccRefusal
from openimcc.model import ImccLoadedDatapack, load_datapack

if TYPE_CHECKING:
    from openimcc.gas import ImccGasDatapack
    from openimcc.redox_pack import RedoxPack


_DATA_DIR = Path(__file__).parent / "data"
_PACKS_DIR = _DATA_DIR / "packs"


class ProfileError(ImccRefusal):
    """A requested pack profile cannot be used as specified."""

    code = "profile_error"


class ProfileConflictError(ProfileError):
    """A profile and explicit pack arguments were supplied together."""

    code = "profile_pack_conflict"


class ProfileRedoxUnavailableError(ProfileError):
    """The published SF04 records do not define a redox model."""

    code = "profile_redox_unavailable"


@dataclass(frozen=True)
class Profile:
    """A loaded selection of melt, gas, and redox data with its provenance."""

    name: str
    melt_pack: ImccLoadedDatapack
    gas_datapack: ImccGasDatapack
    redox_pack_path: Path | None
    provenance: str

    @cached_property
    def redox_pack(self) -> RedoxPack:
        """Return the redox pack selected by this profile, loaded once."""
        if self.redox_pack_path is None:
            raise ProfileRedoxUnavailableError(
                "the SF04 publication has no redox model; use "
                "profile='openimcc' or pass explicit packs"
            )
        from openimcc.redox_pack import load_redox_pack

        return load_redox_pack(self.redox_pack_path)


def _candidate_profile(name: str, filename: str, provenance: str) -> Profile:
    from openimcc.gas import load_gas_datapack

    gas = load_gas_datapack(
        gas_path=_DATA_DIR / "gas" / "gas-shomate-default-v2.csv",
        oxide_path=_DATA_DIR / "gas" / "condensate-default-v1.csv",
    )
    return Profile(
        name=name,
        melt_pack=load_datapack(_PACKS_DIR / filename),
        gas_datapack=gas,
        redox_pack_path=_PACKS_DIR / "openimcc-redox-v6.json",
        provenance=provenance,
    )


def load_profile(name: str) -> Profile:
    """Load an openimcc, candidate, or published SF04 profile.

    The returned frozen object can be reused across evaluations without
    reloading its packs.
    """
    if name == "openimcc":
        from openimcc.gas import load_gas_datapack

        gas = load_gas_datapack(
            gas_path=_DATA_DIR / "gas" / "gas-shomate-default-v2.csv",
            oxide_path=_DATA_DIR / "gas" / "condensate-default-v1.csv",
        )
        return Profile(
            name="openimcc",
            melt_pack=load_datapack(),
            gas_datapack=gas,
            redox_pack_path=_PACKS_DIR / "openimcc-redox-v6.json",
            provenance=(
                "D066 v4 melt, versioned openimcc gas and condensate tables, "
                "and the v6 Cr, Sr, Li, Zn, and Yb redox pack."
            ),
        )
    if name in {"sr-candidate", "sr-candidate-v1"}:
        pack_version = "v2" if name == "sr-candidate" else "v1"
        return _candidate_profile(
            name,
            f"imcc-sf04-d066-sr-candidate-{pack_version}.json",
            (
                "Opt-in provisional SrO full-component candidate melt pack, "
                "versioned openimcc gas and condensate tables, and the v6 "
                "Cr, Sr, Li, Zn, and Yb redox pack."
            ),
        )
    if name == "p2o5-candidate":
        return _candidate_profile(
            name,
            "imcc-sf04-d066-p2o5-candidate-v1.json",
            (
                "Opt-in provisional P2O5 full-component candidate melt pack, "
                "versioned openimcc gas and condensate tables, and the v6 "
                "Cr, Sr, Li, Zn, and Yb redox pack."
            ),
        )
    if name == "cao-aluminate-candidate":
        return _candidate_profile(
            name,
            "imcc-sf04-d066-cao-aluminate-candidate-v1.json",
            (
                "Opt-in CaO–Al2O3 associate activity-fit candidate melt pack, "
                "versioned openimcc gas and condensate tables, and the v6 "
                "Cr, Sr, Li, Zn, and Yb redox pack."
            ),
        )
    if name == "silicate-candidate":
        return _candidate_profile(
            name,
            "imcc-sf04-d066-silicate-candidate-v2.json",
            (
                "Opt-in provisional Mg-silicate activity-fit candidate "
                "melt pack, versioned openimcc gas and condensate tables, and "
                "the v6 Cr, Sr, Li, Zn, and Yb redox pack."
            ),
        )
    if name == "published":
        from openimcc.gas import load_gas_datapack

        gas = load_gas_datapack(
            gas_path=_DATA_DIR / "gas" / "gas-shomate.csv",
            oxide_path=_PACKS_DIR / "sf04-published" / "condensate.csv",
        )
        return Profile(
            name="published",
            melt_pack=load_datapack(_PACKS_DIR / "imcc-sf04-v1.0.2.json"),
            gas_datapack=gas,
            redox_pack_path=None,
            provenance=(
                "Published SF04 melt records and the published-comparison gas "
                "tables: shared JANAF gas data with LAM1984/1987 condensates; "
                "no published redox model."
            ),
        )
    raise ProfileError(
        f"unknown profile {name!r}; expected 'openimcc', 'published', or "
        "'sr-candidate', 'sr-candidate-v1', 'p2o5-candidate', or "
        "'cao-aluminate-candidate', or 'silicate-candidate'"
    )


def _resolve_profile(profile: str | Profile) -> Profile:
    if isinstance(profile, Profile):
        return profile
    if not isinstance(profile, str):
        raise ProfileError(
            "profile must be 'openimcc', 'published', 'sr-candidate', "
            "'sr-candidate-v1', 'p2o5-candidate', 'cao-aluminate-candidate', "
            "'silicate-candidate', "
            "or a Profile"
        )
    return load_profile(profile)
