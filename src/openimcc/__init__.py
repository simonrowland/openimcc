"""openimcc -- ideal mixing of complex components for silicate melts.

Public surface. Three groups: the solve, the refusal hierarchy, and the gas
layer.

Raw ``solve_*`` kernel entry points are deliberately NOT exported. Every caller
goes through ``evaluate``, which is where domain and envelope checking lives; a
caller reaching past it gets numbers with no validity labelling attached.

``backend.py`` is not part of this package. The MeltBackend subclasses are glue
for the simulator that spawned this code and stay upstream -- they import this
package, never the other way round.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openimcc.model import (
    ImccAdapterLabels,
    ImccComponentOutsideDomainError,
    ImccCompositionIncompleteError,
    ImccCompositionOutsideValidatedEnvelopeError,
    ImccDatapack,
    ImccFerricInputUnsupportedError,
    ImccLoadedDatapack,
    ImccMalformedDatapackError,
    ImccNonconvergenceError,
    ImccRefusal,
    ImccResult,
    ImccSPComponentRequiresExtensionError,
    ImccTOutsideDatapackDomainError,
    ImccUnprovenDatapackError,
    evaluate,
    label_research_datapack,
    load_datapack,
)
from openimcc.kernel import ImccDataframeUnavailableError, ImccSpeciesNotFoundError


def _resolve_version() -> str:
    """Installed distribution version, or an honest admission.

    A fabricated version attached to a scientific result is worse than an
    unknown one, because it looks reproducible and is not.
    """
    try:
        from importlib.metadata import version

        return version("openimcc")
    except Exception:
        return "0+unknown"


__version__ = _resolve_version()


# --------------------------------------------------------------------------
# Gas layer: exported, but imported lazily.
#
# gas.py is the only module that needs pandas, and `pip install openimcc`
# (core) deliberately does not pull it. If this module imported gas eagerly,
# a core install would fail on `import openimcc` -- so the whole dependency
# tiering, and the CI job that guards it, would be fiction.
#
# PEP 562 module __getattr__ gives `from openimcc import evaluate_gas` without
# paying that cost: the import happens on first attribute access, and a core
# install raises ImportError there, naming the extra, rather than at package
# import.
# --------------------------------------------------------------------------

_GAS_EXPORTS = frozenset({
    "BAR",
    "IMCC_GAS_CHANNEL_SPECIES",
    "IMCC_GAS_UNAVAILABLE_SPECIES",
    "IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES",
    "IMCC_PARENT_OXIDES",
    "ImccGasDatapack",
    "ImccGasInvalidFugacityError",
    "ImccGasResult",
    "ImccGasSpeciesNotFoundError",
    "ImccGasTemperatureOutsideDomainError",
    "R_J_MOL_K",
    "evaluate_gas",
    "gas_species_provenance",
    "load_gas_datapack",
})

if TYPE_CHECKING:  # so type checkers and IDEs still see the gas names
    from openimcc.gas import (
        BAR,
        IMCC_GAS_CHANNEL_SPECIES,
        IMCC_GAS_UNAVAILABLE_SPECIES,
        IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES,
        IMCC_PARENT_OXIDES,
        ImccGasDatapack,
        ImccGasInvalidFugacityError,
        ImccGasResult,
        ImccGasSpeciesNotFoundError,
        ImccGasTemperatureOutsideDomainError,
        R_J_MOL_K,
        evaluate_gas,
        gas_species_provenance,
        load_gas_datapack,
    )


def __getattr__(name: str):
    if name in _GAS_EXPORTS:
        try:
            from openimcc import gas
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                f"openimcc.{name} needs the gas layer, which requires pandas. "
                'Install it with: pip install "openimcc[gas]"'
            ) from exc
        return getattr(gas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "__version__",
    # --- solve ------------------------------------------------------------
    "load_datapack",
    "evaluate",
    "label_research_datapack",
    "ImccDatapack",
    "ImccLoadedDatapack",
    "ImccResult",
    "ImccAdapterLabels",
    # --- refusals ---------------------------------------------------------
    # ImccRefusal is the base: `except ImccRefusal` catches every typed
    # decline, which is what makes the exit-2 contract usable from Python as
    # well as from the shell. Exporting only the leaves would force callers to
    # enumerate them and to re-enumerate on every release.
    "ImccRefusal",
    "ImccTOutsideDatapackDomainError",
    "ImccCompositionOutsideValidatedEnvelopeError",
    "ImccCompositionIncompleteError",
    "ImccComponentOutsideDomainError",
    "ImccSpeciesNotFoundError",
    "ImccDataframeUnavailableError",
    "ImccSPComponentRequiresExtensionError",
    "ImccFerricInputUnsupportedError",
    "ImccMalformedDatapackError",
    "ImccUnprovenDatapackError",
    "ImccNonconvergenceError",
    # --- gas (lazy; needs the [gas] extra) --------------------------------
    "load_gas_datapack",
    "evaluate_gas",
    "gas_species_provenance",
    "ImccGasDatapack",
    "ImccGasInvalidFugacityError",
    "ImccGasResult",
    "ImccGasSpeciesNotFoundError",
    "ImccGasTemperatureOutsideDomainError",
    # BAR is exported on purpose. evaluate_gas returns pressures in BAR while
    # the simulator's own vapour layer returns Pa -- exactly 5 dex apart. That
    # difference is invisible in a log-residual table, so the unit belongs at
    # the public surface where a caller trips over it.
    "BAR",
]
