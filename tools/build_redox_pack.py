#!/usr/bin/env python3
"""Build the immutable redox default from the accepted element fragments."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openimcc.redox_pack import canonical_redox_digest, load_redox_pack


PACK_DIR = ROOT / "src/openimcc/data/packs"
FRAGMENT_DIR = ROOT / "src/openimcc/data/fragments"
BASE_PACK = PACK_DIR / "openimcc-redox-cr-candidate-v4.json"
DEFAULT_PACK = PACK_DIR / "openimcc-redox-v5.json"
REDOX_FRAGMENTS = ("sr", "li", "zn", "yb")
DEFAULT_VERSION = "1.8.0-cr-candidate"


def build_default_redox_pack(
    destination: str | Path = DEFAULT_PACK,
    *,
    base_path: str | Path = BASE_PACK,
) -> str:
    """Append the selected associate fragments to the frozen Cr pack."""
    content = json.loads(Path(base_path).read_text(encoding="utf-8"))
    # Release integration is one new immutable pack assembled from the Cr
    # default and its accepted element rows; runtime continues loading a pack.
    for name in REDOX_FRAGMENTS:
        fragment_path = FRAGMENT_DIR / name / "rows.json"
        fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
        parents = fragment.get(
            "parent_registry_additions", fragment.get("parent_registry", [])
        )
        species = fragment.get(
            "thermo_species_additions", fragment.get("thermo_species", {})
        )
        rows = fragment.get("rows", [])
        for record in [*species.values(), *rows]:
            citations = record.pop("source_citations", [])
            if citations:
                provenance = record.get("provenance")
                if not isinstance(provenance, dict) or not provenance.get("source"):
                    raise ValueError(f"{name} citation record has no provenance source")
                provenance["source"] = (
                    f'{provenance["source"]}; {"; ".join(citations)}'
                )
        known_elements = {parent["element"] for parent in content["parent_registry"]}
        known_species = set(content["thermo_species"])
        added_elements = {parent["element"] for parent in parents}
        if known_elements.intersection(added_elements):
            raise ValueError(f"{name} duplicates an existing redox parent")
        if known_species.intersection(species):
            raise ValueError(f"{name} duplicates an existing thermochemical species")
        if any(parent["element"] != fragment["element"] for parent in parents):
            raise ValueError(f"{name} fragment contains another element's parent")
        if name == "sr":
            for parent in parents:
                parent["redox_thermochemistry_status"] = "populated"
        content["parent_registry"].extend(parents)
        content["thermo_species"].update(species)
        content["rows"].extend(rows)

    content["version"] = DEFAULT_VERSION
    content["model"] = (
        f'{content["model"]}; Sr, Li, Zn, and Yb association rows.'
    )
    content["errata"].append(
        {
            "version": DEFAULT_VERSION,
            "date": "2026-11-09",
            "change": "Adds accepted Sr, Li, Zn, and Yb association rows to the Cr redox pack.",
        }
    )
    content["canonical_content_digest"] = canonical_redox_digest(content)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(content, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return load_redox_pack(output).canonical_digest


if __name__ == "__main__":
    print(build_default_redox_pack())
