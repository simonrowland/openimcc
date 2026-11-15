"""Read the packaged coefficient-ancestry manifest."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def coefficient_ancestry(identity: str | None = None) -> Any:
    """Return packaged default ancestry, or one entry by identity or digest.

    The manifest is loaded from package resources, so this also works from a
    wheel without a source checkout. A selected entry may be addressed by its
    public identity, exact file digest, or runtime-loaded digest.
    """
    manifest = json.loads(
        resources.files("openimcc")
        .joinpath("data", "coefficient-ancestry.json")
        .read_text(encoding="utf-8")
    )
    artifacts = manifest["artifacts"]
    if identity is None:
        return artifacts
    for artifact in artifacts:
        if identity in {
            artifact["identity"],
            artifact["digest"],
            artifact["runtime_digest"],
        }:
            return artifact
    raise KeyError(f"no coefficient-ancestry artifact matches {identity!r}")
