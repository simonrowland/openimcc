"""The published-integrity proof is reserved for packs that passed its gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from openimcc import evaluate, label_research_datapack, load_datapack


def test_research_label_does_not_claim_published_integrity_proof() -> None:
    loaded = load_datapack(Path("src/openimcc/data/packs/imcc-sf04-v1.0.2.json"))
    source = loaded.kernel_datapack
    assert source.identity_is_proven is True
    changed_A = np.array(source.A, copy=True)
    changed_A[0] += 0.5

    # dataclasses.replace does not preserve the init=False identity field.
    altered = replace(source, A=changed_A)
    assert altered.identity_is_proven is False

    research = label_research_datapack(
        altered,
        model_id="IMCC-SF04-RESEARCH",
        coverage="RE-research",
    )
    assert research.identity_is_proven is False
    assert research.binding_digest is not None
    assert research.binding_digest != loaded.binding_digest

    composition = {name: 0.125 for name in source.parent_oxides}
    published_result = evaluate(
        composition,
        2500.0,
        loaded,
        allow_out_of_envelope=True,
    )
    research_result = evaluate(
        composition,
        2500.0,
        research,
        allow_out_of_envelope=True,
    )
    assert not np.array_equal(
        published_result.parent_activity,
        research_result.parent_activity,
    )
    assert research_result.labels.identity["model_id"] == "IMCC-SF04-RESEARCH"
