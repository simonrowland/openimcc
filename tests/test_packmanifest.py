"""Regression tests for datapack manifest validation."""

from __future__ import annotations

import json

from tools import packmanifest


def test_check_rejects_duplicate_bad_record_and_wrong_pack_count(
    tmp_path, monkeypatch, capsys
) -> None:
    pack_dir = tmp_path / "packs"
    pack_dir.mkdir()
    (pack_dir / "one.json").write_text('{"rows": []}\n', encoding="utf-8")
    manifest_path = pack_dir / "MANIFEST.json"
    monkeypatch.setattr(packmanifest, "PACK_DIR", pack_dir)
    monkeypatch.setattr(packmanifest, "MANIFEST", manifest_path)

    manifest = packmanifest.build()
    bad_record = {**manifest["packs"][0], "sha256": "0" * 64}
    manifest["packs"].insert(0, bad_record)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert packmanifest.main(["--check"]) == 1
    assert "duplicate filename(s) in manifest: one.json" in capsys.readouterr().err

    manifest["packs"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert packmanifest.main(["--check"]) == 1
    assert (
        "manifest has 0 pack record(s); found 1 pack(s) on disk"
        in capsys.readouterr().err
    )
