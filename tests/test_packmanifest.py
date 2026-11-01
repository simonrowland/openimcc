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


def _lifecycle_manifest() -> dict:
    return {
        "packs": [
            {
                "file": "melt-default.json",
                "datapack_version": "melt-v2",
                "role": "default",
            },
            {
                "file": "openimcc-redox-default.json",
                "datapack_version": "redox-v4",
                "role": "default",
            },
            {
                "file": "redox-cr-v1.json",
                "role": "historical",
                "family": "redox-cr",
                "superseded_by": "redox-cr-v2.json",
            },
            {
                "file": "redox-cr-v2.json",
                "role": "opt-in",
                "family": "redox-cr",
            },
        ]
    }


def test_lifecycle_roles_validate_defaults_history_and_candidate_family() -> None:
    issues = packmanifest._validate_roles(
        _lifecycle_manifest(), {"melt": "melt-v2", "redox": "redox-v4"}
    )

    assert issues == []


def test_lifecycle_roles_reject_invalid_metadata() -> None:
    manifest = _lifecycle_manifest()
    manifest["packs"][0]["role"] = "primary"
    manifest["packs"][2]["superseded_by"] = "absent.json"
    manifest["packs"].append(
        {"file": "redox-cr-v3.json", "role": "opt-in", "family": "redox-cr"}
    )

    issues = packmanifest._validate_roles(
        manifest, {"melt": "melt-v2", "redox": "redox-v4"}
    )

    assert any("invalid or missing role" in issue for issue in issues)
    assert any("not in the manifest" in issue for issue in issues)
    assert any("exactly one current opt-in" in issue for issue in issues)


def test_lifecycle_roles_reject_self_supersession() -> None:
    manifest = _lifecycle_manifest()
    manifest["packs"][2]["superseded_by"] = "redox-cr-v1.json"

    issues = packmanifest._validate_roles(
        manifest, {"melt": "melt-v2", "redox": "redox-v4"}
    )

    assert any("cannot supersede itself" in issue for issue in issues)


def test_lifecycle_roles_reject_cross_family_supersession() -> None:
    manifest = _lifecycle_manifest()
    manifest["packs"][2]["superseded_by"] = "melt-default.json"

    issues = packmanifest._validate_roles(
        manifest, {"melt": "melt-v2", "redox": "redox-v4"}
    )

    assert any("must stay within candidate family" in issue for issue in issues)


def test_lifecycle_roles_reject_supersession_cycle() -> None:
    manifest = _lifecycle_manifest()
    manifest["packs"][2]["superseded_by"] = "redox-cr-v0.json"
    manifest["packs"].append(
        {
            "file": "redox-cr-v0.json",
            "role": "historical",
            "family": "redox-cr",
            "superseded_by": "redox-cr-v1.json",
        }
    )

    issues = packmanifest._validate_roles(
        manifest, {"melt": "melt-v2", "redox": "redox-v4"}
    )

    assert any("superseded_by cycle" in issue for issue in issues)


def test_lifecycle_defaults_must_match_runtime_versions() -> None:
    manifest = _lifecycle_manifest()

    issues = packmanifest._validate_roles(
        manifest, {"melt": "melt-v1", "redox": "redox-v4"}
    )

    assert any("does not match the runtime default version" in issue for issue in issues)


def test_build_preserves_existing_lifecycle_metadata(tmp_path, monkeypatch) -> None:
    pack_dir = tmp_path / "packs"
    pack_dir.mkdir()
    (pack_dir / "one.json").write_text('{"rows": []}\n', encoding="utf-8")
    manifest_path = pack_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "packs": [
                    {
                        "file": "one.json",
                        "role": "opt-in",
                        "family": "example",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(packmanifest, "PACK_DIR", pack_dir)
    monkeypatch.setattr(packmanifest, "MANIFEST", manifest_path)

    entry = packmanifest.build()["packs"][0]

    assert entry["role"] == "opt-in"
    assert entry["family"] == "example"
