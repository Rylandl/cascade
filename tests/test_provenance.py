import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import cascade
from cascade import provenance
from cascade.provenance import STAMP_SCHEMA, model_hash, spec_hash, stamp, write_stamp


def test_hashes_are_deterministic_and_sensitive():
    spec = cascade.aerobatic_reference_spec()
    assert spec_hash(spec) == spec_hash(cascade.aerobatic_reference_spec())
    heavier = replace(spec, mass_kg=spec.mass_kg * 1.01)
    assert spec_hash(heavier) != spec_hash(spec)
    model = spec.to_model()
    assert model_hash(model) == model_hash(spec.to_model())
    assert model_hash(heavier.to_model()) != model_hash(model)


def test_stamp_carries_versions_numerics_and_seed(tmp_path):
    spec = cascade.aerobatic_reference_spec()
    record = write_stamp(tmp_path / "stamp.json", spec, spec.to_model(), seed=7, run="unit")
    loaded = json.loads((tmp_path / "stamp.json").read_text())
    assert loaded == record
    for key in ("schema", "cascade_version", "jax_version", "backend", "x64_enabled", "seed"):
        assert key in record
    assert record["schema"] == STAMP_SCHEMA
    assert record["seed"] == 7 and record["run"] == "unit"
    assert record["spec_hash"] == spec_hash(spec) and len(record["model_hash"]) == 64
    assert stamp()["schema"] == STAMP_SCHEMA


@pytest.mark.parametrize(
    "field", ["schema", "cascade_version", "git_commit", "spec_hash", "model_hash"]
)
def test_stamp_rejects_reserved_custom_fields(field):
    with pytest.raises(ValueError, match="reserved provenance fields"):
        stamp(**{field: "forged"})


def test_stamp_rejects_non_json_context():
    with pytest.raises(ValueError, match="JSON compliant"):
        stamp(metric=float("nan"))
    with pytest.raises(TypeError, match="not JSON serializable"):
        stamp(callback=lambda: None)


def test_installed_package_does_not_inherit_enclosing_repository_commit(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    installed = tmp_path / ".venv" / "lib" / "python3.13" / "site-packages" / "cascade"
    installed.mkdir(parents=True)
    monkeypatch.setattr(provenance, "__file__", str(installed / "provenance.py"))

    def unexpected_git(*args, **kwargs):
        pytest.fail("an installed distribution must not query an enclosing repository")

    monkeypatch.setattr(provenance.subprocess, "run", unexpected_git)
    assert provenance.git_commit() is None


def test_source_archive_does_not_inherit_enclosing_repository_commit(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    source = tmp_path / "download" / "src" / "cascade"
    source.mkdir(parents=True)
    monkeypatch.setattr(provenance, "__file__", str(source / "provenance.py"))
    monkeypatch.setattr(
        provenance.subprocess, "run", lambda *args, **kwargs: pytest.fail("not a checkout")
    )
    assert provenance.git_commit() is None


def test_git_commit_requires_matching_checkout_root(tmp_path, monkeypatch):
    source = tmp_path / "src" / "cascade"
    source.mkdir(parents=True)
    (tmp_path / ".git").write_text("gitdir: /worktree/metadata\n")
    monkeypatch.setattr(provenance, "__file__", str(source / "provenance.py"))
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=f"{tmp_path}\nabc123\n"),
    )
    assert provenance.git_commit() == "abc123"
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="/another/repo\nabc123\n"),
    )
    assert provenance.git_commit() is None
