import json
from dataclasses import replace

import cascade
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
    assert stamp()["spec_hash" if False else "schema"] == STAMP_SCHEMA
