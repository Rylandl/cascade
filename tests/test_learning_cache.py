import copy
import json
from types import SimpleNamespace

import pytest

from cascade.experiments import Policy
from cascade.experiments.manifest import content_hash
from cascade.learning import workflow
from cascade.learning.workflow import _cached_evaluation


def unused_factory(scenario, model, reference):
    raise AssertionError("cache lookup must not compile or execute a policy")


@pytest.fixture
def cache(tmp_path):
    directory = tmp_path / "evaluation-abc"
    experiment = SimpleNamespace(
        sha256="experiment-sha",
        scenarios=(
            SimpleNamespace(name="nominal", split="evaluation", seeds=(30000, 30001)),
            SimpleNamespace(name="validation", split="validation", seeds=(20000,)),
        ),
    )
    policies = [Policy("learned-7", unused_factory, {"checkpoint_sha256": "checkpoint-sha"})]
    result = {
        "schema": "cascade_results_v1",
        "experiment_sha256": experiment.sha256,
        "policies": [policy.provenance() for policy in policies],
        "episodes": [
            {
                "scenario": "nominal",
                "split": "evaluation",
                "policy": "learned-7",
                "seed": seed,
                "trajectory": f"flights/nominal/learned-7/{seed}.npz",
                "return": 3.0,
            }
            for seed in (30000, 30001)
        ],
    }
    return directory, experiment, policies, result


def save_results(directory, result, *, reports=False):
    directory.mkdir(parents=True, exist_ok=True)
    body = {name: value for name, value in result.items() if name != "results_sha256"}
    stored = {**body, "results_sha256": content_hash(body)}
    (directory / "results.json").write_text(json.dumps(stored))
    if reports:
        for row in stored["episodes"]:
            if row["trajectory"] is not None:
                html = (directory / row["trajectory"]).with_suffix(".html")
                html.parent.mkdir(parents=True, exist_ok=True)
                html.write_text("<html>test report</html>")
    return stored


def lookup(cache, *, reports=False):
    directory, experiment, policies, _ = cache
    return _cached_evaluation(directory, experiment, policies, "evaluation", reports=reports)


def test_complete_cache_reuses_matching_results(cache):
    directory, experiment, policies, result = cache
    stored = save_results(directory, result)
    assert lookup(cache) == (directory, stored)
    # A fresh factory wrapper on resume has the same explicit identity and source hash.
    restored = [Policy(policy.name, unused_factory, dict(policy.metadata)) for policy in policies]
    assert stored["policies"][0]["factory_source_sha256"] is not None
    assert [policy.provenance() for policy in restored] == stored["policies"]
    assert _cached_evaluation(directory, experiment, restored, "evaluation", reports=False) == (
        directory,
        stored,
    )


def test_checkpoint_factory_identity_is_stable_across_independent_loads(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.npz"
    path.write_bytes(b"immutable checkpoint bytes")
    monkeypatch.setattr(
        workflow,
        "load_checkpoint",
        lambda path: SimpleNamespace(
            provenance={"training_seed": 7}, state=SimpleNamespace(iteration=2)
        ),
    )
    first = workflow._checkpoint_policy(path, "learned-7")
    second = workflow._checkpoint_policy(path, "learned-7")
    assert first.factory is not second.factory
    assert first.provenance()["factory_source_sha256"] is not None
    assert first.provenance() == second.provenance()


def test_modified_result_is_not_reused(cache):
    directory, _, _, result = cache
    stored = save_results(directory, result)
    stored["episodes"][0]["return"] = 999.0
    (directory / "results.json").write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="content hash mismatch"):
        lookup(cache)


@pytest.mark.parametrize(
    "mismatch", ["experiment", "checkpoint", "factory_source", "split", "missing", "duplicate"]
)
def test_self_consistent_but_wrong_cache_is_rejected(cache, mismatch):
    directory, _, _, original = cache
    result = copy.deepcopy(original)
    if mismatch == "experiment":
        result["experiment_sha256"] = "other-experiment"
    elif mismatch == "checkpoint":
        result["policies"][0]["metadata"]["checkpoint_sha256"] = "other-checkpoint"
    elif mismatch == "factory_source":
        result["policies"][0]["factory_source_sha256"] = "changed-factory-source"
    elif mismatch == "split":
        result["episodes"][0]["split"] = "validation"
    elif mismatch == "missing":
        result["episodes"].pop()
    else:
        result["episodes"][1] = result["episodes"][0]
    save_results(directory, result)
    with pytest.raises(ValueError, match="mismatch"):
        lookup(cache)


@pytest.mark.parametrize("truncated_json", [False, True])
def test_completed_retry_is_reused_after_interruption(cache, truncated_json):
    directory, _, _, result = cache
    directory.mkdir()
    (directory / "manifest.json").write_text("{}")
    if truncated_json:
        (directory / "results.json").write_text('{"schema":')
    retry = directory.with_name(f"{directory.name}-retry-1")
    stored = save_results(retry, result)
    assert lookup(cache) == (retry, stored)


def test_partial_attempts_are_preserved_when_selecting_fresh_output(cache):
    directory = cache[0]
    directory.mkdir()
    (directory / "manifest.json").write_text("original")
    retry = directory.with_name(f"{directory.name}-retry-1")
    retry.mkdir()
    (retry / "results.json").write_text('{"schema":')
    assert lookup(cache) == (directory.with_name(f"{directory.name}-retry-2"), None)
    assert (directory / "manifest.json").read_text() == "original"
    assert (retry / "results.json").read_text() == '{"schema":'


def test_reports_request_scores_new_attempt_then_reuses_its_reports(cache):
    directory, _, _, result = cache
    base = save_results(directory, result)
    retry = directory.with_name(f"{directory.name}-retry-1")
    assert lookup(cache, reports=True) == (retry, None)
    stored = save_results(retry, result, reports=True)
    assert lookup(cache, reports=True) == (retry, stored)
    assert lookup(cache) == (directory, base)
    (retry / result["episodes"][0]["trajectory"]).with_suffix(".html").unlink()
    assert lookup(cache, reports=True) == (
        directory.with_name(f"{directory.name}-retry-2"),
        None,
    )


def test_failed_flights_do_not_require_nonexistent_html(cache):
    directory, _, _, result = cache
    for row in result["episodes"]:
        row["trajectory"] = None
    stored = save_results(directory, result)
    assert lookup(cache, reports=True) == (directory, stored)


def test_empty_cache_uses_base_directory(cache):
    assert lookup(cache) == (cache[0], None)
    cache[0].mkdir()
    assert lookup(cache) == (cache[0], None)
