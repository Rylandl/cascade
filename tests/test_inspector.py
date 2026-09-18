import json
import re

import jax.numpy as jnp
import numpy as np
import pytest

from cascade import aerobatic_reference, save_trajectory, zero_control, zero_state
from cascade.viz import trajectory_report


def payload(html):
    return json.loads(
        re.search(
            r'<script id="flight-data" type="application/json">(.*?)</script>', html, re.S
        ).group(1)
    )


def test_offline_report_embeds_comparison_timing_events_and_escapes_content(tmp_path):
    model = aerobatic_reference()
    state = zero_state(model, batch_shape=(5,), altitude=50.0)
    state = state._replace(
        rigid_body=state.rigid_body._replace(
            position=state.rigid_body.position.at[:, 0].set(jnp.arange(5))
        )
    )
    first = save_trajectory(
        tmp_path / "a",
        state,
        0.1,
        t0_s=0.1,
        events=[dict(time_s=0.2, label="</script><script>alert(1)</script>")],
    )
    second = save_trajectory(tmp_path / "b", state, 0.1, t0_s=1.0)
    report = trajectory_report(
        first, tmp_path / "report.html", compare=second, labels=("A </script>", "B"), max_points=3
    )
    text = report.read_text()
    flights = payload(text)
    assert flights[0]["time"] == pytest.approx([0.1, 0.3, 0.5])
    assert flights[1]["time"] == pytest.approx([1.0, 1.2, 1.4])
    assert flights[0]["original_samples"] == 5
    assert "<script>alert(1)</script>" not in text
    assert 'src="http' not in text and "fetch(" not in text
    assert flights[0]["events"][0]["label"].startswith("</script>")


def test_sensor_invalid_age_is_a_gap_and_input_files_are_protected(tmp_path):
    states = zero_state(aerobatic_reference(), batch_shape=(2,))
    path = save_trajectory(tmp_path / "a", states, 0.1)
    np.savez(
        path.with_suffix(".diagnostics.npz"),
        sensor_age_s=np.array([[np.inf], [0.1]]),
        sensor_valid=np.array([[False], [True]]),
    )
    report = trajectory_report(path, tmp_path / "a.html")
    assert payload(report.read_text())[0]["series"]["sensor_age_s 0"] == [None, 0.1]
    with pytest.raises(ValueError, match="overwrite"):
        trajectory_report(path, path)
    np.savez(
        path.with_suffix(".diagnostics.npz"),
        sensor_age_s=np.array([[np.inf], [0.1]]),
        sensor_valid=np.array([[True], [True]]),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        trajectory_report(path, tmp_path / "bad.html")


def test_report_does_not_overwrite_diagnostic_input_or_hardlinked_trajectory(tmp_path):
    states = zero_state(aerobatic_reference(), batch_shape=(2,))
    path = save_trajectory(tmp_path / "flight", states, 0.1)
    diagnostics = path.with_suffix(".diagnostics.npz")
    np.savez(diagnostics, reward=np.zeros(2))
    original = diagnostics.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        trajectory_report(path, diagnostics)
    assert diagnostics.read_bytes() == original
    alias = tmp_path / "alias.html"
    alias.hardlink_to(path)
    with pytest.raises(ValueError, match="overwrite"):
        trajectory_report(path, alias)


def test_boolean_event_time_is_rejected_before_browser_rendering(tmp_path):
    states = zero_state(aerobatic_reference(), batch_shape=(2,))
    path = save_trajectory(
        tmp_path / "flight", states, 0.1, events=[{"time_s": True, "label": "invalid time"}]
    )
    with pytest.raises(ValueError, match="events"):
        trajectory_report(path, tmp_path / "report.html")


@pytest.mark.parametrize("validity", [np.array([[1.0], [0.0]]), np.array([[2], [2]])])
def test_sensor_validity_must_be_boolean(tmp_path, validity):
    states = zero_state(aerobatic_reference(), batch_shape=(2,))
    path = save_trajectory(tmp_path / "flight", states, 0.1)
    np.savez(
        path.with_suffix(".diagnostics.npz"),
        sensor_age_s=np.array([[np.inf], [0.1]]),
        sensor_valid=validity,
    )
    with pytest.raises(ValueError, match="sensor_valid"):
        trajectory_report(path, tmp_path / "report.html")


@pytest.mark.parametrize("command_width,applied_width", [(2, 3), (3, 3)])
def test_action_overlay_requires_matching_channels_and_stored_controls(
    tmp_path, command_width, applied_width
):
    model = aerobatic_reference()
    path = save_trajectory(
        tmp_path / "flight",
        zero_state(model, batch_shape=(2,)),
        0.1,
        controls=zero_control(model, batch_shape=(2,)),
    )
    np.savez(
        path.with_suffix(".diagnostics.npz"),
        commanded_action=np.zeros((2, command_width)),
        applied_action=np.zeros((2, applied_width)),
    )
    with pytest.raises(ValueError, match="action"):
        trajectory_report(path, tmp_path / "report.html")


@pytest.mark.parametrize(
    "age,validity",
    [
        (np.array([[-0.1], [0.1]]), np.ones((2, 1), bool)),
        (np.zeros((2, 2)), np.ones((2, 1), bool)),
    ],
)
def test_sensor_ages_require_nonnegative_values_and_aligned_validity(tmp_path, age, validity):
    states = zero_state(aerobatic_reference(), batch_shape=(2,))
    path = save_trajectory(tmp_path / "flight", states, 0.1)
    np.savez(path.with_suffix(".diagnostics.npz"), sensor_age_s=age, sensor_valid=validity)
    with pytest.raises(ValueError, match="sensor_age_s"):
        trajectory_report(path, tmp_path / "report.html")


@pytest.mark.parametrize("shape", [(0,), (2, 1)])
def test_empty_and_batched_trajectories_are_rejected(tmp_path, shape):
    states = zero_state(aerobatic_reference(), batch_shape=shape)
    path = save_trajectory(tmp_path / "flight", states, 0.1)
    with pytest.raises(ValueError, match="nonempty, unbatched"):
        trajectory_report(path, tmp_path / "report.html")


def test_large_finite_velocity_has_finite_speed_and_rates_have_units(tmp_path):
    states = zero_state(aerobatic_reference(), batch_shape=(2,))
    states = states._replace(
        rigid_body=states.rigid_body._replace(velocity=jnp.full((2, 3), 1e30, dtype=jnp.float32))
    )
    path = save_trajectory(tmp_path / "flight", states, 0.1)
    report = trajectory_report(path, tmp_path / "report.html")
    series = payload(report.read_text())[0]["series"]
    np.testing.assert_allclose(series["Ground speed · m/s"], np.sqrt(3) * 1e30, rtol=1e-6)
    assert "Body rate · rad/s roll" in series


@pytest.mark.parametrize(
    "diagnostics",
    [
        {"reward": np.zeros((2, 1))},
        {"reward": np.zeros(1)},
        {"commanded_action": np.zeros((2, 4))},
        {"commanded_action": np.zeros((2, 1, 4)), "applied_action": np.zeros((2, 1, 4))},
    ],
)
def test_malformed_diagnostic_dimensions_and_unpaired_actions_are_rejected(tmp_path, diagnostics):
    path = save_trajectory(
        tmp_path / "flight", zero_state(aerobatic_reference(), batch_shape=(2,)), 0.1
    )
    np.savez(path.with_suffix(".diagnostics.npz"), **diagnostics)
    with pytest.raises(ValueError):
        trajectory_report(path, tmp_path / "report.html")
