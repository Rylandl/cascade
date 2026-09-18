import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cascade
from cascade.integration import repeat_control, rollout
from cascade.provenance import stamp
from cascade.trajectory import TRAJECTORY_SCHEMA, load_trajectory, save_trajectory


def test_trajectory_round_trips_with_controls_and_stamp(tmp_path):
    spec = cascade.aerobatic_reference_spec()
    model = spec.to_model()
    environment = cascade.standard_environment()
    state = cascade.zero_state(model, altitude=50.0, forward_speed=12.0)
    control = cascade.ControlInput(propeller=jnp.array([0.6]), channel=jnp.array([0.1, -0.05, 0.0]))
    controls = repeat_control(control, 40)
    _, trajectory = rollout(model, state, controls, environment, 0.005)
    path = save_trajectory(
        tmp_path / "flight.npz",
        trajectory,
        0.005,
        t0_s=0.005,
        controls=controls,
        stamp=stamp(spec, model, seed=1),
        note="unit test",
    )
    loaded, loaded_controls, metadata = load_trajectory(path)
    assert metadata["schema"] == TRAJECTORY_SCHEMA and metadata["note"] == "unit test"
    assert metadata["stamp"]["spec_name"] == spec.name and metadata["steps"] == 40
    with np.load(path) as data:
        np.testing.assert_allclose(data["time_s"], np.arange(1, 41) * 0.005)
    assert metadata["t0_s"] == 0.005
    assert jnp.allclose(loaded.rigid_body.position, trajectory.rigid_body.position, atol=1e-5)
    assert jnp.allclose(loaded.rigid_body.attitude, trajectory.rigid_body.attitude, atol=1e-6)
    assert jnp.allclose(loaded.rigid_body.velocity, trajectory.rigid_body.velocity, atol=1e-5)
    assert jnp.allclose(
        loaded.actuators.surface_deflection, trajectory.actuators.surface_deflection
    )
    assert jnp.allclose(loaded.aero.separation, trajectory.aero.separation)
    assert jnp.allclose(loaded_controls.channel, controls.channel)


@pytest.fixture
def sample_trajectory():
    model = cascade.aerobatic_reference_spec().to_model()
    # Two independent worlds with three samples each.
    trajectory = cascade.zero_state(model, batch_shape=(3, 2))
    controls = cascade.zero_control(model, batch_shape=(3, 2))
    return trajectory, controls


def test_batched_trajectory_suffix_and_original_v1_compatibility(tmp_path, sample_trajectory):
    trajectory, controls = sample_trajectory
    path = save_trajectory(tmp_path / "flight", trajectory, 0.1, controls=controls)
    assert path == tmp_path / "flight.npz" and path.is_file()
    with np.load(path) as data:
        arrays = dict(data)
    header = json.loads(str(arrays["metadata_json"]))
    del header["t0_s"]  # v1 files created before explicit time origins were supported.
    arrays["metadata_json"] = np.array(json.dumps(header))
    np.savez_compressed(path, **arrays)
    loaded, loaded_controls, metadata = load_trajectory(path)
    np.testing.assert_allclose(loaded.rigid_body.position, trajectory.rigid_body.position)
    np.testing.assert_allclose(loaded_controls.propeller, controls.propeller)
    assert loaded.aero.separation.shape == trajectory.aero.separation.shape
    assert metadata["t0_s"] == 0.0


@pytest.mark.parametrize("dt", [0, -0.1, float("nan"), float("inf"), True, [0.1]])
def test_save_rejects_invalid_timestep(tmp_path, sample_trajectory, dt):
    with pytest.raises(ValueError, match="dt"):
        save_trajectory(tmp_path / "invalid", sample_trajectory[0], dt)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("key", ["schema", "state_schema", "steps", "dt_s"])
def test_save_protects_reserved_metadata(tmp_path, sample_trajectory, key):
    with pytest.raises(ValueError, match="reserved trajectory metadata"):
        save_trajectory(tmp_path / "invalid", sample_trajectory[0], 0.1, **{key: "override"})


def test_save_checks_input_shapes_and_finite_data(tmp_path, sample_trajectory):
    trajectory, controls = sample_trajectory
    invalid = trajectory._replace(
        rigid_body=trajectory.rigid_body._replace(velocity=jnp.zeros((3, 3)))
    )
    with pytest.raises(ValueError, match="velocity must have shape"):
        save_trajectory(tmp_path / "invalid", invalid, 0.1)
    invalid = trajectory._replace(
        rigid_body=trajectory.rigid_body._replace(
            position=trajectory.rigid_body.position.at[0].set(jnp.nan)
        )
    )
    with pytest.raises(ValueError, match="position must contain only finite"):
        save_trajectory(tmp_path / "invalid", invalid, 0.1)
    with pytest.raises(ValueError, match="control_propeller must match"):
        save_trajectory(
            tmp_path / "invalid",
            trajectory,
            0.1,
            controls=controls._replace(propeller=controls.propeller[:-1]),
        )
    with pytest.raises(ValueError, match="t0_s"):
        save_trajectory(tmp_path / "invalid", trajectory, 0.1, t0_s=float("inf"))
    with pytest.raises(ValueError, match="JSON compliant"):
        save_trajectory(tmp_path / "invalid", trajectory, 0.1, note={"value": float("nan")})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema", "cascade_trajectory_v2", "unsupported trajectory schema"),
        ("state_schema", "ned_frd_xyzw", "unsupported state schema"),
        ("dt_s", -0.1, "dt_s must be positive"),
        ("steps", 4, "canonical_state must have shape"),
        ("steps", 3.0, "steps must be a nonnegative integer"),
        ("t0_s", 1.0, "time_s must equal"),
    ],
)
def test_load_rejects_inconsistent_metadata(tmp_path, sample_trajectory, field, value, error):
    path = save_trajectory(tmp_path / "flight", sample_trajectory[0], 0.1)
    with np.load(path) as data:
        arrays = dict(data)
    header = json.loads(str(arrays["metadata_json"]))
    header[field] = value
    arrays["metadata_json"] = np.array(json.dumps(header))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match=error):
        load_trajectory(path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("canonical_state", np.zeros((3, 2, 12)), "canonical_state must have shape"),
        ("surface_deflection_rad", np.zeros((2, 2, 5)), "surface_deflection_rad must have shape"),
        ("separation", np.zeros((3, 2, 0)), "separation and surface_deflection_rad"),
        ("propeller_speed_rad_s", np.full((3, 2, 1), np.nan), "finite real numbers"),
        ("time_s", np.array([0.0, 0.1, 0.3]), "time_s must equal"),
        ("time_s", np.array([0.0, 0.1]), "time_s must have shape"),
        ("control_propeller", None, "must both be present or absent"),
        ("control_channel", None, "must both be present or absent"),
        ("control_channel", np.zeros((2, 3)), "control_channel must have shape"),
        ("metadata_json", np.array("[]"), "must contain a JSON object"),
    ],
)
def test_load_rejects_inconsistent_arrays(tmp_path, sample_trajectory, field, value, error):
    trajectory, controls = sample_trajectory
    path = save_trajectory(tmp_path / "flight", trajectory, 0.1, controls=controls)
    with np.load(path) as data:
        arrays = dict(data)
    if value is None:
        del arrays[field]
    else:
        arrays[field] = value
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match=error):
        load_trajectory(path)


def test_empty_trajectory_round_trips(tmp_path):
    model = cascade.aerobatic_reference_spec().to_model()
    trajectory = cascade.zero_state(model, batch_shape=(0,))
    path = save_trajectory(tmp_path / "empty", trajectory, 0.1)
    loaded, controls, metadata = load_trajectory(path)
    assert loaded.rigid_body.position.shape == (0, 3)
    assert controls is None and metadata["steps"] == 0


@pytest.mark.parametrize("scale", [0.0, 0.5, 2.0])
def test_invalid_quaternions_are_rejected_on_save_and_load(tmp_path, sample_trajectory, scale):
    trajectory, _ = sample_trajectory
    invalid = trajectory._replace(
        rigid_body=trajectory.rigid_body._replace(attitude=trajectory.rigid_body.attitude * scale)
    )
    with pytest.raises(ValueError, match="quaternion norms"):
        save_trajectory(tmp_path / "invalid", invalid, 0.1)
    path = save_trajectory(tmp_path / "valid", trajectory, 0.1)
    with np.load(path) as data:
        arrays = dict(data)
    arrays["canonical_state"][..., 6:10] *= scale
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="quaternion norms"):
        load_trajectory(path)


def test_near_unit_quaternion_roundoff_is_normalized(tmp_path, sample_trajectory):
    trajectory, _ = sample_trajectory
    trajectory = trajectory._replace(
        rigid_body=trajectory.rigid_body._replace(attitude=trajectory.rigid_body.attitude * 1.00005)
    )
    path = save_trajectory(tmp_path / "rounded", trajectory, 0.1)
    loaded, _, _ = load_trajectory(path)
    np.testing.assert_allclose(np.linalg.norm(loaded.rigid_body.attitude, axis=-1), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "field",
    [
        "canonical_state",
        "surface_deflection_rad",
        "propeller_speed_rad_s",
        "separation",
        "control_propeller",
        "control_channel",
    ],
)
def test_load_rejects_active_dtype_overflow(tmp_path, sample_trajectory, field):
    trajectory, controls = sample_trajectory
    path = save_trajectory(tmp_path / "flight", trajectory, 0.1, controls=controls)
    with np.load(path) as data:
        arrays = dict(data)
    arrays[field] = arrays[field].astype(np.float64)
    arrays[field][..., 0] = 1e100
    np.savez_compressed(path, **arrays)
    previous = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        with (
            np.errstate(over="ignore"),
            pytest.raises(ValueError, match="after JAX dtype conversion"),
        ):
            load_trajectory(path)
        jax.config.update("jax_enable_x64", True)
        loaded, loaded_controls, _ = load_trajectory(path)
        for leaf in jax.tree.leaves((loaded, loaded_controls)):
            assert np.all(np.isfinite(np.asarray(leaf)))
    finally:
        jax.config.update("jax_enable_x64", previous)
