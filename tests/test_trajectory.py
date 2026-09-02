import jax.numpy as jnp

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
        controls=controls,
        stamp=stamp(spec, model, seed=1),
        note="unit test",
    )
    loaded, loaded_controls, metadata = load_trajectory(path)
    assert metadata["schema"] == TRAJECTORY_SCHEMA and metadata["note"] == "unit test"
    assert metadata["stamp"]["spec_name"] == spec.name and metadata["steps"] == 40
    assert jnp.allclose(loaded.rigid_body.position, trajectory.rigid_body.position, atol=1e-5)
    assert jnp.allclose(loaded.rigid_body.attitude, trajectory.rigid_body.attitude, atol=1e-6)
    assert jnp.allclose(loaded.rigid_body.velocity, trajectory.rigid_body.velocity, atol=1e-5)
    assert jnp.allclose(
        loaded.actuators.surface_deflection, trajectory.actuators.surface_deflection
    )
    assert jnp.allclose(loaded.aero.separation, trajectory.aero.separation)
    assert jnp.allclose(loaded_controls.channel, controls.channel)
