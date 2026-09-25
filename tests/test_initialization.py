"""Physical equilibrium at reset, including coupled panel downwash."""

from contextlib import contextmanager
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cascade
from cascade.aerodynamics import propulsion, surface_air_data
from cascade.math import quaternion_rotate_inverse


@contextmanager
def precision():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def washed_model():
    spec = cascade.aerobatic_reference_spec()
    table = np.zeros((len(spec.surfaces), len(spec.surfaces)))
    table[2, :2] = 0.12
    return replace(spec, downwash_map=tuple(map(tuple, table))).to_model()


@pytest.mark.parametrize("angle", [0.0, 0.12, 0.25, -0.25])
def test_downwash_reset_is_a_separation_equilibrium(angle):
    model = washed_model()
    control = cascade.ControlInput(jnp.array([0.4]), jnp.array([0.03, -0.1, 0.02]))
    env = cascade.standard_environment()
    state = cascade.zero_state(model, altitude=50)
    state = state._replace(
        rigid_body=state.rigid_body._replace(
            velocity=jnp.array([12 * np.cos(angle), 0, 12 * np.sin(angle)])
        )
    )
    state = jax.jit(cascade.equilibrate_internal_state)(model, state, control, env)
    dynamics = cascade.evaluate_dynamics(model, state, control, env)
    np.testing.assert_allclose(dynamics.derivative.aero.separation, 0, atol=2e-6)
    # This checks the effective airflow used by the ODE, not a duplicate reset formula.
    np.testing.assert_allclose(
        state.aero.separation, dynamics.aerodynamics.air.separation_equilibrium, atol=2e-7
    )


def test_zero_downwash_reset_preserves_the_geometric_equilibrium_exactly():
    model = cascade.aerobatic_reference()
    control = cascade.ControlInput(jnp.array([0.4]), jnp.array([0.1, -0.2, 0.0]))
    env = cascade.standard_environment()
    state = cascade.equilibrate_internal_state(
        model, cascade.zero_state(model, forward_speed=12), control, env
    )
    velocity = quaternion_rotate_inverse(state.rigid_body.attitude, state.rigid_body.velocity)
    prop = propulsion(model, state, env, velocity)
    air, _ = surface_air_data(model, state, env, velocity, prop.induced_velocity)
    np.testing.assert_array_equal(state.aero.separation, air.separation_equilibrium)


def test_downwash_equilibrium_supports_batching_and_gradient():
    with precision():
        model, env = washed_model(), cascade.standard_environment()
        state = cascade.zero_state(model, forward_speed=12)

        def reset(elevator):
            control = cascade.ControlInput(jnp.array([0.4]), jnp.array([0.0, elevator, 0.0]))
            return cascade.equilibrate_internal_state(model, state, control, env).aero.separation

        elevator = -0.2
        gradient = jax.jit(jax.jacfwd(reset))(elevator)
        h = 1e-5
        numerical = (reset(elevator + h) - reset(elevator - h)) / (2 * h)
        np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=2e-6)
        batch = jax.jit(jax.vmap(reset))(jnp.array([elevator, 0.1]))
        np.testing.assert_allclose(batch[0], reset(elevator), atol=1e-12)


@pytest.mark.slow
def test_downwash_trim_holds_after_initialization():
    with precision():
        model = washed_model()
        env = cascade.standard_environment()
        trim = cascade.trim_straight_flight(
            model, cascade.StraightFlightCondition(12.0, altitude_m=1000)
        )
        assert trim.success
        initial = cascade.evaluate_dynamics(model, trim.state, trim.control, env)
        np.testing.assert_allclose(initial.derivative.aero.separation, 0, atol=1e-10)
        final, _ = jax.jit(cascade.rollout)(
            model, trim.state, cascade.repeat_control(trim.control, 200), env, 0.005
        )
        np.testing.assert_allclose(
            final.rigid_body.velocity, trim.state.rigid_body.velocity, atol=1e-7
        )
        np.testing.assert_allclose(final.rigid_body.angular_velocity, 0, atol=1e-7)
