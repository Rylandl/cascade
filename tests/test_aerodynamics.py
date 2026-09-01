import jax.numpy as jnp

from cascade.aerodynamics import (
    aerodynamic_coefficients,
    aerodynamics,
    propulsion,
    separation_derivative,
    surface_air_data,
)
from cascade.initialization import standard_environment, zero_state
from cascade.math import quaternion_rotate_inverse
from cascade.reference import aerobatic_reference
from cascade.state import ActuatorState, AeroState


def test_separated_flat_plate_coefficients_cover_full_envelope():
    model = aerobatic_reference()
    alpha = jnp.array([0.0, jnp.pi / 4.0, jnp.pi / 2.0, -jnp.pi / 4.0])[:, None]
    aero_state = AeroState(separation=jnp.ones((4, model.n_surfaces)))

    lift, drag, moment = aerodynamic_coefficients(model, aero_state, alpha)

    assert jnp.allclose(lift[0], 0.0, atol=1e-5)
    assert jnp.allclose(lift[2], 0.0, atol=1e-5)
    assert jnp.allclose(lift[1], -lift[3], atol=1e-5)
    assert jnp.all(drag >= 0.0)
    assert jnp.all(drag[2] > drag[0])
    assert jnp.allclose(moment, 0.0)


def test_separation_equilibrium_and_lag_have_expected_direction():
    model = aerobatic_reference()
    environment = standard_environment(batch_shape=(2,))
    state = zero_state(model, batch_shape=(2,), forward_speed=12.0)
    alpha = jnp.array([0.0, 40.0 * jnp.pi / 180.0])
    velocity = jnp.stack(
        (12.0 * jnp.cos(alpha), jnp.zeros_like(alpha), 12.0 * jnp.sin(alpha)), axis=-1
    )
    state = state._replace(rigid_body=state.rigid_body._replace(velocity=velocity))
    air_velocity_body = quaternion_rotate_inverse(
        state.rigid_body.attitude, state.rigid_body.velocity
    )
    propeller_result = propulsion(model, state.actuators.propeller_speed, environment.density)

    air, _ = surface_air_data(
        model, state, environment, air_velocity_body, propeller_result.induced_velocity
    )
    rate = separation_derivative(model, state.aero, air.separation_equilibrium)

    assert jnp.all(air.separation_equilibrium[0, :3] < 0.01)
    assert jnp.all(air.separation_equilibrium[1, :3] > 0.99)
    assert jnp.all(rate[1] > 0.0)


def test_propwash_reaches_mapped_tail_at_zero_freestream():
    model = aerobatic_reference()
    environment = standard_environment()
    state = zero_state(model)
    state = state._replace(
        actuators=ActuatorState(
            surface_deflection=state.actuators.surface_deflection,
            propeller_speed=jnp.array([700.0]),
        )
    )
    propeller_result = propulsion(model, state.actuators.propeller_speed, environment.density)
    result = aerodynamics(
        model,
        state,
        environment,
        jnp.zeros(3),
        propeller_result.induced_velocity,
    )

    assert result.air.dynamic_pressure[2] > result.air.dynamic_pressure[0]
    assert result.air.dynamic_pressure[3] > result.air.dynamic_pressure[0]


def test_spanwise_flow_produces_opposing_crossflow_drag():
    model = aerobatic_reference()
    environment = standard_environment()
    state = zero_state(model)
    propeller_result = propulsion(model, state.actuators.propeller_speed, environment.density)
    result = aerodynamics(
        model,
        state,
        environment,
        jnp.array([0.0, 10.0, 0.0]),
        propeller_result.induced_velocity,
    )

    assert result.force_body[1] < 0.0
