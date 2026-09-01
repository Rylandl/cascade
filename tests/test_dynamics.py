import jax
import jax.numpy as jnp

from cascade.dynamics import evaluate_dynamics
from cascade.initialization import standard_environment, zero_control, zero_state
from cascade.model import broadcast_model
from cascade.reference import aerobatic_reference
from cascade.state import ActuatorState, AeroState, Environment


def test_gravity_is_world_down_when_aerodynamic_forces_are_disabled():
    model = aerobatic_reference()
    state = zero_state(model)
    control = zero_control(model)
    environment = Environment(
        density=jnp.array(0.0), wind=jnp.zeros(3), gravity=jnp.array([0.0, 0.0, 9.80665])
    )

    result = evaluate_dynamics(model, state, control, environment)

    assert jnp.allclose(result.derivative.rigid_body.velocity, environment.gravity)


def test_propeller_generates_forward_force_and_reaction_torque():
    model = aerobatic_reference()
    state = zero_state(model)
    state = state._replace(
        actuators=ActuatorState(
            surface_deflection=state.actuators.surface_deflection,
            propeller_speed=jnp.array([800.0]),
        )
    )
    result = evaluate_dynamics(model, state, zero_control(model), standard_environment())

    assert result.propulsion.force_body[0] > 0.0
    assert result.propulsion.moment_body[0] < 0.0


def test_dynamics_are_finite_across_full_angle_and_zero_airspeed():
    model = aerobatic_reference()
    alpha = jnp.linspace(-jnp.pi, jnp.pi, 65)
    speed = jnp.concatenate((jnp.array([0.0]), jnp.full(64, 12.0)))
    velocity = jnp.stack(
        (speed * jnp.cos(alpha), jnp.zeros_like(alpha), speed * jnp.sin(alpha)), axis=-1
    )
    state = zero_state(model, batch_shape=(65,))
    state = state._replace(
        rigid_body=state.rigid_body._replace(velocity=velocity),
        aero=AeroState(separation=jnp.full((65, model.n_surfaces), 0.5)),
    )
    result = jax.jit(evaluate_dynamics)(
        model,
        state,
        zero_control(model, batch_shape=(65,)),
        standard_environment(batch_shape=(65,)),
    )

    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(result))


def test_force_gradient_is_finite_at_zero_airspeed():
    model = aerobatic_reference()
    base_state = zero_state(model)
    control = zero_control(model)
    environment = standard_environment()

    def force_from_velocity(velocity):
        state = base_state._replace(rigid_body=base_state.rigid_body._replace(velocity=velocity))
        return jnp.sum(evaluate_dynamics(model, state, control, environment).force_body)

    gradient = jax.grad(force_from_velocity)(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(gradient))


def test_native_batch_matches_single_evaluation():
    model = aerobatic_reference()
    speeds = jnp.array([8.0, 12.0, 16.0])
    batch_state = zero_state(model, batch_shape=(3,))
    batch_state = batch_state._replace(
        rigid_body=batch_state.rigid_body._replace(
            velocity=jnp.stack((speeds, jnp.zeros(3), jnp.ones(3)), axis=-1)
        )
    )
    batch_control = zero_control(model, batch_shape=(3,))
    batch_environment = standard_environment(batch_shape=(3,))
    batch_result = evaluate_dynamics(model, batch_state, batch_control, batch_environment)

    index = 1
    single_state = jax.tree.map(lambda value: value[index], batch_state)
    single_control = jax.tree.map(lambda value: value[index], batch_control)
    single_environment = jax.tree.map(lambda value: value[index], batch_environment)
    single_result = evaluate_dynamics(model, single_state, single_control, single_environment)

    assert jax.tree.all(
        jax.tree.map(
            lambda batched, single: jnp.allclose(batched[index], single),
            batch_result,
            single_result,
        )
    )


def test_model_parameters_can_vary_per_world():
    base_model = aerobatic_reference()
    model = broadcast_model(base_model, (3,))
    model = model._replace(mass=jnp.array([0.8, 1.2, 1.6]))
    state = zero_state(model, batch_shape=(3,))
    state = state._replace(
        actuators=ActuatorState(
            surface_deflection=state.actuators.surface_deflection,
            propeller_speed=jnp.full((3, 1), 600.0),
        )
    )
    environment = Environment(
        density=jnp.zeros(3), wind=jnp.zeros((3, 3)), gravity=jnp.zeros((3, 3))
    )
    result = jax.jit(evaluate_dynamics)(
        model, state, zero_control(model, batch_shape=(3,)), environment
    )

    acceleration = result.derivative.rigid_body.velocity[:, 0]
    assert acceleration[0] > acceleration[1] > acceleration[2]
