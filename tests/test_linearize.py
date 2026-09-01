import jax.numpy as jnp

from cascade.analysis import (
    control_retract,
    linearize_step,
    state_difference,
    state_retract,
    tangent_state_size,
)
from cascade.initialization import equilibrate_internal_state, standard_environment, zero_state
from cascade.integration import rk4_step
from cascade.reference import aerobatic_reference
from cascade.state import ControlInput


def test_step_linearization_is_finite_and_predicts_small_perturbations():
    model = aerobatic_reference()
    environment = standard_environment()
    control = ControlInput(propeller=jnp.array([0.3]), channel=jnp.zeros(3))
    state = equilibrate_internal_state(
        model,
        zero_state(model, altitude=10.0, forward_speed=12.0),
        control,
        environment,
    )
    linearization = linearize_step(model, state, control, environment, 0.01)

    state_size = tangent_state_size(model)
    input_size = model.n_propellers + model.n_control_channels
    assert linearization.state_matrix.shape == (state_size, state_size)
    assert linearization.input_matrix.shape == (state_size, input_size)
    assert jnp.all(jnp.isfinite(linearization.state_matrix))
    assert jnp.all(jnp.isfinite(linearization.input_matrix))

    state_delta = 1e-3 * jnp.linspace(-1.0, 1.0, state_size)
    input_delta = 1e-3 * jnp.array([0.2, -0.5, 0.3, 0.4])
    perturbed_state = state_retract(model, state, state_delta)
    perturbed_control = control_retract(model, control, input_delta)
    actual_next = rk4_step(model, perturbed_state, perturbed_control, environment, 0.01)
    actual_delta = state_difference(linearization.nominal_next_state, actual_next)
    predicted_delta = linearization.predict(state_delta, input_delta)

    relative_error = jnp.linalg.norm(actual_delta - predicted_delta) / jnp.linalg.norm(actual_delta)
    assert relative_error < 2e-3
