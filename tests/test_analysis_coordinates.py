import jax
import jax.numpy as jnp

from cascade.analysis.coordinates import (
    control_retract,
    state_difference,
    state_retract,
    tangent_state_size,
)
from cascade.initialization import zero_control, zero_state
from cascade.math import quaternion_from_rotvec, quaternion_to_rotvec
from cascade.reference import aerobatic_reference


def test_state_retraction_and_difference_are_local_inverses():
    model = aerobatic_reference()
    state = zero_state(model, altitude=3.0, forward_speed=12.0)
    delta = jnp.linspace(-0.01, 0.01, tangent_state_size(model))

    perturbed = state_retract(model, state, delta)
    recovered = state_difference(state, perturbed)

    assert recovered.shape == delta.shape
    assert jnp.allclose(recovered, delta, atol=2e-6)


def test_control_retraction_uses_propellers_then_channels():
    model = aerobatic_reference()
    control = zero_control(model)
    delta = jnp.array([0.4, 0.1, -0.2, 0.3])

    perturbed = control_retract(model, control, delta)

    assert jnp.array_equal(perturbed.propeller, delta[:1])
    assert jnp.array_equal(perturbed.channel, delta[1:])


def test_quaternion_maps_have_finite_identity_jacobians():
    exponential_jacobian = jax.jacfwd(quaternion_from_rotvec)(jnp.zeros(3))
    logarithm_jacobian = jax.jacfwd(quaternion_to_rotvec)(jnp.array([0.0, 0.0, 0.0, 1.0]))

    assert jnp.all(jnp.isfinite(exponential_jacobian))
    assert jnp.all(jnp.isfinite(logarithm_jacobian))
    assert jnp.allclose(exponential_jacobian[:3], 0.5 * jnp.eye(3))
    assert jnp.allclose(logarithm_jacobian[:, :3], 2.0 * jnp.eye(3))
