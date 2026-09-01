import jax.numpy as jnp

from cascade.actuators import actuator_targets, control_from_actuators
from cascade.reference import aerobatic_reference
from cascade.state import ControlInput


def test_control_from_actuators_inverts_actuator_targets():
    model = aerobatic_reference()
    control = ControlInput(propeller=jnp.array([0.4]), channel=jnp.array([0.3, -0.2, 0.1]))

    recovered = control_from_actuators(model, actuator_targets(model, control))

    assert jnp.allclose(recovered.propeller, control.propeller, atol=1e-6)
    assert jnp.allclose(recovered.channel, control.channel, atol=1e-5)


def test_channels_are_linear_until_the_physical_surface_limit():
    model = aerobatic_reference()

    def deflection(aileron):
        control = ControlInput(propeller=jnp.zeros(1), channel=jnp.array([aileron, 0.0, 0.0]))
        return actuator_targets(model, control).surface_deflection[0]

    # 1.15 * 25 deg is still inside the 30 deg limit, so the map stays linear past unit range.
    assert jnp.allclose(deflection(1.15), 1.15 * deflection(1.0), atol=1e-6)
    assert jnp.allclose(deflection(3.0), model.actuators.surface_limit[0])
