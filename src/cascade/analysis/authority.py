"""Control authority: angular acceleration per unit channel command at a flight condition."""

from __future__ import annotations

import jax
import numpy as np

from cascade.dynamics import evaluate_dynamics


def control_authority(model, state, control, environment) -> np.ndarray:
    """Angular acceleration per unit channel command, ``(3, C)``, with the surfaces placed at
    the command's steady deflection (so actuator lag does not hide the authority)."""

    def acceleration(channel):
        deflection = model.actuators.surface_map @ channel + model.actuators.surface_bias
        placed = state._replace(actuators=state.actuators._replace(surface_deflection=deflection))
        result = evaluate_dynamics(model, placed, control._replace(channel=channel), environment)
        return result.derivative.rigid_body.angular_velocity

    return np.asarray(jax.jacfwd(acceleration)(control.channel))


__all__ = ["control_authority"]
