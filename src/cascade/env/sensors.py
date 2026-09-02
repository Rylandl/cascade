"""Observation noise: white noise per observation block and a per-episode bias."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from cascade.model import AircraftModel


class SensorNoise(NamedTuple):
    """Per-block observation noise: white noise standard deviations and a per-episode bias
    standard deviation, drawn once in :func:`reset`. Zeros (the default) give the true state.

    Blocks follow :func:`observation`: air velocity and airspeed as a fraction of the reference
    speed, alpha and beta in radians, body rates in rad/s, gravity direction (unit vector
    components), heading error sin and cos, position error as a fraction of 10 m, surface
    deflections in radians, propeller speed fraction.
    """

    air_std: Array
    angle_std: Array
    rate_std: Array
    rate_bias_std: Array
    gravity_std: Array
    heading_std: Array
    position_std: Array
    actuator_std: Array


def sensor_noise(
    *,
    air_std: float = 0.0,
    angle_std: float = 0.0,
    rate_std: float = 0.0,
    rate_bias_std: float = 0.0,
    gravity_std: float = 0.0,
    heading_std: float = 0.0,
    position_std: float = 0.0,
    actuator_std: float = 0.0,
) -> SensorNoise:
    values = (
        air_std,
        angle_std,
        rate_std,
        rate_bias_std,
        gravity_std,
        heading_std,
        position_std,
        actuator_std,
    )
    return SensorNoise(*(jnp.asarray(value) for value in values))


def _noise_vectors(model: AircraftModel, noise: SensorNoise) -> tuple[Array, Array]:
    """White-noise and bias standard deviations laid out like the observation."""

    def block(value, size):
        return jnp.broadcast_to(value, (size,))

    actuators = model.n_surfaces + model.n_propellers
    white = jnp.concatenate(
        (
            block(noise.air_std, 4),
            block(noise.angle_std, 2),
            block(noise.rate_std, 3),
            block(noise.gravity_std, 3),
            block(noise.heading_std, 2),
            block(noise.position_std, 3),
            block(noise.actuator_std, actuators),
        )
    )
    bias = jnp.concatenate((jnp.zeros(6), block(noise.rate_bias_std, 3), jnp.zeros(8 + actuators)))
    return white, bias
