"""What a policy observes, and how noisily.

:class:`ObservationSpec` selects the blocks of :func:`cascade.env.observation`; the default
is everything (a privileged observation useful for learning research), and
:func:`onboard_observation` is what a small autopilot actually measures: rates and specific
force from an IMU, an attitude estimate (gravity direction and heading), a pitot airspeed,
and a GNSS position error. :class:`SensorNoise` is white noise per block plus per-episode
biases on the gyros and accelerometers; :func:`sensor_noise_from_sensors` builds it from
datasheet units (m/s, rad, rad/s, m/s^2, m) for a task, so the conversion into observation
units is the library's, not the user's.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from cascade.model import AircraftModel

GRAVITY_SCALE_M_S2 = 9.80665  # the specific-force block is in g units
POSITION_SCALE_M = 10.0


class ObservationSpec(NamedTuple):
    """Which observation blocks a policy sees, in this order when present: air velocity in
    body axes (3), airspeed (1), alpha and beta (2), body rates (3), gravity direction in body
    axes (3), heading error sin and cos (2), position error in body axes (3), surface
    deflections (S), propeller speed fractions (P), specific force in body axes in g (3)."""

    air_velocity: bool = True
    airspeed: bool = True
    air_angles: bool = True
    rates: bool = True
    gravity: bool = True
    heading: bool = True
    position_error: bool = True
    surfaces: bool = True
    propellers: bool = True
    specific_force: bool = False


def full_observation() -> ObservationSpec:
    return ObservationSpec()


def onboard_observation() -> ObservationSpec:
    """Rates and specific force (IMU), gravity direction and heading (attitude estimate), pitot
    airspeed, and position error (GNSS): no flow angles, no actuator states."""

    return ObservationSpec(
        air_velocity=False,
        airspeed=True,
        air_angles=False,
        rates=True,
        gravity=True,
        heading=True,
        position_error=True,
        surfaces=False,
        propellers=False,
        specific_force=True,
    )


def block_sizes(model: AircraftModel) -> dict[str, int]:
    return {
        "air_velocity": 3,
        "airspeed": 1,
        "air_angles": 2,
        "rates": 3,
        "gravity": 3,
        "heading": 2,
        "position_error": 3,
        "surfaces": model.n_surfaces,
        "propellers": model.n_propellers,
        "specific_force": 3,
    }


class SensorNoise(NamedTuple):
    """Per-block white-noise standard deviations in observation units, plus per-episode bias
    standard deviations for the gyros and accelerometers (drawn once in :func:`reset`).
    Zeros (the default) give the true state. See :func:`sensor_noise_from_sensors` for
    datasheet units."""

    air_std: Array
    angle_std: Array
    rate_std: Array
    rate_bias_std: Array
    gravity_std: Array
    heading_std: Array
    position_std: Array
    actuator_std: Array
    specific_force_std: Array
    specific_force_bias_std: Array


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
    specific_force_std: float = 0.0,
    specific_force_bias_std: float = 0.0,
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
        specific_force_std,
        specific_force_bias_std,
    )
    return SensorNoise(*(jnp.asarray(value) for value in values))


def sensor_noise_from_sensors(
    reference_speed_m_s: float,
    *,
    airspeed_std_m_s: float = 0.0,
    angle_std_rad: float = 0.0,
    gyro_std_rad_s: float = 0.0,
    gyro_bias_std_rad_s: float = 0.0,
    accelerometer_std_m_s2: float = 0.0,
    accelerometer_bias_std_m_s2: float = 0.0,
    attitude_std_rad: float = 0.0,
    heading_std_rad: float = 0.0,
    position_std_m: float = 0.0,
    actuator_std_rad: float = 0.0,
) -> SensorNoise:
    """Noise from datasheet units for a task whose observations are scaled by
    ``reference_speed_m_s`` (the task's own reference speed). Small angles: an attitude error
    of ``a`` rad moves a unit-vector component by about ``a``."""

    speed = max(float(reference_speed_m_s), 1.0)
    return sensor_noise(
        air_std=airspeed_std_m_s / speed,
        angle_std=angle_std_rad,
        rate_std=gyro_std_rad_s,
        rate_bias_std=gyro_bias_std_rad_s,
        gravity_std=attitude_std_rad,
        heading_std=heading_std_rad,
        position_std=position_std_m / POSITION_SCALE_M,
        actuator_std=actuator_std_rad,
        specific_force_std=accelerometer_std_m_s2 / GRAVITY_SCALE_M_S2,
        specific_force_bias_std=accelerometer_bias_std_m_s2 / GRAVITY_SCALE_M_S2,
    )


def _noise_vectors(
    model: AircraftModel, noise: SensorNoise, spec: ObservationSpec | None = None
) -> tuple[Array, Array]:
    """White-noise and bias standard deviations laid out like the observation of ``spec``."""

    spec = ObservationSpec() if spec is None else spec
    sizes = block_sizes(model)
    white_by_block = {
        "air_velocity": noise.air_std,
        "airspeed": noise.air_std,
        "air_angles": noise.angle_std,
        "rates": noise.rate_std,
        "gravity": noise.gravity_std,
        "heading": noise.heading_std,
        "position_error": noise.position_std,
        "surfaces": noise.actuator_std,
        "propellers": noise.actuator_std,
        "specific_force": noise.specific_force_std,
    }
    bias_by_block = {"rates": noise.rate_bias_std, "specific_force": noise.specific_force_bias_std}
    white, bias = [], []
    for name, size in sizes.items():
        if not getattr(spec, name):
            continue
        white.append(jnp.broadcast_to(white_by_block[name], (size,)))
        bias.append(jnp.broadcast_to(bias_by_block.get(name, jnp.zeros(())), (size,)))
    return jnp.concatenate(white), jnp.concatenate(bias)


__all__ = [
    "GRAVITY_SCALE_M_S2",
    "ObservationSpec",
    "SensorNoise",
    "block_sizes",
    "full_observation",
    "onboard_observation",
    "sensor_noise",
    "sensor_noise_from_sensors",
]
