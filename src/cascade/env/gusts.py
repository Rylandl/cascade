"""Dryden continuous turbulence as a time-major :class:`Environment` sequence.

The Dryden model (MIL-F-8785C) filters white noise through first-order (longitudinal) and
second-order (lateral and vertical) shaping filters whose time constants are the turbulence
length scales divided by the aircraft's nominal airspeed. Gusts are generated in a frame aligned
with a chosen heading, rotated into world NED, and added to a mean wind, so the result feeds
straight into ``rollout(..., environments=...)``. Everything is a pure function of a PRNG key
and broadcasts over a batch of worlds, so gust realizations can differ per world.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from cascade.state import Environment

FEET_PER_METRE = 3.280839895


class DrydenParameters(NamedTuple):
    """Turbulence intensities (m/s) and length scales (m) for the three gust components."""

    sigma_u: Array
    sigma_v: Array
    sigma_w: Array
    length_u: Array
    length_v: Array
    length_w: Array


def dryden_low_altitude(altitude_m: Array, wind_20ft_m_s: Array) -> DrydenParameters:
    """MIL-F-8785C low-altitude (below about 300 m) intensities and length scales.

    ``wind_20ft_m_s`` is the mean wind at 20 ft: about 7.7 m/s for light, 15.4 m/s for moderate
    and 23.2 m/s for severe turbulence. Altitude is clamped to 10 m so the scales stay finite.
    """

    height_ft = jnp.maximum(jnp.asarray(altitude_m), 10.0) * FEET_PER_METRE
    sigma_w = 0.1 * jnp.asarray(wind_20ft_m_s)
    ratio = (0.177 + 0.000823 * height_ft) ** 0.4
    sigma_u = sigma_w / ratio
    length_w_ft = height_ft
    length_u_ft = height_ft / (0.177 + 0.000823 * height_ft) ** 1.2
    return DrydenParameters(
        sigma_u=sigma_u,
        sigma_v=sigma_u,
        sigma_w=sigma_w,
        length_u=length_u_ft / FEET_PER_METRE,
        length_v=length_u_ft / FEET_PER_METRE,
        length_w=length_w_ft / FEET_PER_METRE,
    )


def _first_order(state: Array, noise: Array, tau: Array, sigma: Array, dt: float) -> Array:
    """Exact discretization of ``tau x' + x = sigma sqrt(2 tau) w`` with unit white noise."""

    decay = jnp.exp(-dt / tau)
    return decay * state + sigma * jnp.sqrt(1.0 - jnp.square(decay)) * noise


def _second_order(
    state: Array, noise: Array, tau: Array, sigma: Array, dt: float
) -> tuple[Array, Array]:
    """Dryden lateral/vertical filter ``sigma sqrt(tau/pi) (1 + sqrt(3) tau s) / (1 + tau s)^2``.

    Implemented as a unit-variance Ornstein-Uhlenbeck stage followed by a second stage with
    the same pole. The weighted sum ``sqrt(3) x1 + (1 - sqrt(3)) x2`` realizes the numerator
    ``1 + sqrt(3) tau s``; integrating its spectrum gives twice the target variance, hence the
    ``1 / sqrt(2)``. ``state`` holds the two stage outputs.
    """

    first, second = state[..., 0], state[..., 1]
    decay = jnp.exp(-dt / tau)
    first_next = decay * first + jnp.sqrt(1.0 - jnp.square(decay)) * noise
    second_next = decay * second + (1.0 - decay) * first_next
    gust = (
        sigma / jnp.sqrt(2.0) * (jnp.sqrt(3.0) * first_next + (1.0 - jnp.sqrt(3.0)) * second_next)
    )
    return jnp.stack((first_next, second_next), axis=-1), gust


def dryden_wind_sequence(
    key: Array,
    steps: int,
    dt: float,
    *,
    airspeed_m_s: Array,
    parameters: DrydenParameters,
    heading_rad: Array = 0.0,
    mean_wind_ned: Array | None = None,
    batch_shape: tuple[int, ...] = (),
) -> Array:
    """Generate a time-major ``(steps, *batch_shape, 3)`` world-NED wind with Dryden gusts.

    The longitudinal gust acts along ``heading_rad`` (clockwise from north), the lateral gust to
    its right, and the vertical gust positive down in NED. ``airspeed_m_s`` sets the filter time
    constants ``L / V``. Intensities of zero reproduce the mean wind exactly.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    airspeed = jnp.maximum(jnp.asarray(airspeed_m_s, dtype=jnp.float32), 1e-3)
    heading = jnp.asarray(heading_rad, dtype=jnp.float32)
    tau_u = parameters.length_u / airspeed
    tau_v = parameters.length_v / airspeed
    tau_w = parameters.length_w / airspeed
    noise = jax.random.normal(key, (steps, *batch_shape, 3), dtype=jnp.float32)
    initial = jnp.zeros((*batch_shape, 5), dtype=jnp.float32)

    def step(state, sample):
        u_state = _first_order(state[..., 0], sample[..., 0], tau_u, parameters.sigma_u, dt)
        v_state, v_gust = _second_order(
            state[..., 1:3], sample[..., 1], tau_v, parameters.sigma_v, dt
        )
        w_state, w_gust = _second_order(
            state[..., 3:5], sample[..., 2], tau_w, parameters.sigma_w, dt
        )
        next_state = jnp.concatenate((u_state[..., None], v_state, w_state), axis=-1)
        gust_frame = jnp.stack((u_state, v_gust, w_gust), axis=-1)
        cosine, sine = jnp.cos(heading), jnp.sin(heading)
        gust_ned = jnp.stack(
            (
                cosine * gust_frame[..., 0] - sine * gust_frame[..., 1],
                sine * gust_frame[..., 0] + cosine * gust_frame[..., 1],
                gust_frame[..., 2],
            ),
            axis=-1,
        )
        return next_state, gust_ned

    _, gusts = jax.lax.scan(step, initial, noise)
    mean = jnp.zeros(3) if mean_wind_ned is None else jnp.asarray(mean_wind_ned)
    return gusts + mean


def dryden_environment_sequence(
    key: Array,
    environment: Environment,
    steps: int,
    dt: float,
    *,
    airspeed_m_s: Array,
    parameters: DrydenParameters,
    heading_rad: Array = 0.0,
) -> Environment:
    """Repeat ``environment`` over ``steps`` with Dryden gusts added to its wind.

    The result is time-major and matches the leading batch shape of ``environment.wind``, so it
    can be passed as ``rollout(..., environments=...)``.
    """

    batch_shape = tuple(environment.wind.shape[:-1])
    wind = dryden_wind_sequence(
        key,
        steps,
        dt,
        airspeed_m_s=airspeed_m_s,
        parameters=parameters,
        heading_rad=heading_rad,
        mean_wind_ned=environment.wind,
        batch_shape=batch_shape,
    )
    return Environment(
        density=jnp.broadcast_to(environment.density, (steps, *environment.density.shape)),
        wind=wind,
        gravity=jnp.broadcast_to(environment.gravity, (steps, *environment.gravity.shape)),
    )
