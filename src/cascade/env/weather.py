"""Weather for episodes: a mean wind with a boundary-layer profile, turbulence by class or
by station record, and a gust generator that runs inside the step.

A :class:`WeatherCondition` is what a station reports: wind speed and the direction it blows
from at 10 m, plus the 20 ft wind that drives MIL-F-8785C turbulence intensity (by default the
same speed). :func:`mean_wind_ned` applies a logarithmic profile above the surface roughness
length, so an aircraft near the ground sees less wind than one at cruise height.
:func:`step_gust` advances Dryden filters one control period with the length scales and
intensities of the aircraft's current altitude, so turbulence follows the aircraft down. A
:class:`WeatherRecords` table (hourly station observations: speed, direction, optional gust)
makes "actual weather" a draw from a season of real conditions.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cascade.env.gusts import dryden_low_altitude, first_order_gust, second_order_gust
from cascade.math import normalize

REFERENCE_HEIGHT_M = 10.0
GUST_STATE_SIZE = 5


class WeatherCondition(NamedTuple):
    """Mean wind at 10 m (speed, direction it blows from, clockwise from north), the 20 ft
    wind driving turbulence intensity, the surface roughness length of the site, a mean
    updraft, and one discrete one-minus-cosine gust (amplitude, start time, duration,
    direction in NED; zero amplitude means none)."""

    wind_speed_m_s: Array
    wind_from_rad: Array
    turbulence_wind_20ft_m_s: Array
    roughness_length_m: Array
    vertical_wind_m_s: Array
    gust_amplitude_m_s: Array
    gust_start_s: Array
    gust_duration_s: Array
    gust_direction_ned: Array


def weather_condition(
    wind_speed_m_s: float = 0.0,
    wind_from_rad: float = 0.0,
    *,
    turbulence_wind_20ft_m_s: float | None = None,
    roughness_length_m: float = 0.03,
    vertical_wind_m_s: float = 0.0,
    gust_amplitude_m_s: float = 0.0,
    gust_start_s: float = 0.0,
    gust_duration_s: float = 1.0,
    gust_direction_ned: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> WeatherCondition:
    """``vertical_wind_m_s`` is a mean updraft (positive up), uniform with altitude: thermals,
    ridge lift, or the vertical component a flight campaign inferred. The discrete gust is the
    certification-standard one-minus-cosine shape: ``V(t) = A/2 (1 - cos(2 pi (t - t0) / T))``
    for ``t0 <= t <= t0 + T``, along ``gust_direction_ned`` (an updraft by default)."""

    turbulence = wind_speed_m_s if turbulence_wind_20ft_m_s is None else turbulence_wind_20ft_m_s
    return WeatherCondition(
        wind_speed_m_s=jnp.asarray(wind_speed_m_s),
        wind_from_rad=jnp.asarray(wind_from_rad),
        turbulence_wind_20ft_m_s=jnp.asarray(turbulence),
        roughness_length_m=jnp.asarray(roughness_length_m),
        vertical_wind_m_s=jnp.asarray(vertical_wind_m_s),
        gust_amplitude_m_s=jnp.asarray(gust_amplitude_m_s),
        gust_start_s=jnp.asarray(gust_start_s),
        gust_duration_s=jnp.asarray(gust_duration_s),
        gust_direction_ned=normalize(jnp.asarray(gust_direction_ned, dtype=float)),
    )


def weather_classes() -> dict[str, WeatherCondition]:
    """MIL-F-8785C low-altitude turbulence classes as conditions with a matching mean wind
    from the north: calm, light (7.7 m/s at 20 ft), moderate (15.4), severe (23.2)."""

    return {
        "calm": weather_condition(0.0, 0.0),
        "light": weather_condition(7.7, 0.0),
        "moderate": weather_condition(15.4, 0.0),
        "severe": weather_condition(23.2, 0.0),
    }


def mean_wind_ned(condition: WeatherCondition, altitude_m: Array) -> Array:
    """Mean wind vector in NED at an altitude: log profile over roughness, blowing toward
    the direction opposite ``wind_from_rad``. Below the roughness length the wind is zero."""

    z0 = condition.roughness_length_m
    height = jnp.maximum(jnp.asarray(altitude_m), z0)
    profile = jnp.log(height / z0) / jnp.log(REFERENCE_HEIGHT_M / z0)
    speed = condition.wind_speed_m_s * jnp.maximum(profile, 0.0)
    toward = condition.wind_from_rad + jnp.pi
    down = -condition.vertical_wind_m_s * jnp.ones_like(speed)  # NED: an updraft is negative down
    return jnp.stack((speed * jnp.cos(toward), speed * jnp.sin(toward), down), axis=-1)


def discrete_gust_ned(condition: WeatherCondition, time_s: Array) -> Array:
    """The one-minus-cosine discrete gust at ``time_s``, in NED; zero outside its window."""

    phase = (jnp.asarray(time_s) - condition.gust_start_s) / jnp.maximum(
        condition.gust_duration_s, 1e-6
    )
    inside = (phase >= 0.0) & (phase <= 1.0)
    magnitude = 0.5 * condition.gust_amplitude_m_s * (1.0 - jnp.cos(2.0 * jnp.pi * phase))
    return jnp.where(inside, magnitude, 0.0)[..., None] * condition.gust_direction_ned


def isa_density(altitude_m: Array, sea_level_density_kg_m3: float = 1.225) -> Array:
    """International Standard Atmosphere density in the troposphere (to 11 km)."""

    ratio = jnp.maximum(1.0 - 2.25577e-5 * jnp.asarray(altitude_m), 1e-3)
    return sea_level_density_kg_m3 * ratio**4.2559


def initial_gust_state(batch_shape: tuple[int, ...] = ()) -> Array:
    return jnp.zeros((*batch_shape, GUST_STATE_SIZE))


def step_gust(
    condition: WeatherCondition,
    gust_state: Array,
    key: Array,
    dt: float,
    *,
    airspeed_m_s: Array,
    altitude_m: Array,
    heading_rad: Array,
) -> tuple[Array, Array]:
    """Advance the Dryden filters one period at the aircraft's altitude and airspeed.

    Returns the next filter state and the gust vector in NED. The longitudinal gust acts along
    ``heading_rad``, the lateral to its right, the vertical positive down. A zero turbulence
    wind gives zero gusts and leaves the state at rest.
    """

    parameters = dryden_low_altitude(altitude_m, condition.turbulence_wind_20ft_m_s)
    airspeed = jnp.maximum(jnp.asarray(airspeed_m_s), 1.0)
    tau_u = parameters.length_u / airspeed
    tau_v = parameters.length_v / airspeed
    tau_w = parameters.length_w / airspeed
    noise = jax.random.normal(key, (*gust_state.shape[:-1], 3))
    u_state = first_order_gust(gust_state[..., 0], noise[..., 0], tau_u, parameters.sigma_u, dt)
    v_state, v_gust = second_order_gust(
        gust_state[..., 1:3], noise[..., 1], tau_v, parameters.sigma_v, dt
    )
    w_state, w_gust = second_order_gust(
        gust_state[..., 3:5], noise[..., 2], tau_w, parameters.sigma_w, dt
    )
    next_state = jnp.concatenate((u_state[..., None], v_state, w_state), axis=-1)
    cosine, sine = jnp.cos(heading_rad), jnp.sin(heading_rad)
    gust_ned = jnp.stack(
        (cosine * u_state - sine * v_gust, sine * u_state + cosine * v_gust, w_gust), axis=-1
    )
    return next_state, gust_ned


class WeatherRecords(NamedTuple):
    """Hourly station observations: wind speed (m/s), direction the wind blows from (rad,
    clockwise from north), and gust speed (m/s, or the wind speed where not reported)."""

    wind_speed_m_s: Array
    wind_from_rad: Array
    gust_m_s: Array

    @classmethod
    def from_arrays(cls, wind_speed_m_s, wind_from_deg, gust_m_s=None) -> WeatherRecords:
        speed = jnp.asarray(np.asarray(wind_speed_m_s, dtype=float))
        direction = jnp.deg2rad(jnp.asarray(np.asarray(wind_from_deg, dtype=float)))
        gust = speed if gust_m_s is None else jnp.asarray(np.asarray(gust_m_s, dtype=float))
        return cls(wind_speed_m_s=speed, wind_from_rad=direction, gust_m_s=jnp.maximum(gust, speed))

    @classmethod
    def from_csv(cls, path: str | Path) -> WeatherRecords:
        """Read ``wind_speed_m_s``, ``wind_from_deg``, and optional ``gust_m_s`` columns (any
        other columns are ignored, empty gusts fall back to the wind speed)."""

        speeds, directions, gusts = [], [], []
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                speed = float(row["wind_speed_m_s"])
                speeds.append(speed)
                directions.append(float(row["wind_from_deg"]))
                gust = row.get("gust_m_s", "")
                gusts.append(float(gust) if gust not in ("", None) else speed)
        if not speeds:
            raise ValueError(f"no records in {path}")
        return cls.from_arrays(speeds, directions, gusts)

    def __len__(self) -> int:
        return int(self.wind_speed_m_s.shape[0])


def sample_weather(
    records: WeatherRecords, key: Array, *, roughness_length_m: float = 0.03
) -> WeatherCondition:
    """Draw one record as a condition. The turbulence wind is the reported gust: for a station
    gust, the peak over the observation, which stands in for the 20 ft wind a turbulence
    class would quote."""

    index = jax.random.randint(key, (), 0, len(records))
    return WeatherCondition(
        wind_speed_m_s=records.wind_speed_m_s[index],
        wind_from_rad=records.wind_from_rad[index],
        turbulence_wind_20ft_m_s=records.gust_m_s[index],
        roughness_length_m=jnp.asarray(roughness_length_m),
        vertical_wind_m_s=jnp.zeros(()),
        gust_amplitude_m_s=jnp.zeros(()),
        gust_start_s=jnp.zeros(()),
        gust_duration_s=jnp.ones(()),
        gust_direction_ned=jnp.array([0.0, 0.0, -1.0]),
    )


def sample_weather_uniform(
    key: Array,
    *,
    speed_range_m_s: tuple[float, float] = (0.0, 12.0),
    turbulence_ratio_range: tuple[float, float] = (1.0, 1.5),
    roughness_length_m: float = 0.03,
) -> WeatherCondition:
    """A synthetic condition: uniform speed, uniform direction, turbulence wind a uniform
    multiple of the mean. For when no station records are at hand."""

    k_speed, k_direction, k_ratio = jax.random.split(key, 3)
    speed = jax.random.uniform(k_speed, (), minval=speed_range_m_s[0], maxval=speed_range_m_s[1])
    direction = jax.random.uniform(k_direction, (), minval=0.0, maxval=2.0 * jnp.pi)
    ratio = jax.random.uniform(
        k_ratio, (), minval=turbulence_ratio_range[0], maxval=turbulence_ratio_range[1]
    )
    return WeatherCondition(
        wind_speed_m_s=speed,
        wind_from_rad=direction,
        turbulence_wind_20ft_m_s=speed * ratio,
        roughness_length_m=jnp.asarray(roughness_length_m),
        vertical_wind_m_s=jnp.zeros(()),
        gust_amplitude_m_s=jnp.zeros(()),
        gust_start_s=jnp.zeros(()),
        gust_duration_s=jnp.ones(()),
        gust_direction_ned=jnp.array([0.0, 0.0, -1.0]),
    )


__all__ = [
    "GUST_STATE_SIZE",
    "discrete_gust_ned",
    "isa_density",
    "WeatherCondition",
    "WeatherRecords",
    "initial_gust_state",
    "mean_wind_ned",
    "sample_weather",
    "sample_weather_uniform",
    "step_gust",
    "weather_classes",
    "weather_condition",
]
