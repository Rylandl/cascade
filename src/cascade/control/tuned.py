"""Tuned controllers for the packaged aircraft: fixture data next to the aircraft it fits,
found by closed-loop step response. Any other spec gets
:func:`cascade.control.autotune.tune_cascade`."""

from __future__ import annotations

import jax.numpy as jnp

from cascade.control.loops import (
    AttitudeGains,
    CascadeController,
    GuidanceGains,
    RateGains,
    channel_map,
)
from cascade.control.vtol import TransitionController, default_hover_gains
from cascade.spec import AircraftSpec


def aerobatic_reference_controller() -> CascadeController:
    """Tuned cascade for :func:`cascade.aerobatic_reference`, gains found by closed-loop step
    response at 12 m/s / 20 m from :func:`cascade.trim_straight_flight`.

    Aileron and rudder command roll and yaw directly (positive channel drives positive rate);
    elevator commands pitch with a flipped sign, since positive elevator channel is trailing-edge
    down and drives a nose-down (negative) pitch rate here, confirmed with
    :func:`cascade.aerodynamic_sweep` and a short open-loop rollout from trim. Rate gains:
    ``kp=[0.12, 0.20, 0.35]``, ``ki=[1.0, 2.0, 3.0]``, ``kd=[0.001, 0.005, 0.005]``,
    ``integral_limit=0.6``, ``feedforward=[0.02, 0.02, 0.02]`` — pitch and yaw need higher gain
    than roll because their surfaces have less authority per unit channel at this trim. Attitude
    gains: ``kp=[4.0, 10.0, 3.0]``, ``rate_limit=[3.0, 4.0, 2.0]`` rad/s — pitch also needs a
    higher attitude ``kp`` than roll to settle in time, since the airframe's short-period mode is
    only lightly damped (``damping_ratio`` about 0.33 from :func:`cascade.linearize_step`).
    Guidance gains: ``airspeed_kp=0.12``, ``airspeed_ki=0.06``, ``throttle_trim=0.586``,
    ``throttle_limits=[0.0, 1.0]``, ``altitude_kp=1.0``, ``climb_rate_limit=4.0``,
    ``pitch_trim=0.086`` rad (the trimmed 4.9 deg angle of attack), ``pitch_limits=[-0.35, 0.35]``
    rad, ``heading_kp=1.0``, ``bank_limit=0.45`` rad, ``airspeed_pitch_kp=0.05``.
    """

    from cascade.reference import aerobatic_reference_spec

    channels = channel_map(
        aerobatic_reference_spec(),
        roles={"aileron": "roll", "elevator": "-pitch", "rudder": "yaw"},
        limits=1.0,
    )
    rate = RateGains(
        kp=jnp.array([0.12, 0.20, 0.35]),
        ki=jnp.array([1.0, 2.0, 3.0]),
        kd=jnp.array([0.001, 0.005, 0.005]),
        integral_limit=jnp.array([0.6, 0.6, 0.6]),
        feedforward=jnp.array([0.02, 0.02, 0.02]),
    )
    attitude = AttitudeGains(kp=jnp.array([4.0, 10.0, 3.0]), rate_limit=jnp.array([3.0, 4.0, 2.0]))
    guidance = GuidanceGains(
        airspeed_kp=jnp.asarray(0.12),
        airspeed_ki=jnp.asarray(0.06),
        throttle_trim=jnp.asarray(0.586),
        throttle_limits=jnp.array([0.0, 1.0]),
        altitude_kp=jnp.asarray(1.0),
        climb_rate_limit=jnp.asarray(4.0),
        pitch_trim=jnp.asarray(0.086),
        pitch_limits=jnp.array([-0.35, 0.35]),
        heading_kp=jnp.asarray(1.0),
        bank_limit=jnp.asarray(0.45),
        airspeed_pitch_kp=jnp.asarray(0.05),
    )
    return CascadeController(
        channels=channels,
        rate=rate,
        attitude=attitude,
        guidance=guidance,
        rate_period=1,
        attitude_period=2,
        guidance_period=10,
    )


def skywalker_x8_controller() -> CascadeController:
    """Tuned cascade for :func:`cascade.skywalker_x8`, gains found by closed-loop step response
    at 18 m/s / 100 m from :func:`cascade.trim_straight_flight`.

    The X8 has only aileron and elevator channels (no rudder); its yaw column is all zero and the
    rate loop's commanded yaw authority is unused. Elevator is trailing-edge-down positive, a
    nose-down pitching moment, so like the reference aircraft it maps with a flipped sign,
    confirmed with :func:`cascade.aerodynamic_sweep` and a short open-loop rollout from trim.
    Channel limits are narrower than the physical ``0.7`` rad elevon limit (``0.35`` rad each) so
    a combined aileron-plus-elevator command cannot drive one elevon past its own stall angle.
    Rate gains: ``kp=[0.5, 2.0, 0.0]``, ``ki=[2.0, 5.0, 0.0]``, ``kd=[0.0, 0.08, 0.0]``,
    ``integral_limit=0.3``, ``feedforward=[0.02, 0.02, 0.0]``. The pitch axis carries real
    derivative gain (roll needs none) because the X8's short-period mode is very lightly damped
    (``damping_ratio`` about 0.10, versus 0.33 for the reference aircraft, from
    :func:`cascade.linearize_step`); without it, pitch-rate tracking rings up the mode instead of
    damping it. Attitude gains: ``kp=[8.0, 3.0, 1.0]``, ``rate_limit=[4.0, 1.5, 1.0]`` rad/s.
    Guidance gains: ``airspeed_kp=0.08``, ``airspeed_ki=0.03``, ``throttle_trim=0.437``,
    ``throttle_limits=[0.0, 1.0]``, ``altitude_kp=0.35``, ``climb_rate_limit=3.0``,
    ``pitch_trim=0.024`` rad (the trimmed 1.4 deg angle of attack), ``pitch_limits=[-0.2, 0.2]``
    rad, ``heading_kp=0.8``, ``bank_limit=0.4`` rad, ``airspeed_pitch_kp=0.05``.
    """

    from cascade.reference import skywalker_x8_spec

    channels = channel_map(
        skywalker_x8_spec(),
        roles={"aileron": "roll", "elevator": "-pitch"},
        limits={"aileron": 0.35, "elevator": 0.35},
    )
    rate = RateGains(
        kp=jnp.array([0.5, 2.0, 0.0]),
        ki=jnp.array([2.0, 5.0, 0.0]),
        kd=jnp.array([0.0, 0.08, 0.0]),
        integral_limit=jnp.array([0.3, 0.3, 0.3]),
        feedforward=jnp.array([0.02, 0.02, 0.0]),
    )
    attitude = AttitudeGains(kp=jnp.array([8.0, 3.0, 1.0]), rate_limit=jnp.array([4.0, 1.5, 1.0]))
    guidance = GuidanceGains(
        airspeed_kp=jnp.asarray(0.08),
        airspeed_ki=jnp.asarray(0.03),
        throttle_trim=jnp.asarray(0.437),
        throttle_limits=jnp.array([0.0, 1.0]),
        altitude_kp=jnp.asarray(0.35),
        climb_rate_limit=jnp.asarray(3.0),
        pitch_trim=jnp.asarray(0.024),
        pitch_limits=jnp.array([-0.2, 0.2]),
        heading_kp=jnp.asarray(0.8),
        bank_limit=jnp.asarray(0.4),
        airspeed_pitch_kp=jnp.asarray(0.05),
    )
    return CascadeController(
        channels=channels,
        rate=rate,
        attitude=attitude,
        guidance=guidance,
        rate_period=1,
        attitude_period=2,
        guidance_period=10,
    )


def tailsitter_reference_controller(spec: AircraftSpec) -> TransitionController:
    """Transition controller for :func:`cascade.tailsitter_reference`.

    Gains were set by closed-loop step responses in hover and at 7 m/s cruise; see
    ``docs/tailsitter.md``. Elevator is trailing-edge-down positive and therefore nose-down, so
    its pitch role is negative, as on the other packaged aircraft.
    """

    return TransitionController(
        channels=channel_map(spec, roles={"aileron": "roll", "elevator": "-pitch"}, limits=1.0),
        rate=RateGains(
            kp=jnp.array([0.25, 0.25, 0.15]),
            ki=jnp.array([0.5, 0.5, 0.15]),
            kd=jnp.array([0.0, 0.0, 0.0]),
            integral_limit=jnp.array([0.4, 0.4, 0.3]),
            feedforward=jnp.array([0.0, 0.0, 0.0]),
        ),
        attitude=AttitudeGains(
            kp=jnp.array([4.0, 4.0, 2.0]), rate_limit=jnp.array([4.0, 4.0, 2.0])
        ),
        hover=default_hover_gains()._replace(wing_speed_m_s=jnp.asarray(9.0)),
        forward=GuidanceGains(
            airspeed_kp=jnp.asarray(0.1),
            airspeed_ki=jnp.asarray(0.05),
            throttle_trim=jnp.asarray(0.58),
            throttle_limits=jnp.array([0.2, 1.0]),
            altitude_kp=jnp.asarray(0.5),
            climb_rate_limit=jnp.asarray(1.5),
            pitch_trim=jnp.asarray(0.115),
            pitch_limits=jnp.array([-0.35, 0.6]),
            heading_kp=jnp.asarray(1.0),
            bank_limit=jnp.asarray(0.5),
            airspeed_pitch_kp=jnp.asarray(0.05),
        ),
        switch_airspeed_m_s=jnp.asarray(6.5),
        switch_width_m_s=jnp.asarray(0.5),
        differential_thrust=jnp.array([0.5, -0.5]),
    )


__all__ = [
    "aerobatic_reference_controller",
    "skywalker_x8_controller",
    "tailsitter_reference_controller",
]
