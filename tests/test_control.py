"""Closed-loop tests for the control cascade: rate, attitude, guidance, scheduling, batching,
and differentiability. Trims are cached across tests since they are the same two conditions used
throughout (12 m/s / 20 m for the reference aircraft, 18 m/s / 100 m for the X8)."""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

import cascade
from cascade.analysis import StraightFlightCondition, trim_straight_flight
from cascade.control import (
    GuidanceSetpoint,
    RateState,
    aerobatic_reference_controller,
    attitude_controller,
    cascade_step,
    closed_loop_rollout,
    initial_cascade_state,
    rate_controller,
    skywalker_x8_controller,
)
from cascade.initialization import standard_environment
from cascade.integration import rk4_step
from cascade.math import quaternion_from_euler, quaternion_rotate_inverse
from cascade.state import ControlInput

DT = 0.01
X8_STALL_LIMIT_RAD = np.deg2rad(12.0)


@cache
def _reference_trim():
    model = cascade.aerobatic_reference()
    environment = standard_environment()
    trim = trim_straight_flight(
        model, StraightFlightCondition(airspeed_m_s=12.0, altitude_m=20.0), environment
    )
    assert trim.success, trim.message
    return model, environment, trim


@cache
def _x8_trim():
    model = cascade.skywalker_x8()
    environment = standard_environment()
    trim = trim_straight_flight(
        model, StraightFlightCondition(airspeed_m_s=18.0, altitude_m=100.0), environment
    )
    assert trim.success, trim.message
    return model, environment, trim


def _euler_from_quaternion(quaternion):
    x, y, z, w = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    roll = jnp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = jnp.arcsin(jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _angle_of_attack(states, environment):
    velocity_world = states.rigid_body.velocity - environment.wind
    velocity_body = quaternion_rotate_inverse(states.rigid_body.attitude, velocity_world)
    return jnp.arctan2(velocity_body[..., 2], velocity_body[..., 0])


def _rate_only_rollout(model, environment, channels, trim, gains, rate_setpoint, steps, dt=DT):
    """Rate loop alone against the plant: rate setpoint fed directly, throttle held at trim."""

    def step(carry, _):
        state, rate_state = carry
        command, rate_state = rate_controller(
            gains, rate_state, rate_setpoint, state.rigid_body.angular_velocity, dt
        )
        channel = jnp.clip(
            jnp.einsum("cr,r->c", channels.matrix, command), -channels.limit, channels.limit
        )
        control = ControlInput(propeller=trim.control.propeller, channel=channel)
        next_state = rk4_step(model, state, control, environment, dt)
        return (next_state, rate_state), next_state

    rate_state0 = RateState(integral=jnp.zeros(3), previous_error=jnp.zeros(3))
    _, trajectory = jax.lax.scan(step, (trim.state, rate_state0), None, length=steps)
    return trajectory


def _attitude_rollout(
    model, environment, channels, trim, rate_gains, attitude_gains, attitude_setpoint, steps, dt=DT
):
    """Attitude-then-rate loop against the plant, throttle held at trim (guidance bypassed)."""

    def step(carry, _):
        state, rate_state = carry
        rate_setpoint = attitude_controller(
            attitude_gains, attitude_setpoint, state.rigid_body.attitude
        )
        command, rate_state = rate_controller(
            rate_gains, rate_state, rate_setpoint, state.rigid_body.angular_velocity, dt
        )
        channel = jnp.clip(
            jnp.einsum("cr,r->c", channels.matrix, command), -channels.limit, channels.limit
        )
        control = ControlInput(propeller=trim.control.propeller, channel=channel)
        next_state = rk4_step(model, state, control, environment, dt)
        return (next_state, rate_state), next_state

    rate_state0 = RateState(integral=jnp.zeros(3), previous_error=jnp.zeros(3))
    _, trajectory = jax.lax.scan(step, (trim.state, rate_state0), None, length=steps)
    return trajectory


def _rise_time_and_overshoot(values, target, dt):
    """Time to first reach 90% of ``target`` and the fractional overshoot beyond it."""

    values = np.asarray(values)
    threshold = 0.9 * target
    reached = values >= threshold if target > 0 else values <= threshold
    index = int(np.argmax(reached))
    rise_time = index * dt if bool(reached[index]) else None
    peak = values.max() if target > 0 else values.min()
    overshoot = (abs(float(peak)) - abs(target)) / abs(target)
    return rise_time, overshoot


def _settle_time(values, base, target, dt, tolerance_fraction=0.05):
    """First time after which ``values - base`` stays within ``tolerance_fraction`` of target."""

    values = np.asarray(values) - base
    tolerance = tolerance_fraction * abs(target)
    within = np.abs(values - target) <= tolerance
    for index in range(len(within)):
        if np.all(within[index:]):
            return index * dt
    return None


# ---------------------------------------------------------------------------------------------
# 1. Rate loop alone: 1 rad/s roll-rate step, throttle held at trim.
# ---------------------------------------------------------------------------------------------


def test_reference_rate_loop_roll_step_response():
    model, environment, trim = _reference_trim()
    controller = aerobatic_reference_controller()
    trajectory = _rate_only_rollout(
        model, environment, controller.channels, trim, controller.rate,
        jnp.array([1.0, 0.0, 0.0]), steps=80,
    )
    rise_time, overshoot = _rise_time_and_overshoot(
        trajectory.rigid_body.angular_velocity[:, 0], 1.0, DT
    )
    assert rise_time is not None and rise_time <= 0.5
    assert overshoot < 0.25


def test_x8_rate_loop_roll_step_response():
    model, environment, trim = _x8_trim()
    controller = skywalker_x8_controller()
    trajectory = _rate_only_rollout(
        model, environment, controller.channels, trim, controller.rate,
        jnp.array([1.0, 0.0, 0.0]), steps=80,
    )
    rise_time, overshoot = _rise_time_and_overshoot(
        trajectory.rigid_body.angular_velocity[:, 0], 1.0, DT
    )
    assert rise_time is not None and rise_time <= 0.5
    assert overshoot < 0.25


# ---------------------------------------------------------------------------------------------
# 2. Attitude loop: 20 deg roll step and +5 deg pitch step, guidance bypassed.
# ---------------------------------------------------------------------------------------------


def test_reference_attitude_loop_roll_step_response():
    model, environment, trim = _reference_trim()
    controller = aerobatic_reference_controller()
    roll0, pitch0, yaw0 = _euler_from_quaternion(trim.state.rigid_body.attitude)
    setpoint = quaternion_from_euler(roll0 + np.deg2rad(20.0), pitch0, yaw0)
    trajectory = _attitude_rollout(
        model, environment, controller.channels, trim, controller.rate, controller.attitude,
        setpoint, steps=200,
    )
    roll, _, _ = _euler_from_quaternion(trajectory.rigid_body.attitude)
    settle_time = _settle_time(roll, float(roll0), np.deg2rad(20.0), DT)
    overshoot = (
        float(np.abs(np.asarray(roll) - float(roll0)).max()) - np.deg2rad(20.0)
    ) / np.deg2rad(20.0)
    assert settle_time is not None and settle_time <= 2.0
    assert overshoot < 0.25


def test_reference_attitude_loop_pitch_step_response():
    model, environment, trim = _reference_trim()
    controller = aerobatic_reference_controller()
    roll0, pitch0, yaw0 = _euler_from_quaternion(trim.state.rigid_body.attitude)
    setpoint = quaternion_from_euler(roll0, pitch0 + np.deg2rad(5.0), yaw0)
    trajectory = _attitude_rollout(
        model, environment, controller.channels, trim, controller.rate, controller.attitude,
        setpoint, steps=200,
    )
    _, pitch, _ = _euler_from_quaternion(trajectory.rigid_body.attitude)
    settle_time = _settle_time(pitch, float(pitch0), np.deg2rad(5.0), DT)
    overshoot = (
        float(np.abs(np.asarray(pitch) - float(pitch0)).max()) - np.deg2rad(5.0)
    ) / np.deg2rad(5.0)
    assert settle_time is not None and settle_time <= 2.0
    assert overshoot < 0.25


def test_x8_attitude_loop_roll_step_response():
    model, environment, trim = _x8_trim()
    controller = skywalker_x8_controller()
    roll0, pitch0, yaw0 = _euler_from_quaternion(trim.state.rigid_body.attitude)
    setpoint = quaternion_from_euler(roll0 + np.deg2rad(20.0), pitch0, yaw0)
    trajectory = _attitude_rollout(
        model, environment, controller.channels, trim, controller.rate, controller.attitude,
        setpoint, steps=200,
    )
    roll, _, _ = _euler_from_quaternion(trajectory.rigid_body.attitude)
    settle_time = _settle_time(roll, float(roll0), np.deg2rad(20.0), DT)
    overshoot = (
        float(np.abs(np.asarray(roll) - float(roll0)).max()) - np.deg2rad(20.0)
    ) / np.deg2rad(20.0)
    assert settle_time is not None and settle_time <= 2.0
    assert overshoot < 0.25


def test_x8_attitude_loop_pitch_step_response():
    model, environment, trim = _x8_trim()
    controller = skywalker_x8_controller()
    roll0, pitch0, yaw0 = _euler_from_quaternion(trim.state.rigid_body.attitude)
    setpoint = quaternion_from_euler(roll0, pitch0 + np.deg2rad(5.0), yaw0)
    trajectory = _attitude_rollout(
        model, environment, controller.channels, trim, controller.rate, controller.attitude,
        setpoint, steps=200,
    )
    _, pitch, _ = _euler_from_quaternion(trajectory.rigid_body.attitude)
    settle_time = _settle_time(pitch, float(pitch0), np.deg2rad(5.0), DT)
    overshoot = (
        float(np.abs(np.asarray(pitch) - float(pitch0)).max()) - np.deg2rad(5.0)
    ) / np.deg2rad(5.0)
    assert settle_time is not None and settle_time <= 2.0
    assert overshoot < 0.25


# ---------------------------------------------------------------------------------------------
# 3. Guidance: altitude/airspeed steps (no stall) and a coordinated heading turn.
# ---------------------------------------------------------------------------------------------


def _guidance_step_rollout(model, environment, controller, trim, setpoint, steps):
    cascade_state = initial_cascade_state(controller, trim.state, trim.control)
    setpoints = jax.tree.map(lambda value: jnp.broadcast_to(value, (steps,)), setpoint)
    (final_state, _), (states, controls, _) = closed_loop_rollout(
        model, controller, trim.state, cascade_state, setpoints, environment, DT
    )
    return final_state, states, controls


def test_reference_guidance_altitude_and_airspeed_step():
    model, environment, trim = _reference_trim()
    controller = aerobatic_reference_controller()
    setpoint = GuidanceSetpoint(
        airspeed_m_s=jnp.asarray(14.0), altitude_m=jnp.asarray(25.0), heading_rad=jnp.asarray(0.0)
    )
    final_state, states, _ = _guidance_step_rollout(
        model, environment, controller, trim, setpoint, steps=1000
    )

    altitude = -final_state.rigid_body.position[2]
    airspeed = jnp.linalg.norm(final_state.rigid_body.velocity - environment.wind)
    assert abs(float(altitude) - 25.0) < 1.0
    assert abs(float(airspeed) - 14.0) < 0.3

    alpha = _angle_of_attack(states, environment)
    stall_angle = float(model.surfaces.stall_angle.min())
    assert float(jnp.abs(alpha).max()) < stall_angle


def test_x8_guidance_altitude_and_airspeed_step():
    model, environment, trim = _x8_trim()
    controller = skywalker_x8_controller()
    setpoint = GuidanceSetpoint(
        airspeed_m_s=jnp.asarray(20.0), altitude_m=jnp.asarray(105.0), heading_rad=jnp.asarray(0.0)
    )
    final_state, states, _ = _guidance_step_rollout(
        model, environment, controller, trim, setpoint, steps=1000
    )

    altitude = -final_state.rigid_body.position[2]
    airspeed = jnp.linalg.norm(final_state.rigid_body.velocity - environment.wind)
    assert abs(float(altitude) - 105.0) < 1.0
    assert abs(float(airspeed) - 20.0) < 0.3

    alpha = _angle_of_attack(states, environment)
    assert float(jnp.abs(alpha).max()) < X8_STALL_LIMIT_RAD


def test_reference_guidance_heading_turn():
    model, environment, trim = _reference_trim()
    controller = aerobatic_reference_controller()
    setpoint = GuidanceSetpoint(
        airspeed_m_s=jnp.asarray(12.0), altitude_m=jnp.asarray(20.0),
        heading_rad=jnp.asarray(np.deg2rad(45.0)),
    )
    _, states, _ = _guidance_step_rollout(
        model, environment, controller, trim, setpoint, steps=1500
    )

    roll, _, yaw = _euler_from_quaternion(states.rigid_body.attitude)
    yaw_deg = np.rad2deg(np.asarray(yaw))
    settle_time = _settle_time(yaw_deg, 0.0, 45.0, DT)
    assert settle_time is not None and settle_time <= 12.0

    # Attitude tracking allows overshoot beyond the guidance-commanded bank (test 2); a generous
    # margin over bank_limit still confirms the turn respects the commanded envelope.
    assert float(jnp.abs(roll).max()) <= 1.2 * float(controller.guidance.bank_limit)

    altitude = -np.asarray(states.rigid_body.position)[:, 2]
    assert float(np.abs(altitude - 20.0).max()) < 3.0


def test_x8_guidance_heading_turn():
    model, environment, trim = _x8_trim()
    controller = skywalker_x8_controller()
    setpoint = GuidanceSetpoint(
        airspeed_m_s=jnp.asarray(18.0), altitude_m=jnp.asarray(100.0),
        heading_rad=jnp.asarray(np.deg2rad(45.0)),
    )
    _, states, _ = _guidance_step_rollout(
        model, environment, controller, trim, setpoint, steps=1500
    )

    roll, _, yaw = _euler_from_quaternion(states.rigid_body.attitude)
    yaw_deg = np.rad2deg(np.asarray(yaw))
    settle_time = _settle_time(yaw_deg, 0.0, 45.0, DT)
    assert settle_time is not None and settle_time <= 12.0

    assert float(jnp.abs(roll).max()) <= 1.2 * float(controller.guidance.bank_limit)

    altitude = -np.asarray(states.rigid_body.position)[:, 2]
    assert float(np.abs(altitude - 100.0).max()) < 3.0


# ---------------------------------------------------------------------------------------------
# 4. Batching: one batched rollout over 4 worlds matches 4 single rollouts.
# ---------------------------------------------------------------------------------------------


def test_batched_rollout_matches_single_world_rollouts():
    model, environment, trim = _reference_trim()
    controller = aerobatic_reference_controller()
    steps = 150
    airspeeds = jnp.array([12.0, 13.0, 11.0, 12.5])
    altitudes = jnp.array([20.0, 22.0, 18.0, 21.0])
    headings = jnp.array([0.0, 0.1, -0.1, 0.05])

    batch_state = jax.tree.map(lambda value: jnp.broadcast_to(value, (4, *value.shape)), trim.state)
    batch_environment = standard_environment(batch_shape=(4,))
    batch_cascade_state = initial_cascade_state(controller, batch_state, trim.control)
    batch_setpoints = GuidanceSetpoint(
        airspeed_m_s=jnp.broadcast_to(airspeeds, (steps, 4)),
        altitude_m=jnp.broadcast_to(altitudes, (steps, 4)),
        heading_rad=jnp.broadcast_to(headings, (steps, 4)),
    )
    (_, _), (batch_states, batch_controls, _) = closed_loop_rollout(
        model, controller, batch_state, batch_cascade_state, batch_setpoints, batch_environment, DT
    )

    for index in range(4):
        single_cascade_state = initial_cascade_state(controller, trim.state, trim.control)
        single_setpoints = GuidanceSetpoint(
            airspeed_m_s=jnp.full((steps,), airspeeds[index]),
            altitude_m=jnp.full((steps,), altitudes[index]),
            heading_rad=jnp.full((steps,), headings[index]),
        )
        (_, _), (single_states, single_controls, _) = closed_loop_rollout(
            model, controller, trim.state, single_cascade_state, single_setpoints, environment, DT
        )
        assert jnp.allclose(
            batch_states.rigid_body.position[:, index], single_states.rigid_body.position, atol=1e-4
        )
        assert jnp.allclose(
            batch_states.rigid_body.attitude[:, index], single_states.rigid_body.attitude, atol=1e-4
        )
        assert jnp.allclose(
            batch_controls.channel[:, index], single_controls.channel, atol=1e-5
        )
        assert jnp.allclose(
            batch_controls.propeller[:, index], single_controls.propeller, atol=1e-5
        )


# ---------------------------------------------------------------------------------------------
# 5. Differentiability: grad of roll-rate tracking MSE wrt RateGains.kp, then tune it down.
# ---------------------------------------------------------------------------------------------


def test_rate_gain_gradient_is_finite_and_tuning_reduces_error():
    model, environment, trim = _reference_trim()
    controller = aerobatic_reference_controller()
    steps = 100  # 1 s at dt = 0.01
    target = 1.0
    rate_setpoint = jnp.array([target, 0.0, 0.0])

    # Deliberately detuned: far too little proportional authority to track the step.
    detuned = controller.rate._replace(kp=jnp.array([0.02, 0.02, 0.02]))

    def tracking_error(kp):
        gains = detuned._replace(kp=kp)
        trajectory = _rate_only_rollout(
            model, environment, controller.channels, trim, gains, rate_setpoint, steps
        )
        roll_rate = trajectory.rigid_body.angular_velocity[:, 0]
        return jnp.mean(jnp.square(roll_rate - target))

    grad_fn = jax.grad(tracking_error)
    kp = detuned.kp
    gradient = grad_fn(kp)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.any(jnp.abs(gradient) > 0.0))

    initial_error = float(tracking_error(kp))
    learning_rate = 0.02
    for _ in range(20):
        kp = kp - learning_rate * grad_fn(kp)
    final_error = float(tracking_error(kp))
    assert final_error < initial_error


# ---------------------------------------------------------------------------------------------
# 6. Rate scheduling: with attitude_period 2, the attitude loop's rate-setpoint output only
#    changes on every other simulation step.
# ---------------------------------------------------------------------------------------------


def test_attitude_output_updates_only_on_its_own_schedule():
    model, environment, trim = _reference_trim()
    controller = aerobatic_reference_controller()
    assert controller.attitude_period == 2

    cascade_state = initial_cascade_state(controller, trim.state, trim.control)
    setpoint = GuidanceSetpoint(
        airspeed_m_s=jnp.asarray(12.0), altitude_m=jnp.asarray(23.0),
        heading_rad=jnp.asarray(np.deg2rad(6.0)),
    )
    state = trim.state
    held_rate_setpoints = [cascade_state.held_rate_setpoint]
    for _ in range(20):
        control, cascade_state = cascade_step(
            controller, cascade_state, setpoint, state, environment, DT
        )
        held_rate_setpoints.append(cascade_state.held_rate_setpoint)
        state = rk4_step(model, state, control, environment, DT)

    held_rate_setpoints = jnp.stack(held_rate_setpoints)
    changed = jnp.any(jnp.abs(held_rate_setpoints[1:] - held_rate_setpoints[:-1]) > 0.0, axis=-1)
    # changed[call] reflects whether call `call` (0-indexed, using step_index == call before the
    # increment) refreshed the held value; the schedule only allows that when call % 2 == 0.
    odd_calls = changed[1::2]
    assert bool(jnp.all(~odd_calls))
    assert bool(jnp.any(changed))


# ---------------------------------------------------------------------------------------------
# 7. Finite: a 20 s guidance rollout of the reference through a changing setpoint stays finite
#    with every channel command inside its limit.
# ---------------------------------------------------------------------------------------------


def test_reference_long_rollout_stays_finite_and_within_channel_limits():
    model, environment, trim = _reference_trim()
    controller = aerobatic_reference_controller()

    hold = GuidanceSetpoint(
        airspeed_m_s=jnp.full((500,), 12.0), altitude_m=jnp.full((500,), 20.0),
        heading_rad=jnp.zeros((500,)),
    )
    climb_and_speed = GuidanceSetpoint(
        airspeed_m_s=jnp.full((1000,), 14.0), altitude_m=jnp.full((1000,), 25.0),
        heading_rad=jnp.zeros((1000,)),
    )
    turn = GuidanceSetpoint(
        airspeed_m_s=jnp.full((500,), 14.0), altitude_m=jnp.full((500,), 25.0),
        heading_rad=jnp.full((500,), np.deg2rad(45.0)),
    )
    setpoints = jax.tree.map(
        lambda *parts: jnp.concatenate(parts, axis=0), hold, climb_and_speed, turn
    )
    assert jax.tree.leaves(setpoints)[0].shape[0] == 2000  # 20 s at dt = 0.01

    cascade_state = initial_cascade_state(controller, trim.state, trim.control)
    (final_state, final_cascade_state), (states, controls, _) = closed_loop_rollout(
        model, controller, trim.state, cascade_state, setpoints, environment, DT
    )

    for leaf in jax.tree.leaves(states):
        assert bool(jnp.all(jnp.isfinite(leaf)))
    for leaf in jax.tree.leaves(controls):
        assert bool(jnp.all(jnp.isfinite(leaf)))
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(jax.tree.leaves(final_cascade_state)[0]))))

    assert bool(jnp.all(jnp.abs(controls.channel) <= controller.channels.limit + 1e-6))
    assert bool(jnp.all(controls.propeller >= -1e-6))
    assert bool(jnp.all(controls.propeller <= 1.0 + 1e-6))
