import jax
import jax.numpy as jnp
import numpy as np

import cascade
from cascade.control import GuidanceSetpoint
from cascade.math import quaternion_from_euler, quaternion_rotate
from cascade.vtol import (
    HoverSetpoint,
    initial_transition_state,
    speed_profile_schedule,
    tailsitter_reference_controller,
    transition_rollout,
    trapezoid_speed_profile,
    velocity_ramp_schedule,
)

DT = 0.01


def setup():
    spec = cascade.tailsitter_reference_spec()
    model = spec.to_model()
    environment = cascade.standard_environment()
    state = cascade.zero_state(model, altitude=1.5)
    state = state._replace(
        rigid_body=state.rigid_body._replace(attitude=quaternion_from_euler(0.0, jnp.pi / 2.0, 0.0))
    )
    control = cascade.ControlInput(propeller=jnp.array([0.78, 0.78]), channel=jnp.zeros(2))
    state = cascade.equilibrate_internal_state(model, state, control, environment)
    return model, environment, tailsitter_reference_controller(spec), state


def tilt_degrees(trajectory):
    x_body = jax.vmap(lambda q: quaternion_rotate(q, jnp.array([1.0, 0.0, 0.0])))(
        trajectory.rigid_body.attitude
    )
    return np.degrees(np.arccos(np.clip(np.asarray(-x_body[:, 2]), -1.0, 1.0)))


def test_hover_hold_stays_put_upright_and_finite():
    model, environment, controller, state = setup()
    steps = 600
    hover = HoverSetpoint(
        position_ned=jnp.broadcast_to(jnp.array([0.0, 0.0, -1.5]), (steps, 3)),
        velocity_ned=jnp.zeros((steps, 3)),
        azimuth_rad=jnp.zeros(steps),
        tilt_forward_rad=jnp.zeros(steps),
    )
    forward = GuidanceSetpoint(
        airspeed_m_s=jnp.full(steps, 7.0),
        altitude_m=jnp.full(steps, 1.5),
        heading_rad=jnp.zeros(steps),
    )

    (final, _), (trajectory, controls, weight) = jax.jit(transition_rollout)(
        model, controller, state, initial_transition_state(state), hover, forward, environment, DT
    )

    position = np.asarray(trajectory.rigid_body.position)
    assert np.all(np.isfinite(position))
    assert np.max(np.linalg.norm(position - np.array([0.0, 0.0, -1.5]), axis=1)) < 0.5
    assert tilt_degrees(trajectory).max() < 10.0
    assert float(jnp.max(weight)) < 0.05
    assert float(jnp.max(jnp.abs(controls.channel))) < 0.3


def transition(model, environment, controller, state, acceleration):
    steps = 800
    hover = velocity_ramp_schedule(
        steps,
        DT,
        start_position_ned=jnp.array([0.0, 0.0, -1.5]),
        heading_rad=jnp.array(0.0),
        cruise_speed_m_s=jnp.array(8.0),
        acceleration_m_s2=acceleration,
        hold_steps=200,
    )
    forward = GuidanceSetpoint(
        airspeed_m_s=jnp.full(steps, 8.0),
        altitude_m=jnp.full(steps, 1.5),
        heading_rad=jnp.zeros(steps),
    )
    return transition_rollout(
        model, controller, state, initial_transition_state(state), hover, forward, environment, DT
    )


def test_transition_reaches_cruise_from_hover():
    model, environment, controller, state = setup()
    fly = jax.jit(
        lambda acceleration: transition(model, environment, controller, state, acceleration)
    )
    (final, _), (trajectory, controls, weight) = fly(jnp.asarray(3.5))

    position = np.asarray(trajectory.rigid_body.position)
    speed = np.linalg.norm(np.asarray(trajectory.rigid_body.velocity), axis=1)
    tilt = tilt_degrees(trajectory)
    assert np.all(np.isfinite(position))
    assert abs(speed[-1] - 8.0) < 1.0
    assert tilt[-1] > 65.0
    assert float(weight[-1]) > 0.5
    assert np.max(-position[:, 2]) - 1.5 < 2.5
    assert np.min(-position[:, 2]) > 1.0
    assert float(jnp.max(jnp.abs(controls.channel))) <= 1.0
    assert float(jnp.max(controls.propeller)) <= 1.0
    # The forward blend engages only once airspeed builds.
    assert float(jnp.max(weight[:250])) < 0.05


def test_transition_is_differentiable_in_the_ramp_acceleration():
    model, environment, controller, state = setup()

    def final_speed_error(acceleration):
        (final, _), _ = transition(model, environment, controller, state, acceleration)
        return jnp.square(jnp.linalg.norm(final.rigid_body.velocity) - 8.0)

    gradient = jax.jit(jax.grad(final_speed_error))(jnp.asarray(3.5))
    assert jnp.isfinite(gradient)


def test_round_trip_returns_to_hover():
    model, environment, controller, state = setup()
    steps = 1600
    speed_command = trapezoid_speed_profile(
        steps,
        DT,
        hold_steps=200,
        cruise_speed_m_s=jnp.asarray(8.0),
        acceleration_m_s2=jnp.asarray(3.5),
        cruise_steps=300,
        deceleration_m_s2=jnp.asarray(2.0),
    )
    hover = speed_profile_schedule(
        speed_command,
        DT,
        start_position_ned=jnp.array([0.0, 0.0, -1.5]),
        heading_rad=jnp.array(0.0),
        cruise_speed_m_s=jnp.asarray(8.0),
    )
    forward = GuidanceSetpoint(
        airspeed_m_s=jnp.maximum(speed_command, 6.5),
        altitude_m=jnp.full(steps, 1.5),
        heading_rad=jnp.zeros(steps),
    )

    (final, _), (trajectory, controls, weight) = jax.jit(transition_rollout)(
        model, controller, state, initial_transition_state(state), hover, forward, environment, DT
    )

    position = np.asarray(trajectory.rigid_body.position)
    speed = np.linalg.norm(np.asarray(trajectory.rigid_body.velocity), axis=1)
    tilt = tilt_degrees(trajectory)
    assert np.all(np.isfinite(position))
    assert speed[-1] < 0.5
    assert tilt[-1] < 10.0
    assert float(weight[-1]) < 0.05
    assert float(jnp.max(weight)) > 0.8
    assert np.linalg.norm(position[-1] - np.asarray(hover.position_ned[-1])) < 1.0
    assert np.min(-position[:, 2]) > 0.7 and np.max(-position[:, 2]) < 3.5
    assert float(jnp.max(jnp.abs(controls.channel))) < 0.6


def test_environment_sequence_matches_the_constant_path_and_adds_wind():
    from cascade.state import Environment

    model, environment, controller, state = setup()
    steps = 200
    hover = HoverSetpoint(
        position_ned=jnp.broadcast_to(jnp.array([0.0, 0.0, -1.5]), (steps, 3)),
        velocity_ned=jnp.zeros((steps, 3)),
        azimuth_rad=jnp.zeros(steps),
        tilt_forward_rad=jnp.zeros(steps),
    )
    forward = GuidanceSetpoint(
        airspeed_m_s=jnp.full(steps, 8.0),
        altitude_m=jnp.full(steps, 1.5),
        heading_rad=jnp.zeros(steps),
    )
    constant = Environment(
        density=jnp.broadcast_to(environment.density, (steps,)),
        wind=jnp.broadcast_to(environment.wind, (steps, 3)),
        gravity=jnp.broadcast_to(environment.gravity, (steps, 3)),
    )
    run = jax.jit(
        lambda envs: transition_rollout(
            model,
            controller,
            state,
            initial_transition_state(state),
            hover,
            forward,
            environment,
            DT,
            environments=envs,
        )
    )

    (_, _), (plain, _, _) = run(None)
    (_, _), (sequenced, _, _) = run(constant)
    assert jnp.allclose(plain.rigid_body.position, sequenced.rigid_body.position, atol=1e-4)

    gust = constant._replace(wind=constant.wind.at[100:, 1].set(1.0))
    (_, _), (gusty, _, _) = run(gust)
    assert jnp.allclose(gusty.rigid_body.position[:100], plain.rigid_body.position[:100], atol=1e-4)
    assert (
        float(jnp.max(jnp.abs(gusty.rigid_body.position[150:] - plain.rigid_body.position[150:])))
        > 1e-3
    )


def test_hover_holds_against_a_spanwise_wind_with_differential_thrust():
    # A spanwise wind weathervanes the wing about its belly normal; only differential thrust
    # controls that axis in hover, and without it this run falls over within seconds.
    model, environment, controller, state = setup()
    steps = 600
    hover = HoverSetpoint(
        position_ned=jnp.broadcast_to(jnp.array([0.0, 0.0, -1.5]), (steps, 3)),
        velocity_ned=jnp.zeros((steps, 3)),
        azimuth_rad=jnp.zeros(steps),
        tilt_forward_rad=jnp.zeros(steps),
    )
    forward = GuidanceSetpoint(
        airspeed_m_s=jnp.full(steps, 8.0),
        altitude_m=jnp.full(steps, 1.5),
        heading_rad=jnp.zeros(steps),
    )
    windy = environment._replace(wind=jnp.array([0.0, 1.0, 0.0]))

    (_, _), (trajectory, controls, _) = jax.jit(transition_rollout)(
        model, controller, state, initial_transition_state(state), hover, forward, windy, DT
    )

    position = np.asarray(trajectory.rigid_body.position)
    assert np.all(np.isfinite(position))
    assert np.max(np.linalg.norm(position - np.array([0.0, 0.0, -1.5]), axis=1)) < 0.6
    assert tilt_degrees(trajectory).max() < 15.0
    # The propellers actually split to do it.
    assert float(jnp.max(jnp.abs(controls.propeller[:, 0] - controls.propeller[:, 1]))) > 0.01


def test_round_trip_survives_dryden_gusts():
    from cascade.gusts import dryden_environment_sequence, dryden_low_altitude

    model, environment, controller, state = setup()
    steps = 1600
    speed_command = trapezoid_speed_profile(
        steps,
        DT,
        hold_steps=200,
        cruise_speed_m_s=jnp.asarray(8.0),
        acceleration_m_s2=jnp.asarray(3.5),
        cruise_steps=300,
        deceleration_m_s2=jnp.asarray(2.0),
    )
    hover = speed_profile_schedule(
        speed_command,
        DT,
        start_position_ned=jnp.array([0.0, 0.0, -1.5]),
        heading_rad=jnp.array(0.0),
        cruise_speed_m_s=jnp.asarray(8.0),
    )
    forward = GuidanceSetpoint(
        airspeed_m_s=jnp.maximum(speed_command, 6.5),
        altitude_m=jnp.full(steps, 1.5),
        heading_rad=jnp.zeros(steps),
    )
    gusts = dryden_environment_sequence(
        jax.random.PRNGKey(1),
        environment,
        steps,
        DT,
        airspeed_m_s=jnp.asarray(4.0),
        parameters=dryden_low_altitude(jnp.asarray(1.5), jnp.asarray(2.0)),
    )

    (_, _), (trajectory, controls, weight) = jax.jit(transition_rollout)(
        model,
        controller,
        state,
        initial_transition_state(state),
        hover,
        forward,
        environment,
        DT,
        environments=gusts,
    )

    position = np.asarray(trajectory.rigid_body.position)
    speed = np.linalg.norm(np.asarray(trajectory.rigid_body.velocity), axis=1)
    assert np.all(np.isfinite(position))
    assert float(jnp.max(weight)) > 0.8
    assert speed[-1] < 1.5
    assert np.min(-position[:, 2]) > 0.4
    assert np.linalg.norm(position[-1] - np.asarray(hover.position_ned[-1])) < 3.0
    assert float(jnp.max(jnp.abs(controls.channel))) <= 1.0


def test_coordinated_turn_rates_match_the_banked_turn():
    from cascade.control import coordinated_turn_rates

    level = quaternion_from_euler(0.0, 0.0, 0.0)
    assert jnp.allclose(coordinated_turn_rates(level, jnp.asarray(10.0), jnp.asarray(9.81)), 0.0)
    banked = quaternion_from_euler(jnp.deg2rad(30.0), 0.0, 0.0)
    rates = coordinated_turn_rates(banked, jnp.asarray(10.0), jnp.asarray(9.81))
    turn_rate = 9.81 * jnp.tan(jnp.deg2rad(30.0)) / 10.0
    assert jnp.allclose(
        rates, jnp.array([0.0, turn_rate * 0.5, turn_rate * jnp.cos(jnp.deg2rad(30.0))]), atol=1e-6
    )
    # Hover airspeeds are floored, so the feedforward stays bounded.
    slow = coordinated_turn_rates(banked, jnp.asarray(0.0), jnp.asarray(9.81))
    assert jnp.all(jnp.abs(slow) < 3.0)


def test_round_trip_with_a_turn_in_cruise():
    model, environment, controller, state = setup()
    steps = 2000
    speed_command = trapezoid_speed_profile(
        steps,
        DT,
        hold_steps=200,
        cruise_speed_m_s=jnp.asarray(8.0),
        acceleration_m_s2=jnp.asarray(3.5),
        cruise_steps=700,
        deceleration_m_s2=jnp.asarray(2.0),
    )
    time = jnp.arange(steps) * DT
    heading = jnp.deg2rad(90.0) * jnp.clip((time - 6.0) / 3.0, 0.0, 1.0)
    hover = speed_profile_schedule(
        speed_command,
        DT,
        start_position_ned=jnp.array([0.0, 0.0, -1.5]),
        heading_rad=heading,
        cruise_speed_m_s=jnp.asarray(8.0),
    )
    forward = GuidanceSetpoint(
        airspeed_m_s=jnp.maximum(speed_command, 6.5),
        altitude_m=jnp.full(steps, 1.5),
        heading_rad=heading,
    )

    (final, _), (trajectory, controls, weight) = jax.jit(transition_rollout)(
        model, controller, state, initial_transition_state(state), hover, forward, environment, DT
    )

    position = np.asarray(trajectory.rigid_body.position)
    velocity = np.asarray(trajectory.rigid_body.velocity)
    speed = np.linalg.norm(velocity, axis=1)
    assert np.all(np.isfinite(position))
    # Track follows the heading ramp: east-bound by the end of cruise.
    k = int(11.0 / DT)
    track = np.degrees(np.arctan2(velocity[k, 1], velocity[k, 0]))
    assert abs(track - 90.0) < 5.0
    assert np.min(-position[:, 2]) > 0.7
    # Coordinated: the motors do not split against the turn.
    turning = slice(int(6.5 / DT), int(9.5 / DT))
    split = np.abs(np.asarray(controls.propeller[turning, 0] - controls.propeller[turning, 1]))
    assert split.max() < 0.03
    assert speed[-1] < 0.5
    assert np.linalg.norm(position[-1] - np.asarray(hover.position_ned[-1])) < 1.0


def test_hover_azimuth_across_wind_puts_the_span_into_the_wind():
    from cascade.vtol import hover_azimuth_across_wind

    north_wind = jnp.array([-3.0, 0.0, 0.0])  # air moving south: wind from the north
    azimuth = hover_azimuth_across_wind(north_wind, jnp.asarray(0.1))
    # Perpendicular to the wind, on the side nearer the preferred azimuth.
    assert abs(float(jnp.cos(azimuth - jnp.pi))) < 1e-3 or abs(float(jnp.cos(azimuth))) < 1e-3
    assert abs(float(azimuth) - jnp.pi / 2) < 1e-3
    west_side = hover_azimuth_across_wind(north_wind, jnp.asarray(-0.1))
    assert abs(float(west_side) + jnp.pi / 2) < 1e-3
    # Calm air keeps the preferred azimuth.
    calm = hover_azimuth_across_wind(jnp.zeros(3), jnp.asarray(0.3))
    assert abs(float(calm) - 0.3) < 1e-3
    # Batched and differentiable in the wind.
    winds = jnp.array([[-3.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    batched = hover_azimuth_across_wind(winds, jnp.zeros(2))
    assert batched.shape == (2,)
    gradient = jax.grad(lambda w: hover_azimuth_across_wind(w, jnp.asarray(0.1)))(north_wind)
    assert jnp.all(jnp.isfinite(gradient))
