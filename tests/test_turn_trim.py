import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cascade
from cascade.analysis import SteadyTurnCondition, trim_steady_turn
from cascade.math import quaternion_from_euler, quaternion_multiply


@pytest.mark.parametrize(
    "factory,speed", [(cascade.aerobatic_reference, 12.0), (cascade.skywalker_x8, 18.0)]
)
@pytest.mark.parametrize("rate", [-0.2, 0.2])
def test_turn_relative_equilibrium_matches_analytic_circle_under_rollout(factory, speed, rate):
    model = factory()
    environment = cascade.standard_environment()
    condition = SteadyTurnCondition(speed, rate, heading_rad=0.4, altitude_m=50.0)
    trim = trim_steady_turn(model, condition)
    assert trim.success, trim.message
    assert float(trim.decision[0]) * rate > 0
    assert trim.acceleration_norm < 1e-4
    assert trim.angular_acceleration_norm < 1e-4
    dynamics = jax.jit(cascade.evaluate_dynamics)(model, trim.state, trim.control, environment)
    target_acceleration = np.cross([0, 0, rate], np.asarray(trim.state.rigid_body.velocity))
    assert np.linalg.norm(np.asarray(dynamics.derivative.rigid_body.velocity)) > 1
    assert np.allclose(dynamics.derivative.rigid_body.velocity, target_acceleration, atol=1e-4)
    final, trajectory = jax.jit(cascade.rollout)(
        model, trim.state, cascade.repeat_control(trim.control, 400), environment, 0.005
    )
    t = 2.0
    heading = condition.heading_rad
    expected_position = np.array(
        [
            speed / rate * (np.sin(heading + rate * t) - np.sin(heading)),
            speed / rate * (np.cos(heading) - np.cos(heading + rate * t)),
            -50.0,
        ]
    )
    expected_velocity = speed * np.array(
        [np.cos(heading + rate * t), np.sin(heading + rate * t), 0]
    )
    expected_attitude = quaternion_multiply(
        quaternion_from_euler(0.0, 0.0, rate * t), trim.state.rigid_body.attitude
    )
    assert np.allclose(final.rigid_body.position, expected_position, atol=0.003)
    assert np.allclose(final.rigid_body.velocity, expected_velocity, atol=0.003)
    assert abs(float(jnp.dot(final.rigid_body.attitude, expected_attitude))) > 0.99999
    assert np.allclose(
        trajectory.rigid_body.angular_velocity, trim.state.rigid_body.angular_velocity, atol=0.003
    )


def test_turn_climb_and_wind_follow_a_translated_helix():
    model = cascade.aerobatic_reference()
    environment = cascade.standard_environment()._replace(wind=jnp.array([1.5, -2.0, 0.2]))
    speed, rate, gamma, duration = 14.0, 0.15, 0.06, 1.0
    trim = trim_steady_turn(
        model,
        SteadyTurnCondition(speed, rate, flight_path_angle_rad=gamma, altitude_m=50),
        environment,
    )
    assert trim.success, trim.message
    final, _ = jax.jit(cascade.rollout)(
        model, trim.state, cascade.repeat_control(trim.control, 200), environment, 0.005
    )
    horizontal = speed * np.cos(gamma)
    expected = (
        np.array(
            [
                horizontal / rate * np.sin(rate * duration),
                horizontal / rate * (1 - np.cos(rate * duration)),
                -50 - speed * np.sin(gamma) * duration,
            ]
        )
        + np.asarray(environment.wind) * duration
    )
    assert np.allclose(final.rigid_body.position, expected, atol=0.002)


def test_zero_turn_is_exactly_the_straight_trim():
    model = cascade.aerobatic_reference()
    condition = SteadyTurnCondition(12, 0.0, heading_rad=0.3, altitude_m=30)
    straight = cascade.trim_straight_flight(model, condition.straight_condition())
    turn = trim_steady_turn(model, condition)
    assert turn.success == straight.success
    assert np.array_equal(turn.decision, straight.decision)
    for a, b in zip(jax.tree.leaves(turn.state), jax.tree.leaves(straight.state), strict=True):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("speed,rate", [(0, 0.1), (-1, 0.1), (12, np.nan), (12, np.inf)])
def test_invalid_turn_conditions_rejected(speed, rate):
    with pytest.raises(ValueError):
        trim_steady_turn(cascade.aerobatic_reference(), SteadyTurnCondition(speed, rate))


def test_turn_requires_rotationally_invariant_gravity_and_reports_failed_candidates():
    model = cascade.aerobatic_reference()
    condition = SteadyTurnCondition(12, 0.2)
    invalid = cascade.standard_environment()._replace(gravity=jnp.array([1.0, 0.0, 9.81]))
    with pytest.raises(ValueError, match="vertical"):
        trim_steady_turn(model, condition, invalid)
    failed = trim_steady_turn(model, condition, max_evaluations=1)
    assert not failed.success
    assert not failed.optimizer_success
    assert np.all(np.isfinite(failed.residual))
