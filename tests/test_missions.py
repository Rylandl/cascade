import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cascade.control import GuidanceSetpoint, aerobatic_reference_controller, cascade_step
from cascade.env import EpisodeConfig, cascade_policy, observation, reset, step, tracking_task
from cascade.env.episode import current_environment
from cascade.env.missions import orbit_task, scheduled_tracking_task, waypoint_task
from cascade.env.tasks import task_at
from cascade.initialization import zero_state
from cascade.reference import aerobatic_reference


@pytest.fixture(scope="module")
def fixture():
    model = aerobatic_reference()
    task = tracking_task(12.0, 50.0)
    reference = task.reference(model)
    config = EpisodeConfig(
        horizon_steps=4,
        reset_position_std_m=0.0,
        reset_velocity_std_m_s=0.0,
        reset_attitude_std_rad=0.0,
        reset_rate_std_rad_s=0.0,
    )
    return model, reference, config


def test_schedule_interpolates_short_heading_arc_and_holds_endpoints():
    mission = scheduled_tracking_task(
        [0.0, 2.0], [12.0, 16.0], [50.0, 60.0], np.deg2rad([170.0, -170.0])
    )
    targets = jax.jit(jax.vmap(mission.at))(jnp.array([-1.0, 1.0, 3.0]))
    np.testing.assert_allclose(targets.airspeed_m_s, [12.0, 14.0, 16.0])
    np.testing.assert_allclose(targets.altitude_m, [50.0, 55.0, 60.0])
    np.testing.assert_allclose(np.rad2deg(targets.heading_rad), [170.0, 180.0, 190.0], atol=2e-5)
    static = tracking_task(12.0, 50.0)
    assert task_at(static, 999.0) is static


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (scheduled_tracking_task, {"times_s": [1.0, 2.0]}),
        (scheduled_tracking_task, {"times_s": [0.0, 0.0]}),
        (scheduled_tracking_task, {"times_s": [0.0, np.inf]}),
        (scheduled_tracking_task, {"times_s": [0.0]}),
        (scheduled_tracking_task, {"airspeed_m_s": 0.0}),
        (scheduled_tracking_task, {"airspeed_m_s": [12.0, 13.0, 14.0]}),
        (scheduled_tracking_task, {"altitude_m": np.nan}),
        (scheduled_tracking_task, {"heading_weight": -1.0}),
        (waypoint_task, {"positions_ned": [[0, 0, -50], [0, 0, -60]]}),
        (waypoint_task, {"positions_ned": [[0, 0], [1, 1]]}),
        (waypoint_task, {"lookahead_s": 0.0}),
        (waypoint_task, {"position_scale_m": -1.0}),
        (orbit_task, {"radius_m": 0.0}),
        (orbit_task, {"center_ned": [0.0, np.nan, 0.0]}),
        (orbit_task, {"clockwise": "yes"}),
        (orbit_task, {"capture_distance_m": 0.0}),
    ],
)
def test_host_builders_reject_invalid_missions(factory, kwargs):
    defaults = {
        scheduled_tracking_task: dict(times_s=[0.0, 1.0], airspeed_m_s=12.0, altitude_m=50.0),
        waypoint_task: dict(
            times_s=[0.0, 1.0], positions_ned=[[0, 0, -50], [12, 0, -50]], airspeed_m_s=12.0
        ),
        orbit_task: dict(center_ned=[0, 0, -50], radius_m=100.0, airspeed_m_s=12.0),
    }
    with pytest.raises(ValueError):
        factory(**(defaults[factory] | kwargs))


def test_host_builder_rejects_overflow_after_precision_conversion():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="active JAX dtype"):
            scheduled_tracking_task([0, 1], 12.0, 1e100)
        with pytest.raises(ValueError, match="increasing"):
            scheduled_tracking_task([0, 1, 1 + 1e-10], 12.0, 50.0)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_poststep_reward_and_observation_share_target_and_vmap_matches(fixture):
    model, reference, config = fixture
    mission = scheduled_tracking_task([0.0, 0.1], [12.0, 14.0], [50.0, 54.0], [0.0, 0.4])
    state, _ = reset(model, config, mission, reference, jax.random.PRNGKey(1))
    action = jnp.zeros(model.n_propellers + model.n_control_channels)
    advance = jax.jit(lambda s: step(model, config, mission, reference, s, action))
    next_state, obs, reward, _, info = advance(state)
    assert float(next_state.time_s) == pytest.approx(1.0 / config.control_frequency_hz)
    target = mission.at(next_state.time_s, next_state.aircraft.rigid_body)
    expected_cost = target.cost(
        next_state.aircraft.rigid_body, current_environment(reference, state), action
    )
    np.testing.assert_allclose(info["cost"], expected_cost, atol=1e-6)
    np.testing.assert_allclose(reward, jnp.exp(-expected_cost), atol=1e-6)
    np.testing.assert_allclose(obs, observation(model, target, reference, next_state), atol=1e-6)
    states = jax.tree.map(lambda leaf: jnp.stack([leaf, leaf]), state)
    batch = jax.jit(jax.vmap(advance))(states)
    np.testing.assert_allclose(batch[1], jnp.stack([obs, obs]), atol=1e-6)
    np.testing.assert_allclose(batch[4]["cost"], jnp.repeat(expected_cost, 2), atol=1e-6)


def test_baseline_uses_current_mission_target(fixture):
    model, reference, config = fixture
    mission = scheduled_tracking_task([0.0, 1.0], [12.0, 14.0], [50.0, 60.0], [0.0, 0.5])
    state, obs = reset(model, config, mission, reference, jax.random.PRNGKey(1))
    state = state._replace(time_s=jnp.asarray(0.5), step=jnp.asarray(20))
    controller = aerobatic_reference_controller()
    policy, policy_state = cascade_policy(controller, model, config, mission, reference)
    action, _ = jax.jit(policy)(policy_state, obs, state)
    target = mission.at(state.time_s, state.aircraft.rigid_body)
    expected, _ = cascade_step(
        controller._replace(rate_period=1, attitude_period=1, guidance_period=1),
        policy_state,
        GuidanceSetpoint(target.airspeed_m_s, target.altitude_m, target.heading_rad),
        state.aircraft,
        current_environment(reference, state),
        1.0 / config.control_frequency_hz,
    )
    from cascade.env import control_to_action

    np.testing.assert_allclose(action, control_to_action(config, expected), atol=1e-6)


def test_waypoint_position_cost_and_lookahead_heading(fixture):
    model, reference, _ = fixture
    mission = waypoint_task(
        [0, 10, 20],
        [[0, 0, -50], [100, 0, -60], [100, 100, -60]],
        12.0,
        airspeed_weight=0,
        altitude_weight=0,
        heading_weight=0,
        rate_weight=0,
        effort_weight=0,
    )
    rigid = reference.state.rigid_body._replace(position=jnp.array([50.0, -10.0, -55.0]))
    target = jax.jit(mission.at)(5.0, rigid)
    np.testing.assert_allclose(target.position_ned, [50.0, 0.0, -55.0])
    np.testing.assert_allclose(target.position_error(rigid), [0.0, -10.0, 0.0])
    assert float(target.heading_rad) == pytest.approx(np.arctan2(10, 20))
    assert float(target.cost(rigid, reference.environment, jnp.zeros(4))) == pytest.approx(1.0)
    initial = mission.reference(model)
    np.testing.assert_allclose(initial.state.rigid_body.position, [0.0, 0.0, -50.0])


@pytest.mark.parametrize("clockwise,sign", [(True, 1.0), (False, -1.0)])
def test_orbit_tangent_radial_capture_and_cost(fixture, clockwise, sign):
    model, reference, _ = fixture
    mission = orbit_task(
        [10, 20, -50],
        100,
        12,
        clockwise=clockwise,
        airspeed_weight=0,
        altitude_weight=0,
        heading_weight=0,
        rate_weight=0,
        effort_weight=0,
    )
    rigid = zero_state(model).rigid_body._replace(position=jnp.array([130.0, 20.0, -50.0]))
    target = jax.jit(mission.at)(3.0, rigid)
    np.testing.assert_allclose(target.position_ned, [110.0, 20.0, -50.0])
    expected = sign * (np.pi / 2 + np.arctan2(20, 100))
    assert float(target.heading_rad) == pytest.approx(expected)
    assert float(target.cost(rigid, reference.environment, jnp.zeros(4))) == pytest.approx(0.04)
    at_center = jax.jit(mission.at)(0.0, rigid._replace(position=mission.center_ned))
    assert np.all(np.isfinite(at_center.heading_rad))
    gradient = jax.grad(
        lambda position: mission.at(0.0, rigid._replace(position=position)).heading_rad
    )(mission.center_ned)
    assert np.all(np.isfinite(gradient))
    initial = mission.reference(model)
    np.testing.assert_allclose(initial.state.rigid_body.position, [110.0, 20.0, -50.0])
    assert sign * float(initial.state.rigid_body.velocity[1]) > 0


def test_vmap_across_distinct_schedule_pytrees():
    first = scheduled_tracking_task([0, 2], [12, 14], 50, [0, 0.2])
    second = scheduled_tracking_task([0, 2], [16, 18], 70, [0, -0.2])
    stacked = jax.tree.map(lambda a, b: jnp.stack([a, b]), first, second)
    targets = jax.jit(jax.vmap(lambda task: task_at(task, 1.0)))(stacked)
    np.testing.assert_allclose(targets.airspeed_m_s, [13.0, 17.0])
    np.testing.assert_allclose(targets.heading_rad, [0.1, -0.1])


def test_gain_scheduled_baseline_uses_measured_speed_and_keeps_state(fixture):
    from cascade.control import build_gain_schedule, controller_at_speed
    from cascade.env import control_to_action
    from cascade.math import safe_norm

    model, reference, config = fixture
    mission = scheduled_tracking_task([0, 1], [12, 18], 50, [0, 0.3])
    low = aerobatic_reference_controller()
    high = low._replace(rate=low.rate._replace(kp=2 * low.rate.kp))
    schedule = build_gain_schedule([10, 20], [low, high])
    policy, policy_state = cascade_policy(schedule, model, config, mission, reference)
    policy_state = policy_state._replace(
        rate=policy_state.rate._replace(integral=jnp.array([0.1, -0.1, 0.05]))
    )
    state, obs = reset(model, config, mission, reference, jax.random.PRNGKey(0))
    state = state._replace(time_s=jnp.asarray(0.5), step=jnp.asarray(20))
    action, next_policy = jax.jit(policy)(policy_state, obs, state)
    environment = current_environment(reference, state)
    speed = safe_norm(state.aircraft.rigid_body.velocity - environment.wind)
    active = controller_at_speed(schedule, speed)._replace(
        rate_period=1, attitude_period=1, guidance_period=1
    )
    target = task_at(mission, state.time_s, state.aircraft.rigid_body)
    expected, expected_state = cascade_step(
        active,
        policy_state,
        GuidanceSetpoint(target.airspeed_m_s, target.altitude_m, target.heading_rad),
        state.aircraft,
        environment,
        1.0 / config.control_frequency_hz,
    )
    np.testing.assert_allclose(action, control_to_action(config, expected), atol=1e-6)
    np.testing.assert_allclose(next_policy.rate.integral, expected_state.rate.integral, atol=1e-6)
