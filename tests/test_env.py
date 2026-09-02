import jax
import jax.numpy as jnp
import pytest

from cascade.env import (
    EpisodeConfig,
    action_size,
    action_to_control,
    control_to_action,
    reset,
    rollout_actions,
    step,
    tracking_task,
    trimmed_reference,
)
from cascade.reference import aerobatic_reference

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def setup():
    model = aerobatic_reference()
    task = tracking_task(12.0, 50.0, 0.0)
    reference = trimmed_reference(model, task)
    config = EpisodeConfig(horizon_steps=60)
    return model, config, task, reference


def test_reset_and_step_are_batched_jitted_and_finite(setup):
    model, config, task, reference = setup
    keys = jax.random.split(jax.random.PRNGKey(0), 16)
    batched_reset = jax.jit(jax.vmap(lambda k: reset(model, config, task, reference, k)))
    states, observations = batched_reset(keys)
    assert observations.shape == (16, 17 + model.n_surfaces + model.n_propellers)
    assert jnp.all(jnp.isfinite(observations))

    batched_step = jax.jit(jax.vmap(lambda s, a: step(model, config, task, reference, s, a)))
    actions = jax.random.uniform(
        jax.random.PRNGKey(1), (16, action_size(model)), minval=-1.0, maxval=1.0
    )
    next_states, observations, rewards, dones, info = batched_step(states, actions)
    assert observations.shape == (16, 17 + model.n_surfaces + model.n_propellers)
    assert rewards.shape == (16,) and dones.shape == (16,)
    assert jnp.all(jnp.isfinite(observations)) and jnp.all((rewards >= 0.0) & (rewards <= 1.0))
    assert jnp.all(next_states.step == 1)


def test_action_mapping_round_trips_the_trim(setup):
    model, config, task, reference = setup
    action = control_to_action(config, reference.control)
    control = action_to_control(model, config, action)
    assert jnp.allclose(control.propeller, reference.control.propeller, atol=1e-6)
    assert jnp.allclose(control.channel, reference.control.channel, atol=1e-6)


def test_holding_the_trim_action_tracks_the_reference(setup):
    model, config, task, reference = setup
    quiet = EpisodeConfig(
        horizon_steps=80,
        reset_position_std_m=0.0,
        reset_velocity_std_m_s=0.0,
        reset_attitude_std_rad=0.0,
        reset_rate_std_rad_s=0.0,
    )
    state, _ = reset(model, quiet, task, reference, jax.random.PRNGKey(0))
    actions = jnp.broadcast_to(
        control_to_action(quiet, reference.control), (80, action_size(model))
    )
    final, (observations, rewards, dones) = jax.jit(
        lambda s, a: rollout_actions(model, quiet, task, reference, s, a)
    )(state, actions)
    rigid = final.aircraft.rigid_body
    assert abs(float(-rigid.position[2]) - 50.0) < 1.0
    assert abs(float(jnp.linalg.norm(rigid.velocity)) - 12.0) < 1.0
    assert float(jnp.mean(rewards)) > 0.8
    assert not bool(dones[:-1].any()) and bool(dones[-1])  # only the horizon ends it


def test_crash_and_horizon_terminate(setup):
    model, config, task, reference = setup
    state, _ = reset(model, config, task, reference, jax.random.PRNGKey(2))
    underground = state._replace(
        aircraft=state.aircraft._replace(
            rigid_body=state.aircraft.rigid_body._replace(
                position=state.aircraft.rigid_body.position.at[2].set(5.0)
            )
        )
    )
    action = control_to_action(config, reference.control)
    _, _, reward, done, info = step(model, config, task, reference, underground, action)
    assert bool(done) and bool(info["crashed"]) and float(reward) == 0.0
    at_horizon = state._replace(step=jnp.asarray(config.horizon_steps - 1))
    _, _, reward, done, info = step(model, config, task, reference, at_horizon, action)
    assert bool(done) and bool(info["truncated"]) and not bool(info["crashed"])
    assert float(reward) > 0.0


def test_return_is_differentiable_in_the_actions(setup):
    model, config, task, reference = setup
    state, _ = reset(model, config, task, reference, jax.random.PRNGKey(3))
    actions = jnp.broadcast_to(
        control_to_action(config, reference.control), (10, action_size(model))
    )

    def episode_return(actions):
        _, (_, rewards, _) = rollout_actions(model, config, task, reference, state, actions)
        return jnp.sum(rewards)

    gradient = jax.jit(jax.grad(episode_return))(actions)
    assert gradient.shape == actions.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


@pytest.fixture(scope="module")
def hover_setup():
    from cascade.env import hover_reference, hover_task
    from cascade.reference import tailsitter_reference

    model = tailsitter_reference()
    task = hover_task([0.0, 0.0, -1.5], azimuth_rad=0.0)
    reference = hover_reference(model, task)
    config = EpisodeConfig(
        control_frequency_hz=100.0,
        horizon_steps=50,
        reset_position_std_m=0.2,
        reset_velocity_std_m_s=0.2,
        reset_attitude_std_rad=0.05,
        reset_rate_std_rad_s=0.1,
        upright_limit_rad=3.2,
    )
    return model, config, task, reference


def test_hover_reference_is_nose_up_and_weight_balanced(hover_setup):
    from cascade.dynamics import evaluate_dynamics
    from cascade.math import quaternion_rotate

    model, config, task, reference = hover_setup

    up = quaternion_rotate(reference.state.rigid_body.attitude, jnp.array([1.0, 0.0, 0.0]))
    assert float(up[2]) < -0.99
    result = evaluate_dynamics(model, reference.state, reference.control, reference.environment)
    acceleration = result.derivative.rigid_body.velocity
    assert float(jnp.linalg.norm(acceleration)) < 0.3 * 9.81


def test_hover_task_episode_is_finite_and_rewards_staying_put(hover_setup):
    from cascade.env import action_size

    model, config, task, reference = hover_setup
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    states, observations = jax.jit(jax.vmap(lambda k: reset(model, config, task, reference, k)))(
        keys
    )
    assert observations.shape == (8, 17 + model.n_surfaces + model.n_propellers)
    assert jnp.all(jnp.isfinite(observations))
    hold = jnp.broadcast_to(
        control_to_action(config, reference.control), (20, 8, action_size(model))
    )
    run = jax.jit(
        jax.vmap(lambda s, a: rollout_actions(model, config, task, reference, s, a), in_axes=(0, 1))
    )
    final, (observations, rewards, dones) = run(states, hold)
    assert jnp.all(jnp.isfinite(observations))
    # Open-loop hover drifts but does not fall over inside a fifth of a second.
    assert float(jnp.mean(rewards)) > 0.5
    assert not bool(dones.any())


def test_hover_return_is_differentiable_in_the_actions(hover_setup):
    from cascade.env import action_size

    model, config, task, reference = hover_setup
    state, _ = reset(model, config, task, reference, jax.random.PRNGKey(5))
    actions = jnp.broadcast_to(
        control_to_action(config, reference.control), (10, action_size(model))
    )

    def episode_return(actions):
        _, (_, rewards, _) = rollout_actions(model, config, task, reference, state, actions)
        return jnp.sum(rewards)

    gradient = jax.jit(jax.grad(episode_return))(actions)
    assert jnp.all(jnp.isfinite(gradient)) and float(jnp.max(jnp.abs(gradient))) > 0.0


def test_rollout_policy_with_a_constant_policy_matches_rollout_actions(setup):
    from cascade.env import action_size, rollout_policy

    model, config, task, reference = setup
    state, _ = reset(model, config, task, reference, jax.random.PRNGKey(7))
    action = control_to_action(config, reference.control)
    constant = lambda policy_state, obs, env_state: (action, policy_state)  # noqa: E731
    final_a, (_, rewards_a, _) = rollout_actions(
        model,
        config,
        task,
        reference,
        state,
        jnp.broadcast_to(action, (config.horizon_steps, action_size(model))),
    )
    final_b, (_, actions_b, rewards_b, _) = rollout_policy(
        model, config, task, reference, state, constant, None
    )
    assert jnp.allclose(rewards_a, rewards_b)
    assert jnp.allclose(final_a.aircraft.rigid_body.position, final_b.aircraft.rigid_body.position)
    assert actions_b.shape == (config.horizon_steps, action_size(model))


def test_cascade_baseline_tracks_the_reference_from_perturbed_starts(setup):
    from cascade.control import aerobatic_reference_controller
    from cascade.env import cascade_policy, rollout_policy

    model, config, task, reference = setup
    policy, policy_state = cascade_policy(
        aerobatic_reference_controller(), model, config, task, reference
    )
    keys = jax.random.split(jax.random.PRNGKey(11), 8)
    states, _ = jax.vmap(lambda k: reset(model, config, task, reference, k))(keys)
    run = jax.jit(
        jax.vmap(lambda s: rollout_policy(model, config, task, reference, s, policy, policy_state))
    )
    finals, (observations, actions, rewards, dones) = run(states)
    assert jnp.all(jnp.isfinite(observations)) and jnp.all(jnp.abs(actions) <= 1.0 + 1e-6)
    # No crashes: only the horizon ends the episodes.
    assert not bool(dones[:, :-1].any())
    # The baseline earns a reference score a learner can be judged against.
    assert float(jnp.mean(rewards)) > 0.6
    assert float(jnp.mean(rewards[:, -10:])) > 0.8


def test_episodes_vmap_over_randomised_models(setup):
    from cascade.model import broadcast_model

    model, config, task, reference = setup
    scales = jnp.array([0.8, 1.0, 1.2])
    models = broadcast_model(model, (3,))
    models = models._replace(
        mass=models.mass * scales, inertia=models.inertia * scales[:, None, None]
    )
    keys = jax.random.split(jax.random.PRNGKey(13), 3)
    batched_reset = jax.jit(jax.vmap(lambda m, k: reset(m, config, task, reference, k)))
    states, observations = batched_reset(models, keys)
    assert observations.shape[0] == 3 and jnp.all(jnp.isfinite(observations))
    action = control_to_action(config, reference.control)
    batched_step = jax.jit(
        jax.vmap(lambda m, s: step(m, config, task, reference, s, action), in_axes=(0, 0))
    )
    next_states, observations, rewards, dones, _ = batched_step(models, states)
    assert jnp.all(jnp.isfinite(observations)) and rewards.shape == (3,)
    # The trim control lifts the light aircraft and lets the heavy one sink: the models differ.
    climb = -next_states.aircraft.rigid_body.velocity[:, 2]
    assert float(climb[0]) > float(climb[2])


def test_transition_task_baseline_reaches_cruise_from_hover():
    from cascade.control import GuidanceSetpoint, tailsitter_reference_controller
    from cascade.control.vtol import velocity_ramp_schedule
    from cascade.env import hover_reference, rollout_policy, transition_policy, transition_task
    from cascade.reference import tailsitter_reference, tailsitter_reference_spec

    spec = tailsitter_reference_spec()
    model = tailsitter_reference()
    task = transition_task(8.0, 1.5, 0.0)
    reference = hover_reference(model, task)
    config = EpisodeConfig(
        control_frequency_hz=100.0,
        horizon_steps=800,
        reset_position_std_m=0.1,
        reset_velocity_std_m_s=0.1,
        reset_attitude_std_rad=0.03,
        reset_rate_std_rad_s=0.05,
        upright_limit_rad=3.2,
    )
    dt = 1.0 / config.control_frequency_hz
    hover = velocity_ramp_schedule(
        config.horizon_steps,
        dt,
        start_position_ned=jnp.array([0.0, 0.0, -1.5]),
        heading_rad=jnp.array(0.0),
        cruise_speed_m_s=jnp.array(8.0),
        acceleration_m_s2=jnp.array(3.5),
        hold_steps=200,
    )
    forward = GuidanceSetpoint(
        airspeed_m_s=jnp.full(config.horizon_steps, 8.0),
        altitude_m=jnp.full(config.horizon_steps, 1.5),
        heading_rad=jnp.zeros(config.horizon_steps),
    )
    policy, policy_state = transition_policy(
        tailsitter_reference_controller(spec), model, config, task, reference, hover, forward
    )
    state, first = reset(model, config, task, reference, jax.random.PRNGKey(3))
    assert jnp.all(jnp.isfinite(first))
    final, (observations, actions, rewards, dones) = jax.jit(
        lambda s: rollout_policy(model, config, task, reference, s, policy, policy_state)
    )(state)
    assert jnp.all(jnp.isfinite(observations))
    assert not bool(dones[:-1].any())
    speed = float(jnp.linalg.norm(final.aircraft.rigid_body.velocity))
    assert abs(speed - 8.0) < 1.5
    # Hover earns little on a cruise task; cruise earns most of the reward.
    assert float(jnp.mean(rewards[:100])) < 0.6
    assert float(jnp.mean(rewards[-100:])) > 0.7


def test_a_gradient_step_through_the_dynamics_improves_a_policy(setup):
    import importlib.util
    import pathlib

    from cascade.env import action_size, rollout_policy

    path = pathlib.Path(__file__).parent.parent / "examples" / "learn_tracking_policy.py"
    spec = importlib.util.spec_from_file_location("learn_tracking_policy", path)
    learn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(learn)

    model, config, task, reference = setup
    trim_action = control_to_action(config, reference.control)
    params = learn.initial_parameters(
        jax.random.PRNGKey(0), 17 + model.n_surfaces + model.n_propellers, action_size(model)
    )
    # Break the symmetry of the zero output layer so the gradient reaches the first layer too.
    params["w2"] = 0.01 * jax.random.normal(jax.random.PRNGKey(1), params["w2"].shape)
    keys = jax.random.split(jax.random.PRNGKey(2), 4)

    def mean_return(params):
        def policy(policy_state, observation, env_state):
            return learn.policy_network(params, observation, trim_action), policy_state

        def one(key):
            state, _ = reset(model, config, task, reference, key)
            _, (_, _, rewards, _) = rollout_policy(
                model, config, task, reference, state, policy, None
            )
            return jnp.sum(rewards)

        return jnp.mean(jax.vmap(one)(keys))

    value_and_grad = jax.jit(jax.value_and_grad(mean_return))
    before, gradient = value_and_grad(params)
    assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(gradient))
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(gradient)))
    assert float(norm) > 0.0
    stepped = jax.tree.map(lambda p, g: p + 1e-3 * g / norm, params, gradient)
    after, _ = value_and_grad(stepped)
    assert float(after) > float(before)


def test_sensor_noise_and_delay(setup):
    from cascade.env import observation, sensor_noise

    model, config, task, reference = setup
    key = jax.random.PRNGKey(21)
    clean_state, clean = reset(model, config, task, reference, key)
    assert jnp.allclose(clean, observation(model, task, reference, clean_state))

    noise = sensor_noise(rate_std=0.01, rate_bias_std=0.02, air_std=0.02, gravity_std=0.01)
    noisy_state, noisy = reset(model, config, task, reference, key, noise)
    # Same physical start, different reading; the bias sits on the rate block only.
    assert jnp.allclose(
        noisy_state.aircraft.rigid_body.position, clean_state.aircraft.rigid_body.position
    )
    assert not jnp.allclose(noisy, clean)
    assert jnp.all(noisy_state.sensor_bias[:6] == 0.0)
    assert jnp.any(noisy_state.sensor_bias[6:9] != 0.0)
    action = control_to_action(config, reference.control)
    stepped, reading, _, _, _ = step(model, config, task, reference, noisy_state, action, noise)
    truth = observation(model, task, reference, stepped)
    assert jnp.all(jnp.isfinite(reading)) and not jnp.allclose(reading, truth)
    assert float(jnp.max(jnp.abs(reading - truth))) < 0.2

    delayed_config = EpisodeConfig(horizon_steps=60, observation_delay_steps=2)
    state, first = reset(model, delayed_config, task, reference, key)
    state, one, _, _, _ = step(model, delayed_config, task, reference, state, action)
    state, two, _, _, _ = step(model, delayed_config, task, reference, state, action)
    state, three, _, _, _ = step(model, delayed_config, task, reference, state, action)
    # Two periods of latency: the first two readings repeat the reset observation.
    assert jnp.allclose(one, first) and jnp.allclose(two, first)
    assert not jnp.allclose(three, first)
    assert jnp.allclose(three, state.observation_buffer[0])


def test_weather_shifts_the_reset_velocity_and_gusts_the_episode(setup):
    from cascade.control import aerobatic_reference_controller
    from cascade.env import cascade_policy, rollout_policy
    from cascade.env.weather import weather_condition

    model, config, task, reference = setup
    quiet = EpisodeConfig(
        horizon_steps=160,
        reset_position_std_m=0.0,
        reset_velocity_std_m_s=0.0,
        reset_attitude_std_rad=0.0,
        reset_rate_std_rad_s=0.0,
    )
    weather = weather_condition(6.0, jnp.deg2rad(90.0), turbulence_wind_20ft_m_s=12.0)
    state, first = reset(model, quiet, task, reference, jax.random.PRNGKey(4), weather=weather)
    # Airspeed at reset equals the trim airspeed although the ground velocity carries the wind.
    airspeed = float(jnp.linalg.norm(state.aircraft.rigid_body.velocity - state.wind_ned))
    assert abs(airspeed - 12.0) < 0.05
    assert float(state.wind_ned[1]) < -4.0  # an east wind blows toward the west
    policy, policy_state = cascade_policy(
        aerobatic_reference_controller(), model, quiet, task, reference
    )
    final, (observations, actions, rewards, dones) = jax.jit(
        lambda s: rollout_policy(
            model, quiet, task, reference, s, policy, policy_state, None, weather
        )
    )(state)
    assert jnp.all(jnp.isfinite(observations))
    assert not bool(dones[:-1].any())
    # Turbulence moved the wind, and the cascade still held the reference.
    assert float(jnp.std(final.wind_ned)) > 0.0 or float(jnp.linalg.norm(final.gust)) > 0.0
    assert abs(float(-final.aircraft.rigid_body.position[2]) - 50.0) < 5.0
    assert float(jnp.mean(rewards[-40:])) > 0.5


def test_observation_size_and_layout_match_the_vector(setup):
    from cascade.env import observation_layout, observation_size

    model, config, task, reference = setup
    state, obs = reset(model, config, task, reference, jax.random.PRNGKey(0))
    assert obs.shape == (observation_size(model),)
    layout = observation_layout(model)
    assert layout.propellers.stop == observation_size(model)
    assert obs[layout.surfaces].shape == (model.n_surfaces,)
    # Gravity direction is a unit vector in body axes.
    assert abs(float(jnp.linalg.norm(obs[layout.gravity])) - 1.0) < 1e-4
