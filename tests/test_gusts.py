import jax
import jax.numpy as jnp

from cascade.gusts import (
    DrydenParameters,
    dryden_environment_sequence,
    dryden_low_altitude,
    dryden_wind_sequence,
)
from cascade.initialization import standard_environment, zero_control, zero_state
from cascade.integration import repeat_control, rollout
from cascade.reference import aerobatic_reference


def calm_parameters(sigma_u, sigma_v, sigma_w):
    values = (sigma_u, sigma_v, sigma_w, 200.0, 200.0, 50.0)
    return DrydenParameters(*(jnp.asarray(value) for value in values))


def test_low_altitude_parameters_follow_the_specification():
    parameters = dryden_low_altitude(jnp.asarray(50.0), jnp.asarray(15.4))
    # sigma_w = 0.1 W20; at 164 ft the ratio (0.177 + 0.000823 h)^0.4 is about 0.6, so
    # the horizontal intensities exceed the vertical one.
    assert jnp.allclose(parameters.sigma_w, 1.54)
    assert parameters.sigma_u > parameters.sigma_w
    assert jnp.allclose(parameters.length_w, 50.0, atol=0.01)
    assert parameters.length_u > parameters.length_w


def test_zero_intensity_reproduces_the_mean_wind_exactly():
    parameters = calm_parameters(0.0, 0.0, 0.0)
    wind = dryden_wind_sequence(
        jax.random.key(0),
        40,
        0.01,
        airspeed_m_s=15.0,
        parameters=parameters,
        mean_wind_ned=jnp.array([3.0, -1.0, 0.5]),
    )
    assert wind.shape == (40, 3)
    assert jnp.allclose(wind, jnp.array([3.0, -1.0, 0.5]))


def test_gust_variances_match_the_intensities_and_realizations_are_deterministic():
    parameters = dryden_low_altitude(jnp.asarray(100.0), jnp.asarray(15.4))
    key = jax.random.key(7)
    wind = dryden_wind_sequence(
        key, 40_000, 0.02, airspeed_m_s=18.0, parameters=parameters, batch_shape=(8,)
    )
    again = dryden_wind_sequence(
        key, 40_000, 0.02, airspeed_m_s=18.0, parameters=parameters, batch_shape=(8,)
    )
    assert wind.shape == (40_000, 8, 3)
    assert jnp.array_equal(wind, again)
    # Discard the filter transient, then compare per-component standard deviations.
    settled = wind[2_000:]
    observed = jnp.std(settled.reshape(-1, 3), axis=0)
    expected = jnp.stack((parameters.sigma_u, parameters.sigma_v, parameters.sigma_w))
    assert jnp.allclose(observed, expected, rtol=0.12)
    assert jnp.abs(jnp.mean(settled)) < 0.1 * float(parameters.sigma_w)


def test_heading_rotates_the_longitudinal_gust_into_ned():
    parameters = calm_parameters(1.0, 0.0, 0.0)
    north = dryden_wind_sequence(
        jax.random.key(1), 500, 0.02, airspeed_m_s=15.0, parameters=parameters
    )
    east = dryden_wind_sequence(
        jax.random.key(1),
        500,
        0.02,
        airspeed_m_s=15.0,
        parameters=parameters,
        heading_rad=jnp.pi / 2.0,
    )
    assert jnp.allclose(north[:, 1], 0.0, atol=1e-6)
    assert jnp.allclose(east[:, 0], 0.0, atol=1e-5)
    assert jnp.allclose(east[:, 1], north[:, 0], atol=1e-5)


def test_gusty_rollout_runs_batched_under_jit_and_stays_finite():
    model = aerobatic_reference()
    environment = standard_environment(batch_shape=(4,))
    environment = environment._replace(wind=jnp.broadcast_to(jnp.array([2.0, 0.0, 0.0]), (4, 3)))
    parameters = dryden_low_altitude(jnp.asarray(20.0), jnp.asarray(15.4))
    gusts = dryden_environment_sequence(
        jax.random.key(3), environment, 50, 0.01, airspeed_m_s=12.0, parameters=parameters
    )
    assert gusts.wind.shape == (50, 4, 3)
    assert not jnp.allclose(gusts.wind[:, 0], gusts.wind[:, 1])
    state = zero_state(model, batch_shape=(4,), altitude=20.0, forward_speed=12.0)
    controls = repeat_control(zero_control(model, batch_shape=(4,)), 50)

    final, trajectory = jax.jit(rollout, static_argnames=("step",))(
        model, state, controls, environment, 0.01, environments=gusts
    )
    calm, _ = jax.jit(rollout, static_argnames=("step",))(model, state, controls, environment, 0.01)

    assert jnp.all(jnp.isfinite(trajectory.rigid_body.velocity))
    assert not jnp.allclose(final.rigid_body.velocity, calm.rigid_body.velocity)
