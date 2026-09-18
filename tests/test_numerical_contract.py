"""Release checks of numerical behavior, independent of a specific flight controller."""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cascade


@contextmanager
def precision(x64=True):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", x64)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize(
    "loader,speed", [(cascade.aerobatic_reference, 12.0), (cascade.skywalker_x8, 18.0)]
)
@pytest.mark.parametrize("x64", [False, True])
def test_precision_jit_native_batch_and_vmap_agree(loader, speed, x64):
    with precision(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        model = loader()
        environment = cascade.standard_environment()
        control = cascade.ControlInput(
            propeller=jnp.full((model.n_propellers,), 0.45, dtype=dtype),
            channel=jnp.zeros(model.n_control_channels, dtype=dtype),
        )
        state = cascade.equilibrate_internal_state(
            model,
            cascade.zero_state(model, forward_speed=speed, altitude=50.0),
            control,
            environment,
        )
        states = jax.tree.map(lambda x: jnp.stack([x, x, x]), state)
        states = states._replace(
            rigid_body=states.rigid_body._replace(
                velocity=states.rigid_body.velocity.at[:, 2].set(jnp.array([-0.4, 0.0, 0.7]))
            )
        )

        def scalar(s):
            return cascade.rk4_step(model, s, control, environment, 0.005)

        expected = jax.tree.map(
            lambda *xs: jnp.stack(xs),
            *(scalar(jax.tree.map(lambda x, i=i: x[i], states)) for i in range(3)),
        )
        vectorized = jax.jit(jax.vmap(scalar))(states)
        native = jax.jit(cascade.rk4_step)(model, states, control, environment, 0.005)
        for actual in (native, vectorized):
            for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                assert got.dtype == dtype
                np.testing.assert_allclose(
                    got, want, rtol=2e-6 if not x64 else 1e-12, atol=2e-6 if not x64 else 1e-12
                )


@pytest.mark.parametrize(
    "loader,speed", [(cascade.aerobatic_reference, 12.0), (cascade.skywalker_x8, 18.0)]
)
@pytest.mark.parametrize("x64", [False, True])
def test_rollout_gradient_matches_central_difference(loader, speed, x64):
    with precision(x64):
        model = loader()
        environment = cascade.standard_environment()
        neutral = cascade.ControlInput(
            jnp.full(model.n_propellers, 0.45), jnp.zeros(model.n_control_channels)
        )
        state = cascade.equilibrate_internal_state(
            model,
            cascade.zero_state(model, forward_speed=speed, altitude=50.0),
            neutral,
            environment,
        )

        def pitch_rate(elevator):
            control = neutral._replace(channel=neutral.channel.at[1].set(elevator))
            final, _ = cascade.rollout(
                model, state, cascade.repeat_control(control, 20), environment, 0.005
            )
            return final.rigid_body.angular_velocity[1]

        value = jax.jit(pitch_rate)
        at = jnp.array(0.025)
        epsilon = 2e-5 if x64 else 2e-3
        numerical = (value(at + epsilon) - value(at - epsilon)) / (2 * epsilon)
        automatic = jax.jit(jax.grad(pitch_rate))(at)
        assert abs(float(automatic)) > 1e-4
        np.testing.assert_allclose(
            automatic, numerical, rtol=2e-4 if x64 else 5e-3, atol=1e-7 if x64 else 1e-4
        )


@pytest.mark.parametrize(
    "loader,speed", [(cascade.aerobatic_reference, 12.0), (cascade.skywalker_x8, 18.0)]
)
def test_rk4_timestep_refinement_converges(loader, speed):
    # Float64 separates integration error from rounding. Use the same physical interval and
    # nonzero aerodynamic/actuator dynamics, with the finest solution as an independent reference.
    with precision():
        model = loader()
        environment = cascade.standard_environment()
        control = cascade.ControlInput(
            jnp.full(model.n_propellers, 0.45), jnp.zeros(model.n_control_channels).at[1].set(0.025)
        )
        state = cascade.equilibrate_internal_state(
            model,
            cascade.zero_state(model, forward_speed=speed, altitude=50.0),
            control,
            environment,
        )

        def run(steps):
            final, _ = jax.jit(cascade.rollout)(
                model, state, cascade.repeat_control(control, steps), environment, 0.2 / steps
            )
            return np.concatenate([np.asarray(x).ravel() for x in final.rigid_body])

        reference = run(320)
        errors = [np.linalg.norm(run(steps) - reference) for steps in (20, 40, 80)]
        assert errors[0] > 1e-12, errors
        assert errors[1] < errors[0] / 8, errors
        assert errors[2] < errors[1] / 8, errors
