import jax
import jax.numpy as jnp

from cascade.initialization import standard_environment, zero_control, zero_state
from cascade.integration import repeat_control, rollout
from cascade.reference import aerobatic_reference

STEPS = 20
DT = 0.01


def test_repeated_environment_sequence_matches_constant_path():
    model = aerobatic_reference()
    state = zero_state(model, altitude=20.0, forward_speed=12.0)
    controls = repeat_control(zero_control(model), steps=STEPS)
    environment = standard_environment()
    environments = repeat_control(environment, steps=STEPS)

    jit_rollout = jax.jit(rollout, static_argnames=("step",))
    final_constant, trajectory_constant = jit_rollout(model, state, controls, environment, DT)
    final_sequence, trajectory_sequence = jit_rollout(
        model, state, controls, environment, DT, environments=environments
    )

    for constant_leaf, sequence_leaf in zip(
        jax.tree.leaves(trajectory_constant), jax.tree.leaves(trajectory_sequence), strict=True
    ):
        assert jnp.array_equal(constant_leaf, sequence_leaf)
    for constant_leaf, sequence_leaf in zip(
        jax.tree.leaves(final_constant), jax.tree.leaves(final_sequence), strict=True
    ):
        assert jnp.array_equal(constant_leaf, sequence_leaf)


def test_wind_step_only_changes_the_second_half_trajectory():
    model = aerobatic_reference()
    state = zero_state(model, altitude=20.0, forward_speed=12.0)
    controls = repeat_control(zero_control(model), steps=STEPS)
    still_air = standard_environment()
    windy = still_air._replace(wind=jnp.array([3.0, 0.0, 0.0]))

    half = STEPS // 2
    first_half = repeat_control(still_air, steps=half)
    second_half = repeat_control(windy, steps=half)
    environments = jax.tree.map(
        lambda calm, wind: jnp.concatenate((calm, wind), axis=0), first_half, second_half
    )

    _, trajectory_wind = rollout(model, state, controls, still_air, DT, environments=environments)
    _, trajectory_no_wind = rollout(model, state, controls, still_air, DT)

    wind_leaves = jax.tree.leaves(trajectory_wind)
    no_wind_leaves = jax.tree.leaves(trajectory_no_wind)

    for wind_leaf, no_wind_leaf in zip(wind_leaves, no_wind_leaves, strict=True):
        assert jnp.array_equal(wind_leaf[:half], no_wind_leaf[:half])

    assert any(
        not jnp.array_equal(wind_leaf[half:], no_wind_leaf[half:])
        for wind_leaf, no_wind_leaf in zip(wind_leaves, no_wind_leaves, strict=True)
    )


def test_gradient_through_environment_sequence_is_finite():
    model = aerobatic_reference()
    state = zero_state(model, altitude=20.0, forward_speed=12.0)
    controls = repeat_control(zero_control(model), steps=STEPS)
    base_environment = standard_environment()

    def final_north_position(wind_speed):
        wind = jnp.stack((wind_speed, jnp.array(0.0), jnp.array(0.0)))
        environments = repeat_control(base_environment._replace(wind=wind), steps=STEPS)
        final, _ = rollout(model, state, controls, base_environment, DT, environments=environments)
        return final.rigid_body.position[0]

    gradient = jax.jit(jax.grad(final_north_position))(jnp.array(2.0))
    assert jnp.isfinite(gradient)
