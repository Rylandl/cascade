import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cascade.dynamics import evaluate_dynamics
from cascade.env import (
    EpisodeConfig,
    Randomisation,
    control_to_action,
    randomisation,
    reset,
    sample_models,
    step,
    tracking_task,
    trimmed_reference,
)
from cascade.initialization import standard_environment, zero_control, zero_state
from cascade.model import broadcast_model
from cascade.reference import aerobatic_reference, skywalker_x8


def test_sampled_models_scale_named_leaves_and_shift_the_centre_of_mass():
    model = aerobatic_reference()
    spec = randomisation(
        mass=(0.8, 1.2),
        inertia=(0.7, 1.4),
        lift_curve_slope=(0.9, 1.1),
        surface_time_constant=(0.5, 2.0),
        thrust=(0.85, 1.15),
        center_of_mass_shift_m=(-0.02, 0.02),
    )
    models = sample_models(model, spec, jax.random.PRNGKey(0), 64)
    mass = np.asarray(models.mass) / float(model.mass)
    assert mass.shape == (64,) and mass.min() >= 0.8 and mass.max() <= 1.2 and mass.std() > 0.05
    slope = np.asarray(models.surfaces.lift_curve_slope) / np.asarray(
        model.surfaces.lift_curve_slope
    )
    assert np.allclose(slope, slope[:, :1])  # one factor per world, shared over surfaces
    assert slope.min() >= 0.9 and slope.max() <= 1.1
    # The inertia inverse follows the scaled inertia.
    product = np.einsum(
        "nij,njk->nik", np.asarray(models.inertia), np.asarray(models.inertia_inverse)
    )
    assert np.allclose(product, np.eye(3), atol=1e-4)
    # A forward centre-of-mass shift moves every part aft in body coordinates.
    shift = (
        np.asarray(model.surfaces.position)[None, :, 0]
        - np.asarray(models.surfaces.position)[:, :, 0]
    )
    assert np.allclose(shift, shift[:, :1]) and np.abs(shift).max() <= 0.02 + 1e-6
    propeller_shift = (
        np.asarray(model.propellers.position)[None, :, 0]
        - np.asarray(models.propellers.position)[:, :, 0]
    )
    assert np.allclose(propeller_shift[:, 0], shift[:, 0])
    assert np.allclose(np.asarray(models.body.reference_position)[:, 0], -shift[:, 0])
    # Untouched leaves stay nominal.
    assert np.allclose(np.asarray(models.surfaces.chord), np.asarray(model.surfaces.chord))


def test_randomised_batch_flies_under_vmap():
    model = aerobatic_reference()
    task = tracking_task(12.0, 50.0, 0.0)
    reference = trimmed_reference(model, task)
    config = EpisodeConfig(horizon_steps=20)
    models = sample_models(
        model, randomisation(mass=(0.9, 1.1), thrust=(0.9, 1.1)), jax.random.PRNGKey(1), 8
    )
    keys = jax.random.split(jax.random.PRNGKey(2), 8)
    states, obs = jax.jit(jax.vmap(lambda m, k: reset(m, config, task, reference, k)))(models, keys)
    action = control_to_action(config, reference.control)
    _, obs, rewards, dones, _ = jax.jit(
        jax.vmap(lambda m, s: step(m, config, task, reference, s, action))
    )(models, states)
    assert jnp.all(jnp.isfinite(obs)) and rewards.shape == (8,)


@pytest.mark.parametrize("factory", [aerobatic_reference, skywalker_x8])
@pytest.mark.parametrize("rates", [(0.0, 0.0, 0.0), (0.3, 0.5, -0.4)])
def test_cg_shift_preserves_loads_at_same_physical_motion_and_translates_wrench(factory, rates):
    model = factory()
    state = zero_state(model, forward_speed=14.0)
    state = state._replace(
        rigid_body=state.rigid_body._replace(
            velocity=jnp.array([14.0, 0.6, 1.2]), angular_velocity=jnp.array(rates)
        ),
        actuators=state.actuators._replace(
            surface_deflection=jnp.linspace(-0.05, 0.08, model.n_surfaces),
            propeller_speed=jnp.full((model.n_propellers,), 300.0),
        ),
    )
    environment, control = standard_environment(), zero_control(model)
    nominal = evaluate_dynamics(model, state, control, environment)
    models = sample_models(
        model, randomisation(center_of_mass_shift_m=(-0.08, 0.08)), jax.random.PRNGKey(6), 5
    )
    offsets = models.body.reference_position - model.body.reference_position

    # A different point on the same rigid body has v_new = v_old + omega x (CG_new-CG_old).
    # At that same physical motion every aerodynamic/propulsive load must be unchanged.
    def shifted_loads(shifted_model, offset):
        shifted_state = state._replace(
            rigid_body=state.rigid_body._replace(
                velocity=state.rigid_body.velocity
                - jnp.cross(state.rigid_body.angular_velocity, offset)
            )
        )
        result = evaluate_dynamics(shifted_model, shifted_state, control, environment)
        return result.force_body, result.moment_body

    force, moment = jax.jit(jax.vmap(shifted_loads))(models, offsets)
    assert jnp.allclose(force, nominal.force_body, rtol=2e-6, atol=2e-5)
    expected_moment = nominal.moment_body + jnp.cross(offsets, nominal.force_body)
    assert jnp.allclose(moment, expected_moment, rtol=2e-6, atol=2e-5)
    assert jnp.max(jnp.abs(moment - nominal.moment_body)) > 0.1


@pytest.mark.parametrize(
    "bounds",
    [(1.2, 0.8), (float("nan"), 1.0), (0.8, float("inf")), (0.8,), (0.8, 1.0, 1.2), "12"],
)
def test_randomisation_rejects_malformed_ranges(bounds):
    with pytest.raises(ValueError, match="mass"):
        randomisation(mass=bounds)
    # Constructing the named tuple directly cannot bypass validation at sampling time.
    with pytest.raises(ValueError, match="mass"):
        sample_models(aerobatic_reference(), Randomisation({"mass": bounds}), jax.random.key(0), 2)


@pytest.mark.parametrize("bounds", [(-0.1, 1.0), (0.0, 1.0)])
def test_randomisation_rejects_nonphysical_positive_parameter_ranges(bounds):
    with pytest.raises(ValueError, match="mass"):
        randomisation(mass=bounds)
    with pytest.raises(ValueError, match="time_constant"):
        randomisation(surface_time_constant=bounds)


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_randomisation_rejects_invalid_world_counts(count):
    with pytest.raises(ValueError, match="positive integer"):
        sample_models(aerobatic_reference(), randomisation(), jax.random.key(0), count)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("mass_typo", "unknown"),
        ("surfaces.typo", "unknown"),
        ("surfaces", "array leaf"),
        ("n_surfaces", "unknown"),
        ("mass.shape", "unknown"),
        ("", "non-empty"),
        ("body..lift", "non-empty"),
        (1, "non-empty"),
        ("inertia_inverse", "derived"),
        ("surfaces.body_from_surface", "cannot be scaled"),
        ("propellers.direction", "cannot be scaled"),
    ],
)
def test_randomisation_rejects_invalid_scale_paths(name, message):
    with pytest.raises(ValueError, match=message):
        sample_models(
            aerobatic_reference(), Randomisation({name: (0.9, 1.1)}), jax.random.key(0), 2
        )


def test_randomisation_rejects_invalid_cg_bounds_and_batched_input():
    with pytest.raises(ValueError, match="center_of_mass_shift_m"):
        randomisation(center_of_mass_shift_m=(-float("inf"), 0.1))
    with pytest.raises(ValueError, match="unbatched"):
        sample_models(
            broadcast_model(aerobatic_reference(), (3,)), randomisation(), jax.random.key(0), 2
        )


def test_randomisation_still_jits_with_model_and_key_arguments():
    model = skywalker_x8()
    spec = Randomisation({"body.lift.alpha": (0.9, 1.1), "inertia": (0.8, 1.2)}, (-0.1, 0.1))
    sample = jax.jit(lambda m, key: sample_models(m, spec, key, 3))
    models = sample(model, jax.random.key(4))
    assert models.body.reference_position.shape == (3, 3)
    assert jnp.all(jnp.isfinite(models.inertia_inverse))
