import jax
import jax.numpy as jnp
import numpy as np

from cascade.env import (
    EpisodeConfig,
    control_to_action,
    randomisation,
    reset,
    sample_models,
    step,
    tracking_task,
    trimmed_reference,
)
from cascade.reference import aerobatic_reference


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
