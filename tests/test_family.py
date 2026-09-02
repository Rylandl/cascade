import jax
import jax.numpy as jnp

from cascade.archetypes import ConventionalDesign, FlyingWingDesign, design_spec
from cascade.env import EpisodeConfig, cascade_policy, reset, rollout_policy
from cascade.family import family_member, sample_family
from cascade.weather import sample_weather_uniform


def test_flying_wings_share_a_topology_across_layouts():
    pusher = design_spec(FlyingWingDesign(motors="pusher", winglet_area_fraction=0.0))
    twin = design_spec(FlyingWingDesign(motors="twin_tractor", winglet_area_fraction=0.08))
    assert len(pusher.surfaces) == len(twin.surfaces)
    assert len(pusher.propellers) == len(twin.propellers) == 2
    conventional = design_spec(ConventionalDesign(tail="conventional"))
    v_tail = design_spec(ConventionalDesign(tail="v_tail"))
    assert len(conventional.surfaces) == len(v_tail.surfaces)


def test_a_family_flies_as_one_vmap_under_its_own_baselines():
    family = sample_family(FlyingWingDesign, jax.random.PRNGKey(0), 4)
    assert len(family) == 4
    assert family.models.mass.shape == (4,)
    # The tuned gains differ across the family: these are different aircraft.
    assert float(jnp.std(family.controllers.rate.kp[:, 1])) > 0.0
    assert float(jnp.max(family.cruise_speeds_m_s) - jnp.min(family.cruise_speeds_m_s)) > 1.0

    config = EpisodeConfig(control_frequency_hz=50.0, horizon_steps=150)
    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    weather_keys = jax.random.split(jax.random.PRNGKey(2), 4)

    def episode(model, task, reference, controller, key, weather_key):
        weather = sample_weather_uniform(weather_key, speed_range_m_s=(0.0, 4.0))
        state, _ = reset(model, config, task, reference, key, weather=weather)
        policy, policy_state = cascade_policy(controller, model, config, task, reference)
        _, (observations, actions, rewards, dones) = rollout_policy(
            model, config, task, reference, state, policy, policy_state, None, weather
        )
        return observations, rewards, dones

    observations, rewards, dones = jax.jit(jax.vmap(episode))(
        family.models, family.tasks, family.references, family.controllers, keys, weather_keys
    )
    assert observations.shape[:2] == (4, 150)
    assert jnp.all(jnp.isfinite(observations))
    assert not bool(dones[:, :-1].any())
    assert float(jnp.min(jnp.mean(rewards[:, -50:], axis=1))) > 0.4

    # The hidden truth stays with the family, apart from what an episode exposes.
    model, task, reference, controller = family_member(family, 2)
    assert float(model.mass) == float(family.models.mass[2])
    assert family.designs[2].span_m > 0.0
