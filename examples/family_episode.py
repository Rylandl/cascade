"""A family of airframes, each in its own weather, flown by its own auto-tuned baseline.

This is the skeleton of the "no a priori information" demonstration: designs are drawn at
random, trimmed, tuned, and stacked; each episode draws a weather condition; one vmap flies
them all. A learner would replace the baseline policy and see only the observations. The
design parameters printed here are the hidden truth it never gets.
"""

import jax
import jax.numpy as jnp

from cascade.archetypes import ConventionalDesign, FlyingWingDesign
from cascade.env import EpisodeConfig, cascade_policy, reset, rollout_policy
from cascade.family import sample_family
from cascade.weather import sample_weather_uniform

COUNT = 6
CONFIG = EpisodeConfig(control_frequency_hz=50.0, horizon_steps=300)


def fly(family, key):
    keys = jax.random.split(key, len(family))
    weather_keys = jax.random.split(jax.random.fold_in(key, 1), len(family))

    def episode(model, task, reference, controller, episode_key, weather_key):
        weather = sample_weather_uniform(weather_key, speed_range_m_s=(0.0, 8.0))
        state, _ = reset(model, CONFIG, task, reference, episode_key, weather=weather)
        policy, policy_state = cascade_policy(controller, model, CONFIG, task, reference)
        final, (_, _, rewards, dones) = rollout_policy(
            model, CONFIG, task, reference, state, policy, policy_state, None, weather
        )
        return weather.wind_speed_m_s, weather.turbulence_wind_20ft_m_s, rewards, dones

    return jax.jit(jax.vmap(episode))(
        family.models, family.tasks, family.references, family.controllers, keys, weather_keys
    )


def main() -> None:
    for archetype in (FlyingWingDesign, ConventionalDesign):
        family = sample_family(archetype, jax.random.PRNGKey(0), COUNT)
        wind, turbulence, rewards, dones = fly(family, jax.random.PRNGKey(1))
        print(f"\n{archetype.__name__}: {COUNT} designs, each in its own weather")
        print("  span   mass  cruise | wind  turb | pitch kp | return / max   crashed")
        for i, design in enumerate(family.designs):
            crashed = bool(dones[i, :-1].any())
            print(
                f"  {design.span_m:4.2f} {float(family.models.mass[i]):6.2f} "
                f"{float(family.cruise_speeds_m_s[i]):6.1f} | {float(wind[i]):4.1f} "
                f"{float(turbulence[i]):5.1f} | {float(family.controllers.rate.kp[i, 1]):8.3f} | "
                f"{float(jnp.sum(rewards[i])):6.1f} / {CONFIG.horizon_steps}   {crashed}"
            )
        print(f"  mean return {float(jnp.mean(jnp.sum(rewards, axis=1))):.1f}")


if __name__ == "__main__":
    main()
