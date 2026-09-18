"""Show held, delayed, and missing observations during a seeded tracking episode."""

import jax
import numpy as np

import cascade
from cascade.env import (
    EpisodeConfig,
    control_to_action,
    observation_layout,
    reset,
    step,
    tracking_task,
    trimmed_reference,
)
from cascade.env.sensors import (
    SensorBlockConfig,
    SensorPipelineConfig,
    onboard_observation,
    sensor_noise_from_sensors,
)


def main() -> None:
    model = cascade.aerobatic_reference()
    task = tracking_task(12.0, 50.0)
    reference = trimmed_reference(model, task)
    config = EpisodeConfig(
        horizon_steps=16,
        observation=onboard_observation(),
        sensors=SensorPipelineConfig(
            rates=SensorBlockConfig(bias_walk_std=0.001),
            airspeed=SensorBlockConfig(sample_period_steps=2),
            position_error=SensorBlockConfig(
                sample_period_steps=4,
                sample_jitter_steps=1,
                delay_steps=2,
                delay_jitter_steps=1,
                dropout_probability=0.25,
            ),
        ),
    )
    noise = sensor_noise_from_sensors(12.0, gyro_std_rad_s=0.01, position_std_m=0.5)
    state, _ = reset(model, config, task, reference, jax.random.PRNGKey(7), noise)
    action = control_to_action(config, reference.control)

    def run(initial):
        def advance(state, _):
            state, _, _, _, info = step(model, config, task, reference, state, action, noise)
            return state, (state.time_s, info["sensor_age_s"], info["sensor_valid"])

        return jax.lax.scan(advance, initial, None, length=config.horizon_steps)

    _, (times, ages, validity) = jax.jit(run)(state)
    layout = observation_layout(model, config.observation)
    print("time (s)  airspeed age (s)  position age (s)")
    for time, age, valid in zip(
        np.asarray(times), np.asarray(ages), np.asarray(validity), strict=True
    ):
        position_age = (
            f"{age[layout.position_error.start]:.3f}"
            if valid[layout.position_error.start]
            else "missing"
        )
        print(f"{time:8.3f}  {age[layout.airspeed.start]:16.3f}  {position_age:>16s}")


if __name__ == "__main__":
    main()
