"""Model-based baseline policies: the control cascade and the transition controller wrapped
as policies so a learner has a reference score on the same task and horizon."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from cascade.control.loops import (
    CascadeController,
    CascadeState,
    GuidanceSetpoint,
    cascade_step,
    initial_cascade_state,
)
from cascade.control.vtol import (
    HoverSetpoint,
    TransitionController,
    TransitionState,
    transition_step,
)
from cascade.env.episode import EnvState, EpisodeConfig, control_to_action, current_environment
from cascade.env.tasks import ReferenceFlight, TrackingTask, TransitionTask
from cascade.model import AircraftModel


def cascade_policy(
    controller: CascadeController,
    model: AircraftModel,
    config: EpisodeConfig,
    task: TrackingTask,
    reference: ReferenceFlight,
):
    """The control cascade as a policy for a tracking task: the reference score for a learner.

    Every loop runs at the environment's control rate (the controller's periods are replaced
    by one), so the baseline sees exactly what a learned policy sees in time. Returns the
    policy function and its initial :class:`CascadeState`, holding the reference control until
    the first update.
    """

    if not isinstance(task, TrackingTask):
        raise TypeError("cascade_policy needs a TrackingTask; the cascade has no hover mode")
    controller = controller._replace(rate_period=1, attitude_period=1, guidance_period=1)
    setpoint = GuidanceSetpoint(
        airspeed_m_s=task.airspeed_m_s, altitude_m=task.altitude_m, heading_rad=task.heading_rad
    )
    period = 1.0 / config.control_frequency_hz

    def policy(cascade_state: CascadeState, obs: Array, env_state: EnvState):
        control, cascade_state = cascade_step(
            controller,
            cascade_state,
            setpoint,
            env_state.aircraft,
            current_environment(reference, env_state),
            period,
        )
        return control_to_action(config, control), cascade_state

    return policy, initial_cascade_state(controller, reference.state, reference.control)


def transition_policy(
    controller: TransitionController,
    model: AircraftModel,
    config: EpisodeConfig,
    task: TransitionTask,
    reference: ReferenceFlight,
    hover_setpoints: HoverSetpoint,
    forward_setpoints,
):
    """The transition controller as a policy: the baseline for a transition task.

    ``hover_setpoints`` and ``forward_setpoints`` are time-major schedules of at least
    ``horizon_steps`` entries (a :func:`cascade.vtol.velocity_ramp_schedule`, say); the policy
    indexes them by the episode step and runs :func:`cascade.vtol.transition_step` at the
    control rate. Returns the policy and its initial :class:`TransitionState`.
    """

    from cascade.vtol import initial_transition_state

    period = 1.0 / config.control_frequency_hz
    last = config.horizon_steps - 1

    def policy(transition_state: TransitionState, obs: Array, env_state: EnvState):
        index = jnp.minimum(env_state.step, last)
        hover_setpoint = jax.tree.map(lambda leaf: leaf[index], hover_setpoints)
        forward_setpoint = jax.tree.map(lambda leaf: leaf[index], forward_setpoints)
        control, transition_state, _ = transition_step(
            model,
            controller,
            transition_state,
            hover_setpoint,
            forward_setpoint,
            env_state.aircraft,
            current_environment(reference, env_state),
            period,
        )
        return control_to_action(config, control), transition_state

    return policy, initial_transition_state(reference.state)
