"""The public task type must describe every task accepted by episode functions."""

from typing import get_type_hints

import jax.numpy as jnp
import pytest

from cascade.env import (
    Task,
    hover_task,
    observation,
    orbit_task,
    reset,
    rollout_actions,
    rollout_policy,
    scheduled_tracking_task,
    step,
    tracking_task,
    transition_task,
    waypoint_task,
)


@pytest.mark.parametrize(
    "task",
    [
        tracking_task(12.0, 50.0),
        hover_task([0.0, 0.0, -50.0]),
        transition_task(12.0, 50.0),
        scheduled_tracking_task([0.0, 10.0], [12.0, 13.0], [50.0, 52.0]),
        waypoint_task([0.0, 10.0], [[0.0, 0.0, -50.0], [120.0, 0.0, -50.0]], 12.0),
        orbit_task([0.0, 0.0, -50.0], 100.0, 12.0),
    ],
)
def test_task_protocol_includes_all_public_task_families(task):
    assert isinstance(task, Task)
    assert jnp.isfinite(task.reference_speed())


@pytest.mark.parametrize("function", [reset, step, observation, rollout_actions, rollout_policy])
def test_episode_signatures_use_the_complete_task_contract(function):
    assert get_type_hints(function)["task"] is Task


def test_task_protocol_does_not_classify_unrelated_values_as_tasks():
    assert not isinstance(object(), Task)
    assert not isinstance({"reference_speed": 12.0}, Task)
