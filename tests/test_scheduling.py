import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cascade
from cascade.control import (
    GuidanceSetpoint,
    build_gain_schedule,
    controller_at_speed,
    initial_cascade_state,
    scheduled_cascade_step,
    scheduled_closed_loop_rollout,
    step_response,
    tune_gain_schedule,
)
from cascade.control.tuned import aerobatic_reference_controller


@pytest.fixture(scope="module")
def tuned_schedule():
    return tune_gain_schedule(cascade.aerobatic_reference_spec(), [12.0, 16.0])


def test_tuned_schedule_accepts_knots_and_controls_an_intermediate_trim(tuned_schedule):
    schedule, report = tuned_schedule
    assert all(item.trim.success for item in report.tuning)
    assert all(item.finite and item.settled for item in report.responses)
    model = cascade.aerobatic_reference()
    trim = cascade.trim_straight_flight(model, cascade.StraightFlightCondition(14, altitude_m=50))
    controller = jax.jit(controller_at_speed)(schedule, jnp.array(14.0))
    response = step_response(model, controller, trim)
    assert response.finite and response.settled, response


def test_schedule_interpolation_is_jitted_vmappable_and_preserves_endpoints(tuned_schedule):
    schedule, _ = tuned_schedule
    speeds = jnp.array([10.0, 12.0, 14.0, 16.0, 20.0])
    batched = jax.jit(jax.vmap(lambda speed: controller_at_speed(schedule, speed)))(speeds)
    assert np.allclose(batched.rate.kp[0], schedule.controllers.rate.kp[0])
    assert np.allclose(batched.rate.kp[-1], schedule.controllers.rate.kp[-1])
    for index, speed in enumerate(speeds):
        scalar = controller_at_speed(schedule, speed)
        for a, b in zip(jax.tree.leaves(scalar), jax.tree.leaves(batched), strict=True):
            assert np.allclose(a, b[index])
    derivative = jax.grad(lambda v: controller_at_speed(schedule, v).guidance.throttle_trim)(14.0)
    assert jnp.isfinite(derivative) and abs(float(derivative)) > 1e-4


def test_scheduling_preserves_integrators_held_outputs_and_update_counter(tuned_schedule):
    schedule, report = tuned_schedule
    state = report.tuning[0].trim.state
    control = report.tuning[0].trim.control
    controller = controller_at_speed(schedule, 12.0)
    memory = initial_cascade_state(controller, state, control)
    memory = memory._replace(
        step_index=jnp.array(1),
        rate=memory.rate._replace(integral=jnp.array([0.1, -0.1, 0.05])),
        guidance=memory.guidance._replace(airspeed_integral=jnp.array(0.7)),
    )
    # Change the scheduling speed while guidance and attitude updates are held.
    state = state._replace(
        rigid_body=state.rigid_body._replace(velocity=jnp.array([14.0, 0.0, 0.0]))
    )
    setpoint = GuidanceSetpoint(jnp.array(14.0), jnp.array(50.0), jnp.array(0.0))
    _, updated = jax.jit(scheduled_cascade_step)(
        schedule, memory, setpoint, state, cascade.standard_environment(), 0.0025
    )
    assert int(updated.step_index) == 2
    assert np.array_equal(updated.held_attitude_setpoint, memory.held_attitude_setpoint)
    assert float(updated.guidance.airspeed_integral) == pytest.approx(0.7)
    assert np.linalg.norm(updated.rate.integral - memory.rate.integral) < 0.02
    assert np.linalg.norm(updated.rate.integral) > 0.1


def test_scheduled_rollout_tracks_a_speed_change_through_the_interval(tuned_schedule):
    schedule, report = tuned_schedule
    model = cascade.aerobatic_reference()
    trim = report.tuning[0].trim
    controller = controller_at_speed(schedule, 12.0)
    memory = initial_cascade_state(controller, trim.state, trim.control)
    dt, count = 0.0025, 4800
    speed = jnp.where(jnp.arange(count) * dt < 2, 12.0, 15.0)
    targets = GuidanceSetpoint(speed, jnp.full(count, 50.0), jnp.zeros(count))
    (final, updated), (trajectory, _, _) = jax.jit(scheduled_closed_loop_rollout)(
        model, schedule, trim.state, memory, targets, cascade.standard_environment(), dt
    )
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(trajectory))
    assert abs(float(jnp.linalg.norm(final.rigid_body.velocity)) - 15.0) < 1.0
    assert abs(float(final.rigid_body.position[2]) + 50.0) < 1.5
    assert int(updated.step_index) == count


@pytest.mark.parametrize("grid", [[12.0], [12.0, 12.0], [16.0, 12.0], [-1.0, 12.0], [12.0, np.nan]])
def test_invalid_grids_rejected(grid):
    with pytest.raises(ValueError, match="knots"):
        build_gain_schedule(grid, [aerobatic_reference_controller()] * len(grid))


def test_invalid_periods_mappings_shapes_and_limits_rejected():
    controller = aerobatic_reference_controller()
    variants = [
        controller._replace(rate_period=0),
        controller._replace(rate_period=2),
        controller._replace(rate_period=1.5),
        controller._replace(
            channels=controller.channels._replace(matrix=-controller.channels.matrix)
        ),
        controller._replace(rate=controller.rate._replace(kp=jnp.ones(2))),
        controller._replace(
            guidance=controller.guidance._replace(pitch_limits=jnp.array([1.0, -1.0]))
        ),
    ]
    for invalid in variants:
        with pytest.raises(ValueError):
            build_gain_schedule([12, 16], [controller, invalid])


def test_endpoint_raise_is_explicit_and_cannot_be_silently_disabled_under_jit():
    controller = aerobatic_reference_controller()
    schedule = build_gain_schedule([12, 16], [controller, controller])
    controller_at_speed(schedule, 12.0, bounds="raise")
    controller_at_speed(schedule, 16.0, bounds="raise")
    with pytest.raises(ValueError, match="outside"):
        controller_at_speed(schedule, 11.0, bounds="raise")
    with pytest.raises(ValueError, match="host-only"):
        jax.jit(lambda v: controller_at_speed(schedule, v, bounds="raise"))(14.0)
    with pytest.raises(ValueError, match="finite"):
        controller_at_speed(schedule, np.nan)
    # A compiled controller must not conceal an invalid infinite measurement by clamping it.
    for invalid in (np.nan, np.inf, -np.inf):
        queried = jax.jit(controller_at_speed)(schedule, jnp.asarray(invalid))
        assert jnp.all(jnp.isnan(queried.rate.kp))


def test_tuning_does_not_silently_skip_an_untrimmable_knot():
    with pytest.raises(ValueError, match="knot 1 m/s"):
        tune_gain_schedule(cascade.aerobatic_reference_spec(), [1.0, 12.0])
