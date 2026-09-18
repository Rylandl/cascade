import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cascade
from cascade.env.faults import apply_faults, fault_schedule, no_faults
from cascade.integration import repeat_control, rollout


def _fly(model, state, control, environment, schedule, seconds, dt=0.0025, period=0.025):
    """Roll out one control period at a time, applying the schedule at each period's start."""

    states = []
    substeps = int(round(period / dt))
    controls = repeat_control(control, substeps)
    advance = jax.jit(lambda m, s: rollout(m, s, controls, environment, dt)[0])
    for k in range(int(seconds / period)):
        faulted = apply_faults(model, schedule, k * period)
        state = advance(faulted, state)
        states.append(state)
    return states


def test_jam_holds_a_surface_and_hardover_drives_it_to_the_limit():
    spec = cascade.aerobatic_reference_spec()
    model = spec.to_model()
    environment = cascade.standard_environment()
    state = cascade.zero_state(model, altitude=100.0, forward_speed=12.0)
    command = cascade.ControlInput(propeller=jnp.array([0.5]), channel=jnp.array([0.6, 0.0, 0.0]))
    plain = _fly(model, state, command, environment, no_faults(model), 0.5)
    left = [s.name for s in spec.surfaces].index("left_wing")
    moved = float(plain[-1].actuators.surface_deflection[left])
    assert abs(moved) > 0.1
    # Jam the left aileron at 0.1 s: it freezes near where it was, whatever is commanded.
    jam = fault_schedule(model, jams={left: 0.1})
    jammed = _fly(model, state, command, environment, jam, 0.5)
    frozen = float(jammed[3].actuators.surface_deflection[left])
    assert abs(float(jammed[-1].actuators.surface_deflection[left]) - frozen) < 1e-3
    assert abs(frozen) < abs(moved)
    # Hardover at 0.1 s drives the surface to its negative limit and holds it there.
    hard = fault_schedule(model, hardovers={left: (0.1, -1.0)})
    over = _fly(model, state, command, environment, hard, 0.5)
    limit = float(model.actuators.surface_limit[left])
    assert abs(float(over[-1].actuators.surface_deflection[left]) + limit) < 0.02
    # Other surfaces are untouched by either fault.
    for other in range(model.n_surfaces):
        if other != left:
            assert jnp.allclose(
                jammed[-1].actuators.surface_deflection[other],
                plain[-1].actuators.surface_deflection[other],
                atol=1e-5,
            )


def test_motor_out_spins_down_and_partial_power_derates():
    model = cascade.aerobatic_reference()
    environment = cascade.standard_environment()
    state = cascade.zero_state(model, altitude=100.0, forward_speed=12.0)
    command = cascade.ControlInput(propeller=jnp.array([0.8]), channel=jnp.zeros(3))
    state = cascade.equilibrate_internal_state(model, state, command, environment)
    running = float(state.actuators.propeller_speed[0])
    out = _fly(model, state, command, environment, fault_schedule(model, motor_out={0: 0.0}), 1.0)
    assert 0.5 * running < float(out[0].actuators.propeller_speed[0]) < running
    assert float(out[-1].actuators.propeller_speed[0]) < 0.05 * running
    half = _fly(
        model,
        state,
        command,
        environment,
        fault_schedule(model, partial_power={0: (0.0, 0.5)}),
        1.0,
    )
    derated = float(half[-1].actuators.propeller_speed[0])
    assert 0.6 * running < float(half[0].actuators.propeller_speed[0]) < running
    assert 0.4 * running < derated < 0.6 * running
    # Before its time a fault does nothing: the schedule equals the nominal model.
    later = apply_faults(model, fault_schedule(model, motor_out={0: 5.0}), 1.0)
    assert np.allclose(
        np.asarray(later.actuators.propeller_speed_max),
        np.asarray(model.actuators.propeller_speed_max),
    )


def test_motor_fault_transients_match_the_lag_under_jit_and_vmap():
    model = cascade.aerobatic_reference()
    # Make rate saturation negligible so the first-order response has an analytic solution.
    tau = 0.2
    model = model._replace(
        actuators=model.actuators._replace(
            propeller_time_constant=jnp.full((model.n_propellers,), tau),
            propeller_acceleration_limit=jnp.full((model.n_propellers,), 1e9),
        )
    )
    control = cascade.ControlInput(propeller=jnp.ones(1), channel=jnp.zeros(3))
    environment = cascade.standard_environment()._replace(density=jnp.asarray(0.0))
    state = cascade.equilibrate_internal_state(
        model, cascade.zero_state(model, altitude=100.0), control, environment
    )
    schedules = jax.tree.map(
        lambda *values: jnp.stack(values),
        fault_schedule(model, motor_out={0: 0.0}),
        fault_schedule(model, partial_power={0: (0.0, 0.5)}),
    )
    dt, steps = 0.005, 20

    def fly(schedule):
        faulted = apply_faults(model, schedule, 0.0)
        return rollout(faulted, state, repeat_control(control, steps), environment, dt)[1]

    trajectory = jax.jit(jax.vmap(fly))(schedules)
    speed = np.asarray(trajectory.actuators.propeller_speed[..., 0])
    initial = float(state.actuators.propeller_speed[0])
    targets = initial * np.array([0.0, 0.5])
    time = dt * np.arange(1, steps + 1)
    expected = targets[:, None] + (initial - targets[:, None]) * np.exp(-time / tau)
    np.testing.assert_allclose(speed, expected, rtol=2e-6, atol=1e-4)
    assert np.all(np.diff(speed, axis=1) < 0.0)


@pytest.mark.parametrize("initial_fraction,throttle", [(0.0, 1.0), (1.0, 0.0), (0.5, 2.0)])
def test_nominal_motor_stays_inside_initial_and_target_speed(initial_fraction, throttle):
    model = cascade.aerobatic_reference()
    state = cascade.zero_state(model, altitude=100.0)
    maximum = model.actuators.propeller_speed_max
    initial = initial_fraction * maximum
    state = state._replace(actuators=state.actuators._replace(propeller_speed=initial))
    control = cascade.ControlInput(propeller=jnp.array([throttle]), channel=jnp.zeros(3))
    environment = cascade.standard_environment()._replace(density=jnp.asarray(0.0))
    _, trajectory = jax.jit(rollout)(
        model, state, repeat_control(control, 400), environment, 0.0025
    )
    speed = np.asarray(trajectory.actuators.propeller_speed)
    target = np.asarray(maximum) * np.clip(throttle, 0.0, 1.0)
    assert np.all(speed >= np.minimum(initial, target) - 1e-5)
    assert np.all(speed <= np.maximum(initial, target) + 1e-5)
    np.testing.assert_allclose(speed[-1], target, rtol=0.01, atol=1.0)


@pytest.fixture(scope="module")
def fault_model():
    return cascade.aerobatic_reference()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"jams": {-1: 0.0}}, "surface index"),
        ({"jams": {999: 0.0}}, "surface index"),
        ({"jams": {0.5: 0.0}}, "surface index"),
        ({"motor_out": {1: 0.0}}, "propeller index"),
        ({"motor_out": {True: 0.0}}, "propeller index"),
        ({"jams": {0: -0.1}}, "fault time"),
        ({"hardovers": {0: (float("nan"), 1.0)}}, "fault time"),
        ({"motor_out": {0: -float("inf")}}, "fault time"),
        ({"hardovers": {0: (0.0, 0.5)}}, "hardover sign"),
        ({"hardovers": {0: (0.0, float("nan"))}}, "hardover sign"),
        ({"partial_power": {0: (0.0, -0.1)}}, "fraction"),
        ({"partial_power": {0: (0.0, 1.1)}}, "fraction"),
        ({"partial_power": {0: (0.0, float("nan"))}}, "fraction"),
        ({"partial_power": {0: (0.0, float("inf"))}}, "fraction"),
    ],
)
def test_fault_schedule_rejects_invalid_host_inputs(fault_model, kwargs, match):
    with pytest.raises(ValueError, match=match):
        fault_schedule(fault_model, **kwargs)


def test_infinite_fault_time_is_an_explicit_never(fault_model):
    schedule = fault_schedule(fault_model, motor_out={0: float("inf")})
    unchanged = apply_faults(fault_model, schedule, 1e6)
    for actual, expected in zip(
        jax.tree.leaves(unchanged), jax.tree.leaves(fault_model), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
