import jax.numpy as jnp
import numpy as np

import cascade
from cascade.env.faults import apply_faults, fault_schedule, no_faults
from cascade.integration import repeat_control, rollout


def _fly(model, state, control, environment, schedule, seconds, dt=0.0025, period=0.025):
    """Roll out one control period at a time, applying the schedule at each period's start."""

    states = []
    substeps = int(round(period / dt))
    for k in range(int(seconds / period)):
        faulted = apply_faults(model, schedule, k * period)
        state, _ = rollout(faulted, state, repeat_control(control, substeps), environment, dt)
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
    assert 0.4 * running < derated < 0.6 * running
    # Before its time a fault does nothing: the schedule equals the nominal model.
    later = apply_faults(model, fault_schedule(model, motor_out={0: 5.0}), 1.0)
    assert np.allclose(
        np.asarray(later.actuators.propeller_speed_max),
        np.asarray(model.actuators.propeller_speed_max),
    )
