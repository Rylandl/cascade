# Steady turns and airspeed gain schedules

These tools extend the same full-aircraft dynamics used by ordinary trim and rollouts.
They describe mathematical equilibria and simulator controllers. Successful trim, knot
acceptance, or interpolation does not establish a physical flight envelope.

## Steady-turn trim

```python
import cascade
from cascade.analysis import SteadyTurnCondition, trim_steady_turn

model = cascade.aerobatic_reference()
turn = trim_steady_turn(
    model,
    SteadyTurnCondition(airspeed_m_s=12.0, turn_rate_rad_s=0.2, altitude_m=50.0),
)
assert turn.success, turn.message
```

Positive turn rate rotates the air-relative course clockwise in NED. The default path is
level; `flight_path_angle_rad` adds a constant climb angle. `heading_rad` chooses the initial
air-relative course. The solver keeps roll, pitch, speed and climb angle constant while yaw
advances at the requested rate. In body axes, the angular velocity is
`R_body_from_world @ [0, 0, turn_rate]`. Its roll component is generally nonzero when pitched.

The six reported residuals are:

```text
world acceleration - cross([0, 0, turn_rate], air-relative world velocity)  [m/s², 3]
body angular acceleration                                               [rad/s², 3]
```

The second term includes the rigid body's gyroscopic dynamics; it is not a zero-moment
approximation. The solver uses constant vertical gravity, density and uniform wind. Wind
translates the air-relative circle or helix, so its ground track need not be circular.
Internal actuator and separation states are equilibrated at the turning local flow.

`TurnTrimResult` has the diagnostics and decision layout of `TrimResult`: roll, pitch, yaw
offset, propeller commands, then control channels. `acceleration_norm` is the centripetal
balance **error**, not the nonzero turn acceleration itself. Success requires optimizer
success and a scaled physical residual within tolerance; the small bank/yaw regularizer
does not enter this test. The trim permits sideslip and merely prefers a coordinated solution.
Infeasible conditions return inspectable failed candidates. Zero turn rate delegates to the
straight-flight solver exactly. Warm-start a nearby condition with `initial_decision=turn`.

The solver is host-side SciPy using JAX residuals and Jacobians. The returned state and control
work in JIT/vmap simulation. Component and whole-aircraft coefficient backends are both covered
by circular-rollout regression tests, including left/right turns and a wind-translated climb.

## Building and using a gain schedule

```python
import jax.numpy as jnp
from cascade.control import (
    controller_at_speed,
    initial_cascade_state,
    scheduled_cascade_step,
    tune_gain_schedule,
)

spec = cascade.aerobatic_reference_spec()
schedule, report = tune_gain_schedule(spec, [12.0, 16.0])
controller = controller_at_speed(schedule, 14.0)
trim = report.tuning[0].trim
memory = initial_cascade_state(controller, trim.state, trim.control)
```

`tune_gain_schedule` tunes the same aircraft at each speed with `tune_cascade`. Every knot must
have a successful trim and a finite, settled `step_response`; failed knots raise with their
airspeed and are not silently removed. `ScheduleReport.tuning` and `.responses` retain the
acceptance evidence, and `.simulation_dt_s` / `.response_duration_s` record the verification
timing. Use the same simulation timestep and loop periods when relying on that acceptance.
The default test lasts 12 s, applying heading and altitude steps at 2 s.
It accepts final heading error below 5°, altitude error below 1.5 m, and airspeed error below
2 m/s. Callers still need maneuver-specific checks between knots and during speed changes.

`build_gain_schedule(speeds, controllers)` stacks custom, already accepted points without
rerunning those acceptance tests. Grids must contain at least two finite, positive, strictly
increasing speeds. Controllers must have common unbatched shapes, identical channel mappings
and limits, and identical positive integer loop periods. Parameter values and limits are
validated before stacking. Knots that collapse in the working JAX precision are rejected.

The pure JAX `controller_at_speed` interpolates rate/attitude/guidance gains, rate feedforward,
trim throttle, trim pitch and the scheduled limits. Channel mappings and loop periods remain
fixed. Queries are scalar; use `jax.vmap` for independently scheduled worlds.

| Endpoint policy | Behavior |
| --- | --- |
| `bounds="clamp"` (default) | Use the nearest stored endpoint outside the interval; compatible with JIT/vmap. |
| `bounds="raise"` | Reject an out-of-range query on the host. Traced calls explicitly raise rather than bypassing the check. |

Nonfinite host queries raise. Nonfinite traced measurements propagate nonfinite gains; validate
the measurement boundary or monitor controller outputs. No extrapolation is performed.

`scheduled_cascade_step(schedule, memory, setpoint, state, environment, dt)` schedules on the
current airspeed relative to the supplied wind, then runs the normal cascade update. It passes
the existing `CascadeState` through: integrals, derivative history, held outputs and step index
are not reinitialized when speed crosses a knot. Normal integrator clipping and scheduled loop
updates still apply. This preserves state but does not guarantee a bump-free response to an
abrupt change in speed or limits.

`scheduled_closed_loop_rollout` scans this update with plant integration and accepts the same
time-major `GuidanceSetpoint` and optional `environments` convention as `closed_loop_rollout`.
It returns the final `(AircraftState, CascadeState)` and post-step state/control/controller-state
trajectories. Default endpoint clamping applies throughout.

## Reproducible example

```sh
python examples/flight_envelope.py --output dist/verification/flight-envelope.json
```

The example checks left/right trim rollouts against analytic circles, builds an accepted
airspeed schedule, and tracks a command inside the interpolation interval. It asserts physical
position and tracking tolerances and writes configuration, measured errors and model/runtime
provenance. The focused regression suites are `tests/test_turn_trim.py` and
`tests/test_scheduling.py`.
