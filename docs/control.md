# Control cascade

`cascade.control` is a PX4-style fixed-wing autopilot stack — rate, attitude, and guidance loops,
each a pure function of a PyTree of gains and a PyTree of controller state — composed into a
rate-scheduled cascade and a closed-loop rollout that scans the plant and the controller together.
Everything here is `jit`/`vmap`/`grad` compatible and batches over worlds with the same leading
dimensions as `AircraftState`; nothing branches in Python on an array value, so loop scheduling is
built from `jnp.where` rather than conditionals.

## Loop structure

```text
GuidanceSetpoint (airspeed, altitude, heading)
        │  guidance_controller           every guidance_period steps
        ▼
attitude setpoint (quaternion), throttle
        │  attitude_controller           every attitude_period steps
        ▼
body-rate setpoint [roll, pitch, yaw]
        │  rate_controller               every rate_period steps
        ▼
unit command [roll, pitch, yaw]
        │  ChannelMap
        ▼
ControlInput.channel                     fed to the plant every simulation step
```

### Rate loop — `rate_controller`

PID-plus-feedforward per body axis:

```text
error = rate_setpoint - rate_measured
integral' = clip(integral + error·dt, -integral_limit, integral_limit)
derivative = (error - previous_error) / dt
output = kp·error + ki·integral' + kd·derivative + feedforward·rate_setpoint
```

`RateState` carries only `integral` and `previous_error`, so the derivative term is a plain finite
difference of the error (not a filtered derivative-on-measurement); keep `kd` small relative to
`kp` or a step change in setpoint produces a one-step "derivative kick" (`kd / dt` times the
step). Anti-windup is a hard clip of the integral to `integral_limit` every step. The output is
unit `[roll, pitch, yaw]` commands, not yet in the aircraft's channel units.

### Attitude loop — `attitude_controller`

Proportional-only. The attitude error is the shortest rotation from measured to commanded
attitude, `q_error = q_measured⁻¹ · q_setpoint`, converted to a body-frame rotation vector and
scaled per axis, then saturated to `rate_limit`:

```text
rate_setpoint = clip(kp · quaternion_to_rotvec(q_error), -rate_limit, rate_limit)
```

There is no attitude integrator. A persistent attitude bias — trim drift, an unmodeled moment, the
steady coupling a banked no-rudder airframe produces — is left uncorrected by this loop; only the
rate loop's own integral, or an updated trim, removes it. This matters most for an airframe with a
lightly damped or slowly divergent open-loop mode (see the X8 example below): the attitude loop
settles a *step* comfortably inside a few seconds, but a small residual can drift over tens of
seconds if nothing above it (the guidance loop, over a much longer averaging time) is closing that
loop too.

### Guidance loop — `guidance_controller`

Three independent single-input single-output loops, **not TECS** (Total Energy Control System):
altitude and airspeed do not share energy, so commanding a climb costs airspeed and vice versa,
exactly as an amateur decoupled autopilot without a throttle/pitch energy mix would.

- **Altitude → pitch.** Altitude error feeds a proportional climb-rate command, saturated to
  `climb_rate_limit`. The climb-rate command becomes a pitch offset through the small-angle
  relation `climb_rate / airspeed` (flight-path angle, not angle of attack) about `pitch_trim`,
  reduced by `airspeed_pitch_kp` times any airspeed deficit below the commanded airspeed (a
  stall-protection term), then saturated to `pitch_limits`.
- **Airspeed → throttle.** Airspeed error feeds a PI throttle about `throttle_trim`, saturated to
  `throttle_limits`. The integral is clamped to the value that alone would span the throttle
  range (`throttle_span / airspeed_ki`) — a cheap anti-windup that needs no extra gain field.
- **Heading → bank.** Heading error, wrapped to `[-π, π]`, feeds a proportional bank command,
  saturated to `bank_limit`. The attitude setpoint's yaw is set to the aircraft's *current*
  measured yaw, not the commanded heading:

  ```python
  attitude_setpoint = quaternion_from_euler(bank_setpoint, pitch_setpoint, heading_measured)
  ```

  so the attitude loop only ever closes roll and pitch and never fights the heading loop
  directly — the bank angle alone produces the coordinated turn, the same way a human pilot
  banks and lets the nose come around rather than stepping on the rudder to yaw toward a heading.

There is no wind feedforward: a crosswind or head/tailwind shift is corrected only through the
airspeed and heading errors it eventually causes, not anticipated.

## Sign conventions and `ChannelMap`

Everything above works in the aircraft-agnostic unit convention: **positive roll command demands
positive body roll rate (right wing down); positive pitch demands positive body pitch rate (nose
up); positive yaw demands positive body yaw rate (nose right)**. `ChannelMap` is what turns that
into the aircraft's actual channels:

```python
class ChannelMap(NamedTuple):
    matrix: Array   # [C, 3]: unit [roll, pitch, yaw] -> per-channel command
    limit: Array    # [C]: symmetric clip per channel, in the spec's own channel units
```

`channel_map(spec, roles, limits)` builds it from named roles, one of `"roll"`, `"pitch"`, or
`"yaw"` per channel (channels absent from `roles` get an all-zero row and are never commanded).
Prefix a role with `-` — for example `"-pitch"` — when a *positive* channel command drives that
axis *negative* for this airframe. That sign is a property of the airframe, not of the channel's
name, and must be checked, never assumed. Two ways to check it:

1. **Static sweep.** `cascade.aerodynamic_sweep(model, alpha, control=...)` at a small nonzero
   channel value and read `moment_coefficient_body`, `[Cl, Cm, Cn]` about body roll/pitch/yaw.
2. **Dynamic rollout.** A few RK4 steps from trim with one channel perturbed, reading
   `state.rigid_body.angular_velocity`.

For both packaged aircraft, elevator turned out to need the flipped sign — it is trailing-edge-down
positive, which is a nose-down (negative pitch-rate) moment on both the aerobatic reference and the
X8 — while aileron and rudder map directly:

```python
# aerobatic reference: aileron, elevator, rudder channels, [-1, 1] normalized
channel_map(spec, roles={"aileron": "roll", "elevator": "-pitch", "rudder": "yaw"}, limits=1.0)

# Skywalker X8: aileron, elevator channels (no rudder), radians
channel_map(
    spec, roles={"aileron": "roll", "elevator": "-pitch"},
    limits={"aileron": 0.35, "elevator": 0.35},
)
```

The X8's channel limits (`0.35` rad each) are deliberately narrower than the physical elevon limit
(`0.7` rad): its elevons mix additively (`left = aileron + elevator`, `right = elevator - aileron`),
so an unclipped combination of full aileron and full elevator could drive one elevon well past its
own stall angle. Clipping each generalized channel first keeps every combination inside the
elevons' attached-flow range.

## Scheduling and `CascadeController`

```python
class CascadeController(NamedTuple):
    channels: ChannelMap
    rate: RateGains
    attitude: AttitudeGains
    guidance: GuidanceGains
    rate_period: int        # simulation steps between rate-loop updates
    attitude_period: int    # simulation steps between attitude-loop updates
    guidance_period: int    # simulation steps between guidance-loop updates
```

`cascade_step` evaluates every loop on *every* call — nothing here branches on an array value —
and uses `jnp.where(step_index % period == 0, fresh, held)` at each level to decide whether that
level's output actually changes this step, matching a real flight-control stack where an inner
loop runs faster than the loops above it. A level's own state (the rate loop's integrator, the
guidance loop's airspeed integrator) only advances on its own schedule too, and does so using that
schedule's *elapsed* time (`period * dt`) rather than the raw simulation `dt`, so integral gains
are tuned against the rate the loop actually executes at. `initial_cascade_state(controller,
aircraft_state, control)` zeroes every integrator and holds `control` (typically a trim) until the
first update; `step_index` starts at zero.

`closed_loop_rollout(model, controller, aircraft_state, cascade_state, setpoints, environment, dt)`
scans a time-major sequence of `GuidanceSetpoint`s, evaluating `cascade_step` at the *current*
state and then advancing the plant one `step` (`rk4_step` by default), returning the final
`(state, cascade_state)` and the post-step, time-major `(state, control, cascade_state)`
trajectory — the same `environments` convention as `cascade.rollout`.

## Building a `ChannelMap` for a new airframe

1. Load the spec: `spec = cascade.load_aircraft_spec(path)`.
2. Decide which named channel drives which axis. A conventional layout is straightforward
   (aileron → roll, elevator → pitch, rudder → yaw); an elevon, flaperon, or V-tail mix is already
   resolved by the spec's `control_map_rad` (or the coefficient table's `deflection_map`), so the
   *generalized* channel names are what matters here, not the physical surfaces underneath them.
3. Check every role's sign with `cascade.aerodynamic_sweep` or a short open-loop rollout from
   trim, as above. Do not assume a channel's name implies its sign.
4. Pick `limits` conservatively relative to the physical actuator limits, especially for any
   channel that mixes into a shared physical surface with another channel (see the X8 above).
5. Tune `RateGains`, `AttitudeGains`, and `GuidanceGains` against closed-loop step responses
   (below), not open-loop step responses — see the note on lightly damped modes.

## Tuning procedure and the default gains

Every default gain in `aerobatic_reference_controller()` and `skywalker_x8_controller()` was found
by running an actual closed-loop rollout and reading off the step response, in this order:

1. **Rate loop**, roll axis first (it is normally the best-damped): step the rate setpoint, hold
   throttle at trim, and adjust `kp`/`ki` until the 90%-rise time and overshoot are acceptable.
   Prefer `kp` and `ki` over `kd`: because `RateState` differentiates the raw *error*, not the
   measurement, `kd` produces a one-step kick on every setpoint change, and too much of it can do
   more harm than good.
2. **Check the airframe's own damping** with `cascade.linearize_step` and `cascade.stability_modes`
   at the trim point before tuning pitch or yaw. The X8's short-period mode has a damping ratio of
   about 0.10 versus about 0.33 for the aerobatic reference — proportional-only rate feedback
   rings that mode up instead of damping it (a *smaller* gain does not help; the plant's own
   lightly damped response dominates a weak loop just as much as a strong one). A modest `kd`
   fixes it: the X8's pitch axis carries real derivative gain for exactly this reason, while its
   roll axis (well damped on its own) needs none.
3. **Attitude loop**: step a 20° roll / 5° pitch attitude setpoint through the tuned rate loop and
   raise `kp`/`rate_limit` until the response settles inside the required window with acceptable
   overshoot. A proportional-only loop against a coupled airframe (no rudder, in the X8's case)
   settles to a small non-zero residual within a couple of seconds — see the attitude-loop
   docstring above — so judge settling against the window the maneuver is actually specified over,
   not against an indefinitely long hold.
4. **Guidance loop**: step altitude and airspeed together, then a heading turn alone, watching
   angle of attack (stall margin), altitude drift during the turn, and the achieved bank against
   `bank_limit`. `heading_kp` in particular has a narrow stable range for an airframe whose bank
   angle self-induces yaw (any coordinated-turn airframe): too high and the loop's own turn-rate
   feedback can pump energy into the roll/yaw coupling instead of damping it.

Tuned numbers, and the model characteristics behind each choice, are documented in the docstrings
of `aerobatic_reference_controller()` and `skywalker_x8_controller()` in `cascade/control.py`.

## Differentiable tuning

Every gain is a JAX array, so a tracking error is differentiable with respect to it end to end
through the rollout:

```python
import jax
import jax.numpy as jnp
import cascade
from cascade.control import RateState, rate_controller

model = cascade.aerobatic_reference()
controller = cascade.aerobatic_reference_controller()
trim = cascade.trim_straight_flight(
    model, cascade.StraightFlightCondition(airspeed_m_s=12.0, altitude_m=20.0)
)
environment = cascade.standard_environment()
rate_setpoint = jnp.array([1.0, 0.0, 0.0])  # 1 rad/s roll-rate step

def tracking_error(kp):
    gains = controller.rate._replace(kp=kp)

    def step(carry, _):
        state, rate_state = carry
        command, rate_state = rate_controller(
            gains, rate_state, rate_setpoint, state.rigid_body.angular_velocity, 0.01
        )
        channel = jnp.clip(
            jnp.einsum("cr,r->c", controller.channels.matrix, command),
            -controller.channels.limit, controller.channels.limit,
        )
        control = cascade.ControlInput(propeller=trim.control.propeller, channel=channel)
        next_state = cascade.rk4_step(model, state, control, environment, 0.01)
        return (next_state, rate_state), next_state

    rate_state0 = RateState(integral=jnp.zeros(3), previous_error=jnp.zeros(3))
    _, trajectory = jax.lax.scan(step, (trim.state, rate_state0), None, length=100)
    roll_rate = trajectory.rigid_body.angular_velocity[:, 0]
    return jnp.mean(jnp.square(roll_rate - 1.0))

kp = jnp.array([0.02, 0.02, 0.02])  # deliberately detuned
grad_fn = jax.grad(tracking_error)
for _ in range(20):
    kp = kp - 0.02 * grad_fn(kp)
```

Twenty plain gradient-descent steps from a deliberately detuned `kp` (`0.02` on every axis, versus
the tuned `0.12` on roll) roughly halve the mean-squared roll-rate tracking error over the 1 s
window in `tests/test_control.py`. Nothing here is specific to `kp` — any gain in `RateGains`,
`AttitudeGains`, or `GuidanceGains` composes the same way, and the same rollout batches over a
population of gain candidates with `jax.vmap` for a coarse hyperparameter search before refining
with gradients.

## Explicit limitations

- **Guidance is three decoupled loops, not TECS.** Climbing costs airspeed and accelerating costs
  altitude exactly as they would with no cross-channel compensation; a real energy-managed
  autopilot would trade throttle and pitch against total energy instead.
- **No wind feedforward.** Wind is only ever corrected reactively, through the airspeed and
  heading error it causes, never anticipated from a wind estimate.
- **No attitude integrator.** A sustained bias in roll or pitch attitude is left to the rate
  loop's own integral term or to an updated trim; the attitude loop by itself does not remove it.
- **The heading loop assumes a coordinated-turn airframe.** Yaw is not commanded directly (the
  attitude setpoint's yaw tracks the vehicle's own current yaw); an airframe that does not turn
  from bank alone (no dihedral effect, or an unusual configuration) will not turn well under this
  guidance loop as written.
