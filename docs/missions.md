# Missions

Missions extend the existing hold/hover/transition tasks with changing targets. They reuse
the same aircraft equations, action mapping, sensor pipeline and episode functions.
Builders validate finite inputs, dimensions, positive airspeeds, nonnegative cost weights
and increasing times before JIT compilation. Mission objects are JAX PyTrees; use `vmap`
for batches with matching mission types and numbers of knots.

## Scheduled speed, altitude and heading

```python
import jax
from cascade import aerobatic_reference
from cascade.control import aerobatic_reference_controller
from cascade.env import (
    EpisodeConfig,
    cascade_policy,
    reset,
    rollout_policy,
    scheduled_tracking_task,
)

model = aerobatic_reference()
mission = scheduled_tracking_task(
    times_s=[0, 2, 6, 10],
    airspeed_m_s=[12, 12, 14, 14],
    altitude_m=[50, 50, 55, 55],
    heading_rad=[0, 0, 0.3, 0.3],
)
config = EpisodeConfig(horizon_steps=400)  # 10 seconds at the default 40 Hz
reference = mission.reference(model)
policy, policy_state = cascade_policy(
    aerobatic_reference_controller(),
    model,
    config,
    mission,
    reference,
)
state, observation = reset(model, config, mission, reference, jax.random.PRNGKey(7))
final, (observations, actions, rewards, dones) = jax.jit(
    lambda state: rollout_policy(
        model,
        config,
        mission,
        reference,
        state,
        policy,
        policy_state,
    )
)(state)
```

Each target may be a scalar or one value per time knot. Times start at zero. Interpolation
is linear and values before/after the schedule hold the endpoints. Adjacent headings use
the shortest angular arc; provide intermediate knots for turns of 180 degrees or more.
This defines commands, not a guarantee that the aircraft can follow them.

`task_at(mission, time_s, rigid_body)` returns the instantaneous target. Static tasks pass
through unchanged. `EnvState.time_s` is the control-step index divided by control frequency.
At reset, the observation uses time zero. A policy uses the current time/state; after a step,
both reward and the newly acquired observation use the next time/state. Delayed sensors
retain the target errors that were present at acquisition, just like the other readings.
Rewards charge the commanded action even when actuator transport delays apply.

The baseline accepts either a `CascadeController` or a `GainSchedule`. A schedule is sampled
at measured air-relative speed and preserves its controller state; each baseline loop runs
once per environment control period, as with the original fixed-controller baseline.
It reads the true simulator state; it is a privileged baseline, not an estimator subject
to observation noise, delay or dropout. Learned policies can instead use the sensed vector.

## Timed waypoints

```python
from cascade.env import waypoint_task

mission = waypoint_task(
    times_s=[0, 10, 20],
    positions_ned=[[0, 0, -50], [120, 0, -50], [120, 120, -55]],
    airspeed_m_s=12,
    lookahead_s=2,
    position_scale_m=10,
)
```

The desired NED position moves linearly between waypoints. Guidance points from the actual
aircraft position toward a future path point, while the position cost and observation refer
to the path point at the current time. Consecutive points must differ horizontally; this
fixed-wing task does not model stopping or vertical-only hover moves. At a coincident
lookahead point, heading falls back to the current path segment's direction.

These are **timed waypoints**, not arrival-triggered waypoint switching. Set the episode
horizon to end at the final waypoint time. After that time the final position is held, which
does not supply a fixed-wing loiter maneuver. Segment timing specifies ground motion while
airspeed is an explicit target; check their compatibility with wind and aircraft limits.
Corners are not smoothed automatically. Use sufficiently gentle paths and lookahead settings.

## Orbits

```python
from cascade.env import orbit_task

mission = orbit_task(
    center_ned=[0, 0, -50],
    radius_m=100,
    airspeed_m_s=12,
    clockwise=True,
    capture_distance_m=100,
)
```

The target is the nearest point on the horizontal circle. Heading combines its tangent with
an inward/outward correction proportional to radial displacement through an arctangent.
`capture_distance_m` sets the radial correction's scale. There is no required angular phase
or lap time. `initial_phase_rad` starts clockwise from north; its default is the north point.
Clockwise flight there initially commands east. The center's NED z value sets altitude.

`reference(model)` starts from a straight tangent trim at the initial point; it does not
solve a steady-turn trim. The controller must capture the orbit. Radius/airspeed feasibility,
crosswind compensation and collision avoidance are not supplied by this guidance law.

## Costs and scope

All missions retain the tracking task's speed, altitude, heading, rate and effort costs.
Waypoints/orbits add `position_weight * horizontal_error² / position_scale²`; the waypoint
scale defaults to 10 m and orbit scale is its radius. Altitude keeps its existing 10 m cost
scale, and position observation errors keep the existing 10 m observation scale. Mission
position errors are actual minus desired in NED before conversion to body axes.

Mission completion does not introduce an extra termination condition: crash/attitude limits
and the configured episode horizon still apply. These are research tasks and guidance
baselines, not flight-plan execution or certified navigation. Check tracking performance,
actuator saturation, winds and termination for each chosen scenario.
