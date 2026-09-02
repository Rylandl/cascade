# Architecture

## Product boundary

Cascade models free-flying fixed-wing aircraft whose motion is governed by rigid-body mechanics,
aerodynamic elements, propellers, and actuators. The first milestone excludes ground contact,
landing gear, structural flexibility, onboard estimation, and rendering. Those features must not
leak assumptions into the dynamics API.

The initial quality target is credible simulation of conventional flight, stall entry, sustained
high-alpha motion, and recovery for one identified small aerobatic airframe. It is not universal
post-stall truth across uncalibrated geometries.

## Coordinate conventions

The physics layer follows conventional flight-dynamics coordinates:

- World: North-East-Down (NED), right handed; gravity is positive world `z`.
- Body: Forward-Right-Down (FRD), right handed.
- Attitude: scalar-last quaternion `xyzw`, rotating body vectors into world coordinates.
- Linear velocity: world frame.
- Angular velocity and moments: body frame.
- Surface frame: `x` along the chord, `y` along the span, and `z` completing the right-handed frame.

Keeping the aerospace convention inside the model makes coefficient data and flight-controller
integration less error-prone. MuJoCo, visualization, and robotics APIs should convert at their
boundaries rather than changing the physics convention. `cascade.canonical` is that boundary for
the NWU-world, FLU-body, scalar-first 13-vector state used by identification and control tooling
such as Glassbox; it is the only place frame conversion happens.

## Static model and dynamic state

The numerical model is a JAX PyTree of arrays. Array shapes describe a fixed topology at compile
time:

```text
AircraftModel
├── mass and inertia
├── SurfaceModel[S]
├── PropellerModel[P]
├── ActuatorModel[S, P, C]
└── BodyModel                whole-aircraft coefficient table, deflection_map[3, S]

AircraftState
├── RigidBodyState        position, attitude, velocity, angular velocity
├── ActuatorState         S surface angles, P propeller speeds
└── AeroState             S continuous separation fractions

ControlInput
├── propeller[P]          normalized 0..1 commands
└── channel[C]            linear control coordinates in the units the specification chose
```

`ActuatorModel.surface_map[S, C]` maps named host-side channels such as aileron, elevator, and
rudder into physical surface angles. Channels are linear coordinates in whatever units the
aircraft specification chose for its control map: normalized `[-1, 1]` commands for a
controller-facing airframe, or radians for one whose logs carry generalized surface angles.
Physical limits are enforced on the mapped angle, never on the channel. The map supports
elevons, V-tails, flaperons, differential surfaces, and unconventional layouts without branching
in the compiled dynamics. A surface may have zero area and exist only to carry a physical,
lagged, limited actuator whose angle feeds the coefficient table.

Each compiled rollout has one static aircraft topology. Numerical model arrays can carry the same
leading world dimensions as state, allowing mass, inertia, aerodynamic coefficients, and actuator
parameters to vary per rollout. Different counts of surfaces, propellers, or channels still produce
separate compilations. This is an intentional tradeoff for predictable JAX performance.

## Component aerodynamic model

For surface `i`, local velocity through the air is

```text
v_i = R_bw (v_world - wind_world) + omega_body x r_i + v_slipstream_i
```

and is rotated into the surface frame. Each surface independently computes airspeed, angle of
attack, dynamic pressure, attached-flow coefficients, and separated-flow coefficients. Forces are
rotated back to the body frame and moments are accumulated at the center of mass.

A physical deflection `delta` is split by the static `all_moving_fraction` `w`. The all-moving
share `w delta` rotates the surface frame, which is what a stabilator or V-tail does and shifts
the local angle of attack including its stall. The flap share `(1 - w) delta` leaves the frame
alone and enters the coefficients through the flap effectiveness `tau`:

```text
CL_att = CL0 + CLa (alpha + tau delta_flap)
CD_att = CD0 + k CL_att^2 + CD_flap delta_flap^2
Cm_att = Cm0 + Cma alpha + Cm_flap delta_flap
alpha_sep = alpha + tau delta_flap                  flat-plate incidence
```

The attached model is a conventional lift-slope, profile-drag, induced-drag, and pitching-moment
model. The separated model approaches flat-plate behavior at the flapped incidence and remains
defined for the entire `atan2` angle range, so a stalled aileron or elevator keeps the reduced
authority that post-stall and prop-hanging flight rely on. The flap's extra normal force acts on
the flap, aft of the quarter chord, in both regimes: the attached flap moment and lift increment
fix that arm at `-Cm_flap / (CLa tau)` chords, and the separated flap load keeps it,
`Cm_sep = -arm (CN(alpha_sep) - CN(alpha))`. Without it a stalled flying wing has only its
panels' lever arms about the centre of mass for pitch authority and a tailsitter cannot pitch
up out of forward flight. A continuous separation fraction blends the two:

```text
C = (1 - separation) C_attached + separation C_separated
```

The equilibrium separation fraction is a smooth function of the surface's frame angle of attack;
a flap does not shift the stall angle in this version. The actual fraction follows the
equilibrium with separate separation and reattachment time constants. This gives a compact,
differentiable hysteresis state and leaves room for a calibrated Goman-Khrabrov or learned
residual model without changing `AircraftState`.

## Whole-aircraft coefficient backend

`BodyModel` evaluates the classical polynomial form used by published small-UAV models from the
air velocity and body rates at the center of mass and adds its wrench to the component surfaces:

```text
C_L = CL0 + CLa a + CLq (c q / 2Va) + CLde de
C_D = CD0 + CDa a + CDa2 a^2 + CDb b + CDb2 b^2 + CDq (c q / 2Va) + CDde2 de^2
C_Y = CY0 + CYb b + CYp (b p / 2Va) + CYr (b r / 2Va) + CYda da + CYdr dr
C_l = Cl0 + Clb b + Clp (b p / 2Va) + Clr (b r / 2Va) + Clda da + Cldr dr
C_m = Cm0 + Cma a + Cmq (c q / 2Va) + Cmde de
C_n = Cn0 + Cnb b + Cnp (b p / 2Va) + Cnr (b r / 2Va) + Cnda da + Cndr dr
```

Forces are rotated from the wind frame with the standard wind-to-body rotation; moments are
formed directly in body axes with the reference span and chord. Rate terms are evaluated with
one power of airspeed fewer so the `1 / Va` of the non-dimensional rate never appears and the
block is finite at rest. The static angle-of-attack polynomials blend beyond `stall_angle` to a
flat plate with `normal_force_coefficient` and `pitch_flat_plate`, which keeps a published
low-angle model finite through the full envelope. Generalized aileron, elevator, and rudder
angles come from the physical actuators through `deflection_map[3, S]`, so the coefficient
table sees lagged, limited surfaces exactly as the component model does. An aircraft can be a
coefficient table alone, components alone, or both.

## Propulsion and propwash

Propellers have position, thrust direction, diameter, spin direction, a static torque
coefficient, a polynomial thrust map, and first-order motor dynamics. With `n` in revolutions
per second and `V_a` the axial inflow at the propeller:

```text
T / rho = D^4 sum_{i=1,2} sum_{j=0..2} c_ij n^i (V_a / D)^j      thrust map [2, 3]
Q = rho n^2 D^5 C_Q0
v_i (|V_a| + v_i) = T / (2 rho A)                                  momentum theory with inflow
```

The map contains the classical linear `C_T(J) = C_T0 (1 - J / J_0)` as `c_20 = C_T0`,
`c_11 = -C_T0 / J_0`, reproduces published exit-velocity laws such as the Skywalker X8's
exactly, and is linear in its coefficients for identification. Thrust scales with density, is
exactly zero for a stopped propeller by construction, and becomes windmilling drag where the map
goes negative. The induced velocity is the momentum-theory wake increment at the disk, computed
through the cancellation-free root so it is exactly zero at zero thrust, negative when
windmilling, and finite and differentiable everywhere; `validate_model` checks the map over the
whole shaft-speed range so the root's discriminant stays non-negative. The increment is
distributed to surfaces through `slipstream_map[P, S]`, whose weights are relative to the disk
value: surfaces in the developed wake see up to twice it.

This is deliberately an interface as much as a model. Future versions can add oblique inflow or
model wake contraction and skew while preserving the surface-flow calculation. A
`downwash_map[S, S]` weighting upstream lift into a local downwash would slot in at the same
point.

## Numerical policy

- Quaternion integration is normalized after every complete step.
- RK4 is the default; Euler exists for debugging and throughput comparisons.
- Singular aerodynamic divisions use explicit small speed scales.
- Physical state projection bounds separation fractions, actuator positions, and propeller speeds.
- Smooth actuator rate limiting uses `tanh`; unavoidable hardware bounds use clipping.
- The momentum-theory check on the thrust map keeps the induced-velocity root real.
- `rollout` holds one `Environment` per world by default and accepts a time-major sequence for
  gusts and moving air; both paths trace to the same program shape.
- The core contains no Python-side mutation and no hidden global state.

## API layers

```text
model construction / validation       host-side, ergonomic, not differentiated
                 ↓
pure dynamics and diagnostics          JAX arrays and PyTrees
                 ↓
integration and rollout                jit / grad / scan / vmap
                 ↓
trim, sweeps, local linearization       analysis over the same pure core
                 ↓
canonical boundary and plant adapter    frame conversion, stepped hidden plant
                 ↓
environments and controllers           future autonomy packages
                 ↓
rendering and hardware adapters         boundary coordinate conversion
```

## Roadmap

### Milestone 1 — physics kernel

- Batched 6-DoF rigid-body dynamics.
- Component surfaces with flapped and all-moving controls and full-envelope quasi-steady
  aerodynamics.
- Continuous separation dynamics.
- Advance-ratio propeller, momentum-theory slipstream, and actuator dynamics.
- Differentiable RK4 rollouts and an illustrative reference aircraft.
- Invariant, full-envelope finiteness, conservation, batching, JIT, and gradient tests.

### Milestone 2 — analysis and calibration

- Straight-flight trim and branch continuation across airspeed and angle of attack.
- Quaternion-safe discrete linearization and stability modes from automatic differentiation.
- Vectorized full-envelope aerodynamic coefficient sweeps.
- Versioned, named, unit-explicit aircraft specifications with TOML round-tripping.
- Whole-aircraft coefficient backend for published models; smooth-spline tables later.
- Canonical state boundary and a stepped plant for identification tooling.
- A published, physically identified reference airframe (Skywalker X8) validated against real
  flight through that boundary. The aerobatic fixture remains an illustrative software fixture.

### Milestone 3 — autonomy tooling

- Direct-actuator Gymnasium and native-JAX environments.
- Done: `cascade.env` — native-JAX episode functions (reset, step, policy rollout) over
  tracking, hover, and transition tasks with the control cascade and the transition controller
  as baseline policies, batched by vmap (75k to 130k env steps/s on a laptop CPU) and
  differentiable through the episode: a policy trained by gradient through the dynamics matches
  the tuned cascade in sixty steps. Sensor noise, bias, and delay, and domain randomisation
  over model batches. See `docs/environments.md`.
- Rate, attitude, airspeed, altitude, and path controllers.
- Done: `cascade.control` — a rate-scheduled rate/attitude/guidance cascade (PX4-style),
  differentiable and batchable, with a closed-loop rollout and tuned default controllers for both
  packaged aircraft. See `docs/control.md`.
- Domain randomization, gust models, and observation/sensor models.
- MPC and trajectory-optimization examples through stall and transition.

### Milestone 4 — world integration

- MuJoCo/MJX rendering, geometry, raycasting, and collision queries.
- Differentiable penalty contact for simple belly/skid interactions.
- Higher-fidelity landing gear only if takeoff and landing become a core use case.

### Milestone 5 — unsteady high-alpha fidelity

- Calibrated dynamic-stall model with convective time scaling.
- Measured propeller maps and oblique-inflow corrections.
- Propwash contraction/skew, surface occlusion, and wing downwash on downstream surfaces.
- Optional vortex-wake or learned-memory residual backend.

## Explicit non-goals

- Treating low-angle panel methods as post-stall ground truth.
- Claiming cross-airframe high-alpha transfer without new identification.
- Mixing aircraft with different static topologies inside a single compiled array.
- Coupling the physics kernel to a renderer, environment API, or flight stack.
- Hiding frame conventions or units behind implicit conversions.
