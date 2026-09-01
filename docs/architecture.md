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
boundaries rather than changing the physics convention.

## Static model and dynamic state

The numerical model is a JAX PyTree of arrays. Array shapes describe a fixed topology at compile
time:

```text
AircraftModel
├── mass and inertia
├── SurfaceModel[S]
├── PropellerModel[P]
└── ActuatorModel[S, P, C]

AircraftState
├── RigidBodyState        position, attitude, velocity, angular velocity
├── ActuatorState         S surface angles, P propeller speeds
└── AeroState             S continuous separation fractions

ControlInput
├── propeller[P]          normalized 0..1 commands
└── channel[C]            normalized -1..1 abstract surface channels
```

`ActuatorModel.surface_map[S, C]` maps named host-side channels such as aileron, elevator, and
rudder into physical surface angles. This supports elevons, V-tails, flaperons, differential
surfaces, and unconventional fixed-wing layouts without branching in the compiled dynamics.

Each compiled rollout has one static aircraft topology. Numerical model arrays can carry the same
leading world dimensions as state, allowing mass, inertia, aerodynamic coefficients, and actuator
parameters to vary per rollout. Different counts of surfaces, propellers, or channels still produce
separate compilations. This is an intentional tradeoff for predictable JAX performance.

## Component aerodynamic model

For surface `i`, local velocity through the air is

```text
v_i = R_bw (v_world - wind_world) + omega_body x r_i + v_slipstream_i
```

and is rotated into the possibly deflected surface frame. Each surface independently computes
airspeed, angle of attack, dynamic pressure, attached-flow coefficients, and separated-flow
coefficients. Forces are rotated back to the body frame and moments are accumulated at the center
of mass.

The attached model is a conventional lift-slope, profile-drag, induced-drag, and pitching-moment
model. The separated model approaches flat-plate behavior and remains defined for the entire
`atan2` angle range. A continuous separation fraction blends the two:

```text
C = (1 - separation) C_attached + separation C_separated
```

The equilibrium separation fraction is a smooth function of absolute angle of attack. The actual
fraction follows it with separate separation and reattachment time constants. This gives a compact,
differentiable hysteresis state and leaves room for a calibrated Goman-Khrabrov or learned residual
model without changing `AircraftState`.

## Propulsion and propwash

Propellers have position, thrust direction, disk area, spin direction, thrust/torque coefficients,
and first-order motor dynamics. A static momentum-theory induced velocity is distributed to
surfaces through `slipstream_map[P, S]`.

This is deliberately an interface as much as a model. Future versions can replace static induced
velocity with advance-ratio propeller maps, skewed wakes, or a vortex-particle wake while preserving
the surface-flow calculation.

## Numerical policy

- Quaternion integration is normalized after every complete step.
- RK4 is the default; Euler exists for debugging and throughput comparisons.
- Singular aerodynamic divisions use explicit small speed scales.
- Physical state projection bounds separation fractions, actuator positions, and propeller speeds.
- Smooth actuator rate limiting uses `tanh`; unavoidable hardware bounds use clipping.
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
environments and controllers           future autonomy packages
                 ↓
rendering and hardware adapters         boundary coordinate conversion
```

## Roadmap

### Milestone 1 — physics kernel

- Batched 6-DoF rigid-body dynamics.
- Component surfaces and full-envelope quasi-steady aerodynamics.
- Continuous separation dynamics.
- Propeller, slipstream, and actuator dynamics.
- Differentiable RK4 rollouts and an illustrative reference aircraft.
- Invariant, full-envelope finiteness, batching, JIT, and gradient tests.

### Milestone 2 — analysis and calibration

- Straight-flight trim and branch continuation across airspeed and angle of attack.
- Quaternion-safe discrete linearization and stability modes from automatic differentiation.
- Vectorized full-envelope aerodynamic coefficient sweeps.
- Versioned, named, unit-explicit aircraft specifications with TOML round-tripping.
- Coefficient-table and smooth-spline backends.
- Log schema, parameter estimation, uncertainty, and learned residual forces.
- A physically identified and validated reference-airframe dataset. The current packaged airframe
  remains an illustrative software fixture.

### Milestone 3 — autonomy tooling

- Direct-actuator Gymnasium and native-JAX environments.
- Rate, attitude, airspeed, altitude, and path controllers.
- Domain randomization and observation/sensor models.
- MPC and trajectory-optimization examples through stall.

### Milestone 4 — world integration

- MuJoCo/MJX rendering, geometry, raycasting, and collision queries.
- Differentiable penalty contact for simple belly/skid interactions.
- Higher-fidelity landing gear only if takeoff and landing become a core use case.

### Milestone 5 — unsteady high-alpha fidelity

- Calibrated dynamic-stall model with convective time scaling.
- Advance-ratio and oblique-inflow propeller maps.
- Propwash contraction/skew and surface occlusion.
- Optional vortex-wake or learned-memory residual backend.

## Explicit non-goals

- Treating low-angle panel methods as post-stall ground truth.
- Claiming cross-airframe high-alpha transfer without new identification.
- Mixing aircraft with different static topologies inside a single compiled array.
- Coupling the physics kernel to a renderer, environment API, or flight stack.
- Hiding frame conventions or units behind implicit conversions.
