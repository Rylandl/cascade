# Flight-analysis workflow

Cascade's analysis tools call the same JAX dynamics used by rollouts. There is no reduced trim
model or separate low-angle aerodynamic path, so a state found by the analysis layer can be passed
directly to integration, differentiation, or control code.

## Straight-flight trim and continuation

`trim_straight_flight` holds airspeed, air-relative flight-path angle, and air-relative heading
fixed, then solves for roll, pitch, a yaw offset (the sideslip a rudderless or torque-loaded
aircraft needs to balance yaw), propeller commands, and control-channel commands. Wind changes
the resulting ground track. The solver enforces zero translational and
angular acceleration. Actuator positions, propeller speed, and separation fractions are set to
their equilibrium values before each residual evaluation.

```python
import cascade

model = cascade.aerobatic_reference()
condition = cascade.StraightFlightCondition(
    airspeed_m_s=12.0,
    flight_path_angle_rad=0.0,
    altitude_m=20.0,
)
trim = cascade.trim_straight_flight(model, condition)
assert trim.success, trim.message
```

The decision vector is ordered as roll, pitch, yaw offset, all propeller commands, then all
surface-control channels. `continue_trims` warm-starts each condition with the previous decision. This matters
because post-stall trim is non-convex: conventional and high-alpha equilibria can coexist, and the
initial seed selects a branch. Failed candidates are returned with their physical and normalized
balance residuals; solver termination alone is never reported as successful trim.

The included illustrative aircraft has a mathematical post-stall branch above 50 degrees when
seeded appropriately. That demonstrates that the numerical architecture can represent high-alpha
flight. It is not evidence that the forces, control margins, or stability of a real airframe match
the fixture.

## Aerodynamic sweeps

`aerodynamic_sweep` broadcasts angle of attack, sideslip, and airspeed into a single vectorized
evaluation. It reports body-axis force coefficients `[CX, CY, CZ]`, moment coefficients
`[Cl, Cm, Cn]`, physical wrenches, and each surface's static separation fraction. Moment
normalization uses reference span, chord, and span. Propeller slipstream affects surface loads, but
propeller thrust itself is excluded from aerodynamic coefficients.

Sweeps equilibrate the separation state at every point. They represent a static coefficient curve,
not a pitch-rate sweep with dynamic-stall hysteresis. Use a rollout to study transient separation.
Pass a `control` to sweep control derivatives; channels are linear coordinates in the units the
specification chose, so a published model with radian-valued channels can be checked directly
against its tables. `rollout` accepts a time-major `environments` sequence for gusts.

## Gusts

`cascade.env.gusts` generates Dryden continuous turbulence (MIL-F-8785C) as a time-major
`Environment` sequence: a first-order longitudinal filter and second-order lateral and vertical
filters driven by white noise, with time constants `L / V` from the length scales and the
aircraft's nominal airspeed, rotated from a chosen heading into world NED and added to the mean
wind. `dryden_low_altitude` gives the specification's intensities and length scales below about
300 m from the altitude and the 20 ft wind speed. Realizations are pure functions of a PRNG key
and broadcast over worlds, so every world in a batch can see its own gust history:

```python
import jax
from cascade.env.gusts import dryden_environment_sequence, dryden_low_altitude

parameters = dryden_low_altitude(altitude_m=50.0, wind_20ft_m_s=15.4)
environments = dryden_environment_sequence(
    jax.random.key(0), environment, steps=400, dt=0.01, airspeed_m_s=18.0, parameters=parameters
)
final, trajectory = cascade.rollout(model, state, controls, environment, 0.01, environments=environments)
```

Gusts are frozen-field wind histories, not a gust state coupled to the aircraft; a moving-air
model that varies with position would replace the sequence, not the dynamics.

## Quaternion-safe local linearization

`linearize_step` differentiates one complete integration step. Its state matrix uses 21 local
coordinates for the reference aircraft instead of differentiating a redundant four-component
quaternion:

```text
position[3], body-local attitude error[3], world velocity[3], body rates[3],
surface deflections[S], propeller speeds[P], separation fractions[S]
```

Inputs are ordered as propeller commands followed by control channels. The returned matrices are
discrete-time matrices for the requested timestep and integrator. `stability_modes` maps their
eigenvalues to continuous rates for interpretation. Position states naturally introduce neutral
modes; callers should select the dynamic subsystem appropriate to their analysis.

Run `uv run python examples/trim_envelope.py` for both trim branches, a local mode summary, and a
full-envelope finiteness sweep.
