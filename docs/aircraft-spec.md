# Aircraft specifications

An `AircraftSpec` is the named, versioned, host-side description of an airframe. Calling
`to_model()` validates it and compiles it into the anonymous array-only `AircraftModel` used by
JAX. This split keeps names, schema migration, and file I/O out of compiled simulation while
preserving them for calibration and tooling.

Specifications are TOML with an integer `schema_version`, currently `2`. Every dimensional field
includes its SI unit in its name, such as `mass_kg`, `position_m`, `stall_angle_rad`, or
`actuator_rate_limit_rad_s`; dimensionless coefficients carry none. Surface control maps follow
the ordered names in `controls.channels`; each propeller's slipstream weights and each row of the
body deflection map follow the ordered surface list.

```python
import cascade

spec = cascade.load_aircraft_spec("my_aircraft.toml")
model = spec.to_model()
cascade.save_aircraft_spec(spec, "normalized_aircraft.toml")
```

Validation rejects unsupported schema versions, duplicate names, mismatched map lengths,
non-physical scalar parameters, malformed rotations, non-unit thrust directions, invalid inertia
tensors, and static thrust beyond the momentum-theory bound. A schema-version change is required
for any incompatible semantic or structural change; adding calibrated values to an existing
field set does not change the schema.

## Top level

| Table | Fields |
|---|---|
| root | `schema_version`, `name`, `description` |
| `[rigid_body]` | `mass_kg`, `inertia_kg_m2` (3x3, symmetric positive definite, body FRD) |
| `[reference]` | `area_m2`, `chord_m`, `span_m` used by the body block and by sweeps |
| `[controls]` | `channels`: ordered channel names, the columns of every `control_map_rad` |

Control channels are linear coordinates in the units the control maps imply. A map entry of
`0.436` radians per unit channel makes the channel a normalized `[-1, 1]` stick; an entry of
`1.0` makes the channel itself a surface angle in radians, which is the right choice when flight
logs carry generalized surface angles. Physical limits apply to the mapped angle.

## `[[surfaces]]`

| Field | Unit | Meaning |
|---|---|---|
| `name` | | unique |
| `position_m` | m | aerodynamic center relative to the center of mass, body FRD |
| `body_from_surface` | | proper rotation with surface `x` along the chord and `y` along the span |
| `area_m2`, `chord_m` | m², m | area may be zero for a pure actuator surface |
| `lift_coefficient_zero`, `lift_curve_slope_rad` | 1, 1/rad | attached lift |
| `drag_coefficient_zero`, `induced_drag_factor` | 1 | attached drag `CD0 + k CL²` |
| `moment_coefficient_zero`, `moment_coefficient_alpha_rad` | 1, 1/rad | attached pitching moment about the surface position |
| `stall_angle_rad`, `stall_width_rad` | rad | separation equilibrium sigmoid |
| `normal_force_coefficient`, `edge_drag_coefficient`, `span_drag_coefficient` | 1 | separated flat-plate model and crossflow drag |
| `separation_time_constant_s`, `reattachment_time_constant_s` | s | separation lag |
| `all_moving_fraction` | 1 | `0` for a flap, `1` for a surface that rotates as a whole |
| `flap_effectiveness` | 1 | `d alpha_eff / d delta` for the flap share, typically 0.4–0.6 |
| `moment_coefficient_flap_rad` | 1/rad | intrinsic pitching-moment increment per flap radian |
| `drag_coefficient_flap_rad2` | 1/rad² | profile-drag increment per flap radian squared |
| `control_map_rad` | rad per channel unit | one entry per control channel |
| `actuator_bias_rad`, `actuator_limit_rad` | rad | neutral angle and symmetric physical limit |
| `actuator_time_constant_s`, `actuator_rate_limit_rad_s` | s, rad/s | first-order lag and smooth rate limit |

## `[[propellers]]`

| Field | Unit | Meaning |
|---|---|---|
| `name` | | unique |
| `position_m`, `direction_body` | m, unit vector | hub position and thrust axis, body FRD |
| `diameter_m` | m | disk diameter |
| `thrust_map` | 1 | 2x3 coefficients `c_ij` of `T / rho = D⁴ Σ c_ij n^(i+1) (V_a / D)^j`, rows for `n`, `n²` in rev/s, columns for `(V_a / D)^0..2` |
| `torque_coefficient_static` | 1 | `C_Q0 = Q / (rho n² D⁵)` |
| `spin_direction` | ±1 | sign of the reaction torque about the thrust axis |
| `slipstream_weights` | 1 | per surface, multiplying the disk induced velocity; developed-wake surfaces see up to 2 |
| `speed_min_rad_s`, `speed_max_rad_s` | rad/s | throttle `0..1` maps linearly between them |
| `time_constant_s`, `acceleration_limit_rad_s2` | s, rad/s² | motor lag and smooth acceleration limit |

The classical linear `C_T(J) = C_T0 (1 - J / J_0)` is the map `[[0, -C_T0 / J_0, 0],
[C_T0, 0, 0]]`; a published static thrust `T_s` at speed `omega_max` gives
`C_T0 = T_s (2 pi)² / (rho omega_max² D⁴)`, and geometric pitch over diameter is a good first
guess for `J_0`. Validation checks the map over the shaft-speed range so the momentum-theory
induced-velocity root stays real; for the linear law that reduces to `C_T0 <= (pi / 2) J_0²`.
`scripts/fit_x8_propeller.py` shows how a published exit-velocity law expands into the map.

## `[body]` (optional)

The whole-aircraft coefficient table in the classical polynomial form. It is added to the
component surfaces, so an aircraft described only by a published table uses zero-area surfaces to
carry its physical actuators.

| Table | Fields | Form |
|---|---|---|
| `[body.lift]`, `[body.pitch]` | `zero`, `alpha_rad`, `q`, `elevator_rad` | `C = zero + alpha·a + q·(c q / 2Va) + elevator·de` |
| `[body.drag]` | `zero`, `alpha_rad`, `alpha_sq_rad2`, `beta_rad`, `beta_sq_rad2`, `q`, `elevator_sq_rad2` | quadratic in `a`, `b`, and `de` |
| `[body.side]`, `[body.roll]`, `[body.yaw]` | `zero`, `beta_rad`, `p`, `r`, `aileron_rad`, `rudder_rad` | rates as `b p / 2Va`, `b r / 2Va` |
| `[body]` | `stall_angle_rad`, `stall_width_rad` | blend of the `alpha` polynomials to the flat plate |
| `[body]` | `normal_force_coefficient`, `pitch_flat_plate` | flat-plate `C_N` (about 2) and post-stall pitching moment |
| `[body]` | `deflection_map` | 3 rows (aileron, elevator, rudder) by `S` columns over physical surface angles |

Angles are radians, trailing-edge-down positive for every generalized control, matching the
Beard and McLain convention used by most published small-UAV models. For elevons on a flying
wing with left and right surfaces `L`, `R`: elevator `= (dL + dR) / 2`, aileron `= (dL - dR) / 2`.

## Provenance

The bundled `cascade-aerobatic-reference` file is deliberately labeled as a software fixture. A
physically identified specification should record, in its `description` and comments, the source
of every coefficient group, the reference point the moments are about, the mass and inertia
measurement, and any parameter the author chose between conflicting published values.
