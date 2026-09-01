# Aircraft specifications

An `AircraftSpec` is the named, versioned, host-side description of an airframe. Calling
`to_model()` validates it and compiles it into the anonymous array-only `AircraftModel` used by
JAX. This split keeps names, schema migration, and file I/O out of compiled simulation while
preserving them for calibration and tooling.

Specifications are TOML with an integer `schema_version`. Every dimensional field includes its SI
unit in its name, such as `mass_kg`, `position_m`, `stall_angle_rad`, or
`actuator_rate_limit_rad_s`. Surface control maps follow the ordered names in
`controls.channels`; each propeller's slipstream weights follow the ordered surface list.

```python
import cascade

spec = cascade.load_aircraft_spec("my_aircraft.toml")
model = spec.to_model()
cascade.save_aircraft_spec(spec, "normalized_aircraft.toml")
```

Validation rejects unsupported schema versions, duplicate names, mismatched map lengths,
non-physical scalar parameters, malformed rotations, non-unit thrust directions, and invalid
inertia tensors. A schema-version change is required for any incompatible semantic or structural
change; adding calibrated values to an existing field set does not change the schema.

The bundled `cascade-aerobatic-reference` file is deliberately labeled as a software fixture. A
future physically validated reference dataset should retain raw source provenance, identified
parameter uncertainty, validation maneuvers, and the exact fitting code alongside this compiled
specification.
