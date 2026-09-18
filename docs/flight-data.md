# Flight recording packs and replay

`cascade.experiments` provides a recording format, frozen evaluation packs, and
open-loop replay against nominal and calibrated aircraft specifications. The package does
not ship a licensed measured-flight benchmark. The included example and regression fixtures
are explicitly synthetic; their errors check software behavior rather than physical accuracy.

## Prepare a recording

An upstream adapter must synchronize and resample the original log and convert its frames,
units, actuator conventions and attitude convention. No vendor field names or units are inferred.

```python
from cascade.experiments import FlightRecord

record = FlightRecord(
    name="pitch-up-window-1",
    maneuver_id="flight-03-pitch-up",
    time_s=timestamps,  # (T,), uniformly sampled, seconds
    canonical_state=states,  # (T, 13), described below
    command=commands,  # (T, propellers + control channels)
    kind="measured",  # default is "synthetic"
    wind_ned_m_s=wind,  # constant (3,) or interval-end (T, 3)
    density_kg_m3=density,  # positive scalar or interval-end (T,)
)
record.to_csv("prepared.csv")
```

The 13 state columns are canonical world position NWU, world velocity NWU, scalar-first
quaternion describing body FLU to world NWU, then body angular velocity FLU. All values use SI.
See [canonical state interchange](../src/cascade/canonical.py) for frame conversion helpers. In contrast,
the optional wind vector uses the simulator's **NED** world frame, as its field name states.
Commands use native propeller throttle fractions first, then the aircraft specification's
ordered control channels. Throttles must be in `[0, 1]`; control channels retain their native
meaning and scaling from the aircraft specification.

Row zero is the initial observed state. `command[i]`, `wind_ned_m_s[i]` and
`density_kg_m3[i]` apply throughout `(time_s[i-1], time_s[i]]`. Their row-zero values are
stored but unused during replay. Commands are held constant over each interval and any
integration substeps. Supply a constant wind vector or density scalar to broadcast it.
Omitted environment fields mean zero wind and density `1.225 kg/m³`; they are assumptions,
not estimates of the flight environment. Replay uses standard downward gravity.

Construction copies finite real arrays into read-only float64 storage, validates strictly
increasing uniform timestamps, and requires unit quaternions within `1e-4`. It does not
interpolate missing data or repair attitudes. Replay converts to the configured JAX precision
and rejects input overflow or timestep underflow before integration.

`FlightRecord.from_csv(path, name=..., maneuver_id=..., kind="measured")` accepts these exact
headers, in any order:

| Columns | Meaning |
| --- | --- |
| `time_s` | Timestamp in seconds. |
| `state_0` through `state_12` | The canonical state components above. |
| `command_0` through `command_N` | Consecutive zero-based native input columns. |
| `wind_ned_x_m_s`, `wind_ned_y_m_s`, `wind_ned_z_m_s` | Optional; all three are required together. |
| `density_kg_m3` | Optional positive density. |

Duplicate, missing, ambiguous or extra columns are rejected. `to_csv` writes all environment
columns and enough decimal digits to round-trip the stored arrays. This is a prepared interchange
format; importing a raw autopilot CSV still requires an explicit adapter.

## Freeze the splits

```python
from cascade.experiments import create_flight_pack, load_flight_pack

manifest = create_flight_pack(
    "dist/flight-pack",
    nominal_spec,
    {
        "fitting": fitting_records,
        "validation": validation_records,
        "evaluation": evaluation_records,
    },
    license="CC-BY-4.0",  # the actual recording license
    source="recording source and acquisition protocol",
    description="maneuver selection and preprocessing version",
)
nominal_spec, split_records, metadata = load_flight_pack(manifest)
```

The output directory must be absent or empty. The pack contains `aircraft.toml`, one NPZ per
record, and `manifest.json`. Fitting and evaluation splits must both be nonempty; validation
is optional. Record names must be unique. Windows from the same source maneuver must share
a `maneuver_id` and cannot cross splits. Exact duplicate replay content is also rejected across
splits, even with different labels, shifted time origins, antipodal quaternions or changes to
unused row-zero inputs.

The manifest hashes the aircraft specification, recording files and normalized replay content.
Loading rechecks those hashes, licensing/source declarations, split rules, shapes and command
contracts. NPZ loading disables pickle, and both recording and aircraft paths must remain
inside the pack directory, including when symlinks are present. Hashes establish content
identity; they do not authenticate a publisher or prove redistribution rights. A nonempty
license declaration records the caller's claim, so retain the original grant separately.

The duplicate check is not an overlap detector: overlapping windows with different lengths
or preprocessing can have different hashes. Group all related windows under the same maneuver
ID before splitting, and freeze preprocessing and parameter-selection rules before evaluating.

## Evaluate nominal and calibrated models

```python
from cascade import spec_hash
from cascade.experiments import evaluate_flight_pack
from cascade.experiments.manifest import content_hash

result = evaluate_flight_pack(
    manifest,
    calibrated=fitted_spec,
    calibration={
        "pack_sha256": content_hash(metadata),
        "calibrated_spec_sha256": spec_hash(fitted_spec),
        "fitting_records": [record.name for record in fitting_records],
        "method": "description of the independent fitting procedure",
    },
    substeps=4,
    output="dist/replay-results.json",
)
```

Omit `calibrated` and `calibration` to evaluate only nominal and persistence baselines. A supplied
calibrated specification requires metadata binding that exact specification to the exact pack
and a nonempty, unique list of fitting-only record names. Channel ordering and surface/propeller
names and ordering must match the nominal specification. Use [aircraft calibration](calibration.md)
to estimate bounded parameters from the fitting split and produce this provenance automatically,
or supply an externally fitted model. This evaluator does not optimize or select a best model.
It checks the declaration but cannot
prove that fitting avoided evaluation data. A previously inspected evaluation set does not
become a blind holdout by repacking it.

Replay starts from each maneuver's first measured rigid-body state, initializes unknown actuator
and separation states at equilibrium under its first active command and environment, then runs
open-loop without resetting to later observations. Recorded wind and density are held at each
interval-end value. Increase `substeps` and check numerical convergence for the chosen aircraft
and sampling period. Equilibrium initialization can bias early transient errors when the real
initial actuator state is unknown; document warm-up and exclusion rules upstream.

Each recording gets position, velocity, quaternion angular-distance and angular-rate RMSE values.
Vector errors use Euclidean magnitudes; the initial sample is excluded. Quaternion error treats
`q` and `-q` as the same attitude. The persistence baseline holds all 13 initial components
constant, including position; it is a deliberately simple baseline, not dead reckoning.
Results include model hashes, pack hash, calibration declaration, integration substeps,
initialization/environment assumptions, runtime provenance and the observed evidence kinds.
`split="evaluation"` is the default; fitting or validation scores can be requested explicitly.

## Reproduce the synthetic workflow

```sh
python examples/flight_data_evaluation.py --output dist/synthetic-flight-pack
```

The example generates two separate simulator maneuvers with a known mass perturbation, freezes
fitting/evaluation splits, and compares the nominal specification with that known generator.
It declares the comparison as **known synthetic generator, not fitted**. Its assertions show that
the replay pipeline recovers simulated motion more closely with the generating model; they do
not demonstrate calibration performance or measured-flight predictive accuracy.

For an actual fitting demonstration on synthetic data, run
`python examples/calibrate_aircraft.py --output dist/synthetic-calibration`. That example
estimates mass and inertia from fitting maneuvers before scoring separate recordings.

`tests/test_flight_data.py` also verifies replay with nonzero, changing wind and density, CSV
round trips, tamper detection, split guards and rejected calibration provenance.
