# Research workflows for 0.4

The verified 0.3.0rc1 distributions and evidence in `dist/` are preserved. New work targets
0.4.0.dev0 and does not inherit the earlier candidate's release qualification.

## Implementation and acceptance

- [x] Frozen experiment manifests: aircraft specifications, tasks, weather, faults, sensor
  settings, disjoint seeds/splits; fair policy comparisons, metrics and saved trajectories.
- [x] Standalone HTML trajectory inspector: synchronized plots, applied versus commanded
  actuation, fault markers, stall state and flight comparison, no visualization dependency.
- [x] Mission tasks: scheduled tracking, timed waypoints, orbit following; observations,
  reward and baseline target the same reference at the same time.
- [x] Sensor pipeline: independent rates, drift, dropout, jitter and delay; ages/validity,
  deterministic seeds, JIT/vmap and backward-compatible defaults.
- [x] Envelope tools: verified steady-turn trim and interpolated controller gain schedules.
- [x] Optional public Gymnasium adapter with checker coverage and examples.
- [x] Flight-data evaluation packs: explicit licensing/source metadata, content hashes,
  frozen maneuver splits, nominal/calibrated replay metrics and reproducible example.
  Synthetic fixtures must be labeled; actual flight-accuracy claims require suitable data.
- [x] Documentation, examples, focused tests, complete regression suite, lint, package build
  and clean-install smoke checks for the new feature version.

No package publication, remote tagging or merging is part of this task.

See [feature-readiness](../docs/feature-readiness.md) for the executed verification record,
artifact locations and remaining limits. No measured-flight dataset was supplied; the
evaluation workflow is implemented and the included pack is explicitly synthetic.
