# Changelog

## 0.2.0 (2026-09-02)

Package layout by layer. The top-level `cascade` namespace is the core only (states, models,
specifications, dynamics, integration, trim and analysis, the canonical boundary, the stepped
plant, the packaged aircraft). The layers above moved into packages:

- `cascade.control`: `loops` (the rate/attitude/guidance cascade), `vtol` (hover and
  transition), `autotune`, and `tuned` (the packaged aircraft's controllers).
- `cascade.env`: `episode` (reset, step, rollouts), `tasks`, `sensors`, `baselines`,
  `weather`, `gusts`, `family`.
- `cascade.design`: `archetypes`.
- `cascade.viz`: `geometry`, `render`.

Compatibility shims keep the old module paths importable (`cascade.vtol`, `cascade.autotune`,
`cascade.weather`, `cascade.gusts`, `cascade.family`, `cascade.archetypes`,
`cascade.geometry`, `cascade.render`), and `cascade.env` keeps its public names. Renamed:
`cascade.env.Reference` is now `ReferenceFlight` (the old name is an alias). Moved:
`control_authority` is in `cascade.analysis`. Tasks own their reference speed and reference
flight (`task.reference_speed()`, `task.reference(model)`), so nothing branches on task type.
Added `py.typed`, a `slow` pytest marker, and a MuJoCo `viz` extra.

## 0.1.0

First pass: panel and coefficient aerodynamics, actuators, RK4 rollouts, trim and
continuation, linearisation, the Skywalker X8 validated against real flight through
Glassbox, the tailsitter fixture, the control cascade and transition controller, the episode
environment with sensors, weather, and families, archetypes with automatic tuning, and
MuJoCo rendering.
