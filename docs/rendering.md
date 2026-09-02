# Geometry and rendering

`cascade.geometry` draws an aircraft from its specification and `cascade.render` plays a
Cascade trajectory through MuJoCo to an MP4. Physics stays in JAX; MuJoCo only draws.

## Geometry from the spec

`aircraft_parts(spec)` returns the visual parts in the canonical FLU body frame (x forward,
y left, z up): every surface with area becomes a box from its position (the quarter-chord
reference), chord, area (so span width), and orientation matrix. Flapped surfaces split into a
fixed part and a flap hinged at 70% chord; all-moving surfaces hinge at their quarter chord;
positive deflection is trailing-edge down in both. Propellers are discs that spin about their
axis, with a motor pod. The fuselage the spec does not describe is an ellipsoid spanning the
drawn parts. A spec whose surfaces have no area (the coefficient-backend X8) gets a swept wing
outline from its reference span and chord.

`write_obj(spec, path)` writes the parts as a grouped OBJ mesh for other tools.
`mjcf_string(spec)` / `write_mjcf(spec, path)` write a MuJoCo model for kinematic playback: a
free body with the parts as geoms, hinged flaps and propellers as child bodies with joints,
a ground plane, sky, a directional light, a chase camera and a side camera on the body, and a
ground camera that tracks it. Gravity is off; Cascade sets the pose. The model also gives MJX
users a starting point for comparisons.

## Rendering

Install the `viz` extra (`uv sync --extra viz`, or `pip install cascade-flight[viz]`) and have
ffmpeg on the path. `render_trajectory(spec, trajectory, dt, path, fps=30, camera="chase")`
takes a time-major `AircraftState` from any rollout, sets the free joint from the NED/FRD
state through the canonical NWU/FLU conversion (MuJoCo's own convention), the flap joints from
the actuator deflections, spins the propellers, colours each panel from grey to red by its
separation state, and pipes frames to ffmpeg. Cameras: `chase` (behind and above, turning
with the aircraft), `side`, `follow` (behind and above in world axes, following the position
only, the right one for a tailsitter), and `ground` (a fixed point near the start that tracks it).

Two renders ship in `docs/media/`: [the tailsitter round trip](media/tailsitter_round_trip_follow.mp4)
and [a conventional archetype's heading and altitude steps](media/conventional_steps_chase.mp4).

`examples/render_flight.py` renders the tailsitter round trip (the wings go red through the
stalled transition and grey again in cruise) and a conventional archetype flying heading and
altitude steps under its auto-tuned cascade. `Scene` is the lower-level object for custom
frames: `pose(state, dt)` then `frame(camera)`.

Headless rendering needs an OpenGL context: on macOS the default works from a logged-in
session; on Linux set `MUJOCO_GL=egl` (GPU) or `osmesa` (software). The rendering tests skip
when no context is available.
