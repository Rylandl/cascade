# Airframe archetypes and automatic tuning

`cascade.design.archetypes` turns a handful of design decisions into a full aircraft specification on
the panel backend, and `cascade.control.autotune` tunes the control cascade for any specification from
its own trim and linearisation. Together they produce a family of visibly different, flyable
airframes, each with a reference controller no human tuned, whose parameters a learner never
sees. The designs are plausible, not validated against real aircraft; their job is diversity.

## Designs

| archetype | design decisions |
| --- | --- |
| `FlyingWingDesign` | span, aspect ratio, wing loading, sweep, taper, washout, reflex, camber, static margin, elevon span and chord fractions, winglet area fraction, thrust-to-weight, propeller diameter fraction, motor layout (`pusher` or `twin_tractor`, the tailsitter's), pod mass fraction, cruise lift coefficient |
| `ConventionalDesign` | span, aspect ratio, wing loading, camber, dihedral, tail arm (chords), horizontal and vertical tail volume coefficients, static margin, aileron span and chord fractions, elevator and rudder chord fractions, tail arrangement (`conventional` or `v_tail`), thrust-to-weight, propeller diameter fraction, pod mass fraction, cruise lift coefficient |

`design_spec(design)` builds the `AircraftSpec` (`flying_wing_spec`, `conventional_spec`),
`cruise_speed(design)` gives the design speed from wing loading and cruise lift coefficient,
and `sample_design` / `sample_designs` draw designs uniformly from `FLYING_WING_RANGES` and
`CONVENTIONAL_RANGES` (override any range with a dict).

## Relations

Textbook, and stated in the code:

- Wing lift slope from aspect ratio and sweep (Helmbold); induced-drag factor from span
  efficiency 0.85.
- Plain-flap effectiveness and quarter-chord flap moment from chord fraction (thin airfoil,
  reduced for viscosity), so an elevon or aileron chord fraction sets both lift and moment.
- Sweep and taper place three panels per wing half along the quarter-chord line; washout sets
  each panel's incidence; reflex is the section zero-lift moment.
- Static margin places the centre of mass ahead of the neutral point of the panels (tail
  included, with its effectiveness reduced by the wing's downwash slope `2 CL_alpha / (pi AR)`).
  The tail itself keeps its full lift slope and sees the wing's downwash through the spec's
  `downwash_map`, so pitch damping and elevator power are not reduced along with the static
  stability (an earlier version folded the downwash into the slope; the nominal design's
  pitch authority rose 15% when the map replaced it).
- Tail volume coefficients and tail arm size the tails; a V-tail is two tilted panels carrying
  the horizontal and vertical volumes with a ruddervator mix in the control map.
- Propwash weights from how much of each panel the disk covers; the propeller's static thrust
  from thrust-to-weight and its pitch from cruise (zero-thrust airspeed at full speed is 1.6
  times cruise).
- Inertia from thin-plate panels, tails, motors, and a central pod, with a mass split.

## Validation

`validate_design(design)` trims at cruise, linearises, and returns a `DesignReport`. A design
passes when it trims within limits and 3° below stall, has pitch and roll authority above
15 rad/s² per unit channel (yaw above 3 with a rudder), and has no unstable mode faster than
a spiral. Authority is the Jacobian of angular acceleration with respect to the channels with
the surfaces at their steady deflection (`control_authority`), so actuator lag does not hide
it. From the default ranges about 9 in 10 sampled designs of either archetype pass; the rest
fail at trim or stall margin (flying wings) or at tail authority (conventional).

`scripts/archetype_statistics.py` emits these numbers. Sampled families are visibly diverse: across 40 designs per archetype the cruise speed spans
8 to 23 m/s, the short-period frequency 1 to 3.8 Hz, and pitch authority 23 to 540 rad/s².

## Automatic tuning

`tune_cascade(spec, cruise_speed)` trims, measures each axis's authority and rate damping
(from the linearised step), and places each rate loop at a bandwidth the actuators support
(`0.35 / lag`, at most 12 rad/s, yaw at half): `kp = (bandwidth - damping) / authority`,
`ki = kp · bandwidth`, feedforward half the damping over authority. Attitude gains sit at
bandwidth over 2.5. The airspeed gain comes from the measured acceleration per unit throttle,
the pitch ceiling keeps a climb at the rate limit 2° below stall, and channel signs come from
the sign of the measured authority, so a reversed control map tunes itself. `step_response`
flies a 0.5 rad heading step and a 5 m altitude step from the trim and reports whether the
cascade settles.

The tuner settles the two packaged aircraft and all four nominal archetypes within 1° of
heading and 0.15 m of altitude, and generalises to sampled designs (`tests/test_autotune.py`).
`examples/archetypes.py` prints a table of sampled designs, their reports, and their tuned
gains.

## Families

`cascade.env.family.sample_family(archetype, key, count)` draws valid designs, trims each at its
cruise, tunes a cascade for each, and stacks models, tasks, references, and controllers along
a family axis, so one `jax.vmap` over `cascade.env.reset`, `step`, or `rollout_policy` flies
the whole family, each member under its own baseline and, with `cascade.env.weather`, in its own
weather. Every flying wing has the same surface and propeller count whatever its layout
(winglets of zero area when unwanted, a pusher as two co-located halves), and every
conventional design likewise, which is what makes the stack possible. `family_member(family,
index)` unbatches one member.

The design parameters and reports stay on the `Family` as the hidden truth: an episode
exposes the channel count and the observation, nothing else. `examples/family_episode.py`
flies six designs of each archetype in random weather under their auto-tuned baselines and
prints, beside each return, the span, mass, cruise speed, wind, and tuned gain the policy
never saw.
