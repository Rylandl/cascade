# Airframe archetypes and automatic tuning

`cascade.design.archetypes` turns a handful of design decisions into a full aircraft specification on
the panel backend, and `cascade.control.autotune` tunes the control cascade for any specification from
its own trim and linearisation. Together they generate varied simulated airframes and reference
controllers, using the design rules and tuning heuristics described below. The designs are
plausible software fixtures, not validated against real aircraft; their job is diversity.

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
  stability.
- Tail volume coefficients and tail arm size the tails; a V-tail is two tilted panels carrying
  the horizontal and vertical volumes with a ruddervator mix in the control map.
- Propwash weights from how much of each panel the disk covers; the propeller's static thrust
  from thrust-to-weight and its pitch from cruise (zero-thrust airspeed at full speed is 1.6
  times cruise).
- Every panel's separated load acts a quarter chord aft of its reference point (the flat-plate
  centre-of-pressure march), so an archetype has a stall pitch break; the packaged fixtures
  leave that at zero.
- Inertia from thin-plate panels, tails, motors, and a central pod, with a mass split.

## Validation

`validate_design(design)` trims at cruise, linearises, and returns a `DesignReport`. A design
passes when it trims within limits and 3° below stall, has pitch and roll authority of at least
15 rad/s² per unit channel (yaw at least 3 with a rudder), and has no unstable mode with a time
constant under 2 s. Authority is the Jacobian of angular acceleration with respect to the channels,
with the surfaces at their steady deflection (`control_authority`), so actuator lag does not hide
it. These are internal screening criteria, not certification or evidence of real-aircraft stability.

`python scripts/archetype_statistics.py 40` uses key 0 and the default ranges. The recorded
Python 3.13.11 / JAX 0.11.1 CPU float32 run produced the following ranges among passing designs:

| archetype | pass count | cruise m/s | short-period Hz | pitch authority rad/s² per unit channel |
| --- | ---: | ---: | ---: | ---: |
| flying wing | 36/40 | 7.90–20.24 | 1.13–3.36 | 90.23–479.09 |
| conventional | 38/40 | 7.95–23.00 | 0.86–2.83 | 27.85–240.11 |

Flying-wing rejection reasons included trim, stall margin, and a fast unstable mode;
conventional rejections involved tail authority. A design can have multiple reasons. These
results characterize this finite seeded sample; changing ranges or seeds changes the population.

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

`tests/test_autotune.py` exercises the aerobatic and X8 fixtures plus four nominal layouts,
then a fixed sampled set. Its settling criterion is heading error below 5°, altitude error
below 1.5 m, and airspeed error below 2 m/s after the default 12 s maneuver; the sampled test
requires at least 80% of the screened cases to meet that criterion. This finite regression
coverage does not promise successful tuning for every generated design.
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
