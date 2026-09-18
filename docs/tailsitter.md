# The tailsitter reference

`cascade.tailsitter_reference()` is an illustrative indoor-class twin-motor flying-wing
tailsitter: 100 g, 0.5 m span, two counter-rotating 0.1 m propellers on the leading edge, elevons
spanning the trailing edge, tip winglets. It is a software fixture built to exercise the regime
Cascade exists for, hover-to-cruise transition through post-stall flight, not an identified
vehicle. Its numbers are plausible for a 1S micro airframe.

The corridor and calm-air tables below were reproduced with Python 3.13.11, JAX 0.11.1,
float32, and the CPU backend on macOS arm64. Run `python examples/tailsitter_corridor.py`,
`python examples/tailsitter_transition.py`, and `python examples/tailsitter_tuning.py` to
reproduce them. These checks concern this fixed simulated fixture; they do not establish
hardware flight performance. See [validation status](validation.md).

## What makes it a tailsitter in the model

- Each propeller's wake is mapped onto its own inboard wing panel (far-wake weight 1.8) and
  grazes its winglet; the outboard panels are clean. A 0.1 m propeller washes about 40% of a
  0.25 m half-span. Wake coverage is a modeling choice with a strong effect on hover loads.
- The elevons are flaps on every wing panel, so in hover the washed inboard panels give pitch
  and roll authority at zero airspeed: about half deflection yields several rad/s² about both
  axes from propwash alone.
- Differential thrust yaws in body axes; the symmetric counter-rotating pair cancels reaction
  torque when the shaft speeds are equal.
- Hover needs about 78% throttle; full throttle gives a thrust-to-weight ratio of 1.6.

## The steady transition corridor

![Trim corridor: thrust-borne and conventional branches](figures/tailsitter_corridor.svg)

`examples/tailsitter_corridor.py` traces the two straight-flight branches with
`continue_trims`. With its fixed seed and speed grid, conventional trims pass at 7, 8, and
9 m/s (alpha 9.2°, 5.8°, and 4.3°); the 6 and 6.5 m/s candidates fail the balance check.
That search does not establish an exact stall boundary or exclude other equilibria. The
thrust-borne branch continues through the following near-hover to cruise points, with roll
and sideslip close to zero:

| airspeed m/s | 0.5 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pitch = alpha ° | 72 | 62 | 53 | 52 | 47 | 40 | 34 | 28 | 22 |
| throttle | 0.78 | 0.76 | 0.73 | 0.77 | 0.78 | 0.79 | 0.81 | 0.83 | 0.85 |
| elevator (normalized) | −0.05 | −0.10 | −0.18 | −0.26 | −0.30 | −0.29 | −0.26 | −0.24 | −0.20 |

Two features matter for a transition controller. The sampled branches coexist at 7–8 m/s with very
different incidence, so a transition is a change of branch, not a slide along one. And between
3 and 5 m/s the thrust-borne branch needs the most nose-up elevon: that is the
control-authority pinch where the wings are fully separated and the propwash over the inboard
elevons carries the pitch authority.

The post-stall flap load has a moment arm inferred from the attached flap lift and moment
coefficients ([architecture](architecture.md)). This supplies pitch authority after separation.
The corridor checks exercise that implementation but do not independently validate its
post-stall force or moment law.

## Hover and transition under closed-loop control

![Round trip in calm air and in gusts](figures/tailsitter_round_trip.svg)

`cascade.control.vtol` flies the fixture with the loops in `cascade.control`: hover guidance turns a
position and velocity error into a thrust axis and throttle (with an integral so the wing's
camber lift in its own propwash leaves no standing offset, a wing-lift credit so a fast, tilted
wing is not pushed forward by thrust it no longer needs, and a position-error clip so the loop
tracks velocity rather than lunging at a stale position), the attitude and rate loops track
that axis, and a transition is a scheduled forward tilt proportional to the commanded speed.
The forward-flight guidance blends in when both the measured airspeed and the commanded speed
are above the switch, so decelerating the schedule hands the aircraft back to hover guidance
while it is still fast: the back-transition is a pitch-up onto the thrust-borne branch that
lets drag do the braking. `examples/tailsitter_transition.py` flies the round trip, hold 2 s,
3.5 m/s² to 8 m/s, 3 s of cruise, 2 m/s² back to hover:

| time s | 2 | 3 | 4 | 5 | 7 | 8 | 9 | 10 | 11 | 12 | 15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| commanded m/s | 0 | 3.5 | 7.0 | 8.0 | 8.0 | 6.6 | 4.6 | 2.6 | 0.6 | 0 | 0 |
| airspeed m/s | 0.1 | 2.4 | 6.0 | 7.9 | 7.9 | 7.3 | 5.6 | 3.9 | 1.1 | 0.2 | 0.1 |
| tilt from vertical ° | 2.5 | 48 | 74 | 88 | 85 | 79 | 52 | 35 | 4 | 1.4 | 0.4 |
| forward weight | 0 | 0 | 0.20 | 0.90 | 0.90 | 0.45 | 0 | 0 | 0 | 0 | 0 |
| throttle | 0.80 | 0.88 | 0.84 | 0.62 | 0.62 | 0.48 | 0.72 | 0.74 | 0.80 | 0.81 | 0.80 |
| altitude m | 1.34 | 1.67 | 2.69 | 2.65 | 1.92 | 1.81 | 1.37 | 1.11 | 1.02 | 1.11 | 1.50 |

The example also differentiates final position error through the round trip; the recorded
derivative with respect to deceleration is approximately −0.0301. The schedule and controller
can be optimized through the rollout, subject to the model and numerical limitations.

`tests/test_transition.py` also exercises a 90° heading ramp during cruise and return to
hover. The forward rate setpoint includes coordinated-turn pitch and yaw rates for the
commanded bank (`cascade.control.coordinated_turn_rates`), and the final hover azimuth follows
the commanded heading. The test checks a track error below 5° at the end of cruise and a final
hover position error below 1 m; those are fixture-specific regression thresholds.

## Tuning the schedule by gradient

`examples/tailsitter_tuning.py` differentiates a cost over the whole 16 s round trip (mean
squared altitude error, final position and speed error, elevon effort) with respect to the
schedule's acceleration, deceleration, and cruise tilt, and takes a dozen bounded gradient
steps. In the recorded run, cost falls from 0.404 at the initial evaluated schedule
`[3.50, 2.00, 1.00]` to 0.284 at iteration 11's evaluated schedule `[4.04, 1.67, 1.16]`
(acceleration in m/s², deceleration in m/s², tilt in radians). The final printed schedule
includes one more update and is not evaluated in the loop. The altitude term contributes
most of the improvement. This demonstrates differentiation through this plant/controller
example; it does not establish a global optimum or a portable runtime benchmark.

## Wind and gusts

`transition_rollout(..., environments=)` takes a time-major environment, allowing Dryden
histories from `cascade.env.gusts` to run through both transitions. The fixture maps body-z
rate commands to differential thrust and offers `hover_azimuth_across_wind` to orient the
wing relative to the wind. These are model/control strategies, not verified operating limits
for a physical tailsitter.

The fixed cases in `tests/test_transition.py` include a 1 m/s spanwise wind and a seeded
Dryden round trip. They check finite motion, bounded position/altitude errors, and return to
low speed. Passing those cases does not characterize success probability across turbulence
realizations or establish a maximum safe wind speed.

The stored figures are illustrations generated by `scripts/plot_tailsitter.py`
(`uv run --with matplotlib python scripts/plot_tailsitter.py`); they are not flight measurements.
