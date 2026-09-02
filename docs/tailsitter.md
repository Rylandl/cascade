# The tailsitter reference

`cascade.tailsitter_reference()` is an illustrative indoor-class twin-motor flying-wing
tailsitter: 100 g, 0.5 m span, two counter-rotating 0.1 m propellers on the leading edge, elevons
spanning the trailing edge, tip winglets. It is a software fixture built to exercise the regime
Cascade exists for, hover-to-cruise transition through post-stall flight, not an identified
vehicle. Its numbers are plausible for a 1S micro airframe.

## What makes it a tailsitter in the model

- Each propeller's wake is mapped onto its own inboard wing panel (far-wake weight 1.8) and
  grazes its winglet; the outboard panels are clean. A 0.1 m propeller washes about 40% of a
  0.25 m half-span, and applying the wake to the whole half would let propwash lift carry most
  of the weight and trim the near-hover state at only 44° of pitch.
- The elevons are flaps on every wing panel, so in hover the washed inboard panels give pitch
  and roll authority at zero airspeed: about half deflection yields several rad/s² about both
  axes from propwash alone.
- Differential thrust yaws in body axes; the counter-rotating pair leaves no net reaction torque.
- Hover needs about 78% throttle; full throttle gives a thrust-to-weight ratio of 1.6.

## The steady transition corridor

`examples/tailsitter_corridor.py` traces the two straight-flight branches with
`continue_trims`. The conventional branch exists above the fixture's stall speed of about
7 m/s (alpha 4° at 9 m/s to 10° at 7 m/s). The thrust-borne branch continues from near hover
all the way up through cruise:

| airspeed m/s | 0.5 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pitch ° | 72 | 62 | 54 | 61 | 59 | 56 | 50 | 42 |
| alpha ° | 72 | 62 | 54 | 61 | 55 | 46 | 35 | 26 |
| throttle | 0.78 | 0.76 | 0.74 | 0.79 | 0.81 | 0.84 | 0.86 | 0.88 |
| elevator (normalized) | −0.05 | −0.10 | −0.20 | −0.54 | −0.51 | −0.45 | −0.35 | −0.28 |

Two features matter for a transition controller. The branches coexist above 7 m/s with very
different incidence, so a transition is a change of branch, not a slide along one. And between
3 and 4.5 m/s the thrust-borne branch needs about half nose-up elevon: that is the
control-authority pinch where the wings are fully separated but the propwash over the inboard
elevons is the only pitch authority. (With a reflexed wing section, a positive camber moment,
the pinch reached full deflection; the fixture uses a symmetric section so that hover needs no
standing elevon trim.)

The non-monotonic pitch (a minimum near 2 m/s) comes from the trade between propwash lift on
the inboard panels, which supports weight at low speed, and post-stall wing lift, which grows
with airspeed; both are outputs of the component model, not tuned in.

## Hover and transition under closed-loop control

`cascade.vtol` flies the fixture with the loops in `cascade.control`: hover guidance turns a
position and velocity error into a thrust axis and throttle (with an integral so the wing's
camber lift in its own propwash leaves no standing offset), the attitude and rate loops track
that axis, and a transition is a scheduled forward tilt at high throttle with the forward-flight
guidance blending in above the switch airspeed. `examples/tailsitter_transition.py` holds hover
for two seconds, ramps to 7 m/s at 3.5 m/s², and cruises:

| time s | 2 | 3 | 4 | 5 | 7 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| airspeed m/s | 0.1 | 2.7 | 7.0 | 7.7 | 7.3 | 7.1 |
| tilt from vertical ° | 2.5 | 50 | 82 | 88 | 82 | 81 |
| forward weight | 0 | 0 | 0.73 | 0.91 | 0.82 | 0.77 |
| throttle | 0.80 | 0.93 | 0.70 | 0.55 | 0.59 | 0.61 |
| altitude m | 1.34 | 1.71 | 3.04 | 2.62 | 1.89 | 1.98 |

The whole rollout is one differentiable program, so the ramp acceleration or any gain can be
tuned by gradient through the transition.
