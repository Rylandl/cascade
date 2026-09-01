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
6.5 m/s (alpha 3° at 9 m/s to 10° at 6.5 m/s). The thrust-borne branch continues from near
hover all the way up through cruise:

| airspeed m/s | 0.5 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pitch ° | 67 | 58 | 51 | 58 | 66 | 63 | 56 | 48 | 39 |
| alpha ° | 67 | 58 | 51 | 58 | 64 | 56 | 44 | 34 | 26 |
| throttle | 0.77 | 0.74 | 0.73 | 0.79 | 0.82 | 0.85 | 0.88 | 0.91 | 0.94 |
| elevator (normalized) | 0.01 | −0.04 | −0.14 | −0.45 | −1.00 | −0.84 | −0.61 | −0.47 | −0.36 |

Two features matter for a transition controller. The branches coexist above 6.5 m/s with very
different incidence, so a transition is a change of branch, not a slide along one. And around
3 to 4.5 m/s the thrust-borne branch needs nearly full nose-up elevon: that is the
control-authority pinch where the wings are fully separated but the propwash over the inboard
elevons is the only pitch authority. A schedule that lingers there stalls out of authority.

The non-monotonic pitch (a minimum near 2 m/s) comes from the trade between propwash lift on
the inboard panels, which supports weight at low speed, and post-stall wing lift, which grows
with airspeed; both are outputs of the component model, not tuned in.
