# Review: small-UAS manufacturer VP Engineering (Opus role-play, 2026-09-02)

## 1. Would we adopt it this quarter, and for what

Yes, narrowly: two engineers, one workflow, desk screening of airframe and payload variants before build or flight. Trim, continuation, linearisation, and control authority answer per CG/mass/payload case whether it trims at survey speed, the stall margin, short-period damping, and remaining authority; tune_cascade gives a first-cut gain shape and flags variants that will be painful to tune; weather gives a qualitative wind screen. A 500-case sweep is an overnight job on a laptop.

Not this quarter: fewer flight-test hours, envelope sign-off, customer qualification evidence.

## 2. What blocks adoption, ranked

1. Integration, the killer: no PX4 or ArduPilot surface. "PX4-style" means structural resemblance, not FW_RR_P/FW_PR_*/TECS parameters. No parameter export, no ULog/DataFlash ingestion, no MAVLink or SITL/HIL bridge. Plant is the closest thing but single-world, un-paced, no bus or lockstep.
2. Fidelity for our class: the one validated airframe scores 0.677 to 0.735 of persistence, only after a 50 mm CG shift and a halved wind estimate. No ground contact, gear, takeoff/landing; no downwash; static slipstream; no unsteady stall; no battery/energy model, so no endurance or transition-energy answer.
3. Workflow gaps: three decoupled SISO loops, not TECS; no wind feedforward; no attitude integrator; no waypoint/path following (L1/NPFG); VTOL is tailsitter only (hover_throttle splits thrust equally, no allocation/mixer for quadplane lift rotors, which is most of what we field); no failure modes (motor-out, servo jam), which is what qualification consumes.
4. Skills: JAX pytrees, vmap, one static topology per compilation; a hand-authored spec wants about 25 coefficients per surface we do not have; archetypes are plausible, not validated.
5. Support/licensing: MIT, no legal risk; alpha, 48 commits, one author, shims already needed at 0.2; the risk is key-man.

## 3. Unserved needs

| Need | Business impact | Done looks like | Effort |
| --- | --- | --- | --- |
| Autopilot parameter export | 1 to 2 tuning sorties saved per variant, about 15 variants per year | tune_cascade result to a PX4/ArduPlane param file with a documented gain mapping and back-check | 4 to 8 weeks |
| Ingest our ULog/BIN and fit the airframe | The only thing that earns engineering trust | from_ulog to windows to fitted spec to score vs persistence, per tail number | 6 to 10 weeks |
| Quadplane lift+cruise: rotor allocation and transition | Covers the product line | Multi-rotor mixer, hover/transition on a 4+1 config | 6 to 12 weeks |
| Real-time HIL bridge over Plant | Rig reuse, controller in the loop | Lockstep MAVLink HIL, wall-clock paced | 3 to 6 weeks |
| Wind-limit report per airframe | Customer-facing operating limits | Statistical max wind for a survey mission | 2 to 4 weeks |
| Failure-mode envelope | Qualification evidence | Motor-out/servo-jam sweeps over model perturbation | 3 to 4 weeks |

## 4. Risks for the CEO

Bus factor of one on an alpha library with churning module paths: pin a commit and budget to vendor it. Accuracy overclaim liability if sim envelope numbers are quoted to a defence customer; today the evidence is one airframe below persistence. The real cost is 3 to 6 engineer-months building the bridge, against simply buying more flight hours.

## 5. The one feature that gets it onto a real airframe

A closed loop from our flight log to autopilot parameters: ingest a ULog, fit the spec, report the validation score, emit a PX4/ArduPlane gain set that flies on the first sortie. If only half, the log-fit half.
