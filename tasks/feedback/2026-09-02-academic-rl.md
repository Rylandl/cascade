# Review: academic RL/control researcher (Opus role-play, 2026-09-02)

## 1. Who I am and what I'd use this for

I run a learning-based aerial-control lab: RL, differentiable sim, sim-to-real, mostly quadrotors, publishing at CoRL/ICRA/RSS. I want fixed-wing because reviewers keep asking whether our results transfer off a quad. I'd use Cascade for two things: a morphology-generalization benchmark that isn't "quads with different masses," and a differentiable-sim-vs-model-free comparison on a plant with genuinely nasty dynamics (stall, transition).

## 2. Fit today

Enough to start a paper this semester.

- `src/cascade/env/episode.py` is a real vectorized env: pure reset/step/rollout_policy, vmap over keys and over models, with sensor white noise, per-episode rate bias, and observation delay (`src/cascade/env/sensors.py`). A domain-randomization ablation table out of the box.
- `cascade.env.family.sample_family` + `src/cascade/design/archetypes.py` is the differentiator versus Crazyflow and gym-pybullet-drones: procedurally sampled airframes, each auto-tuned, design parameters held off the observation. A ready-made RMA / morphology-generalization suite.
- A non-strawman baseline: `cascade_policy` scores a hand-tuned cascade on the same task, resets, and horizon.
- The BPTT result is half-written: 60 gradient steps through the physics match the tuned cascade (156.8 vs 156.5).
- The tailsitter transition is a task quads structurally cannot pose, differentiable through both transitions.

## 3. Unserved needs, ranked

1. GPU throughput evidence. All numbers are Apple M3 CPU; no GPU row, CPU-only CI. Done: an A100/H100 table at batch 2^14 to 2^16 plus a GPU CI job. Days.
2. No RL-library glue. No Gymnasium wrapper shipped, no Brax/PureJaxRL adapter, no PPO/SAC reference run. Done: a Brax-style Env shim plus one seeded PPO script with learning curves. 1 to 2 weeks.
3. Sim-to-real is not in this repo. The real-flight validation lives in Glassbox; no logs, no replay script, no PX4/MAVLink or ROS bridge; `plant.py` is HIL-shaped with nothing on the other end. Done: public NTNU logs plus `scripts/replay_flight.py` reproducing the persistence-ratio number. 2 to 3 weeks.
4. No ground contact, takeoff, or landing; termination is only crash altitude. Done: differentiable penalty contact plus a landing task. 2 to 4 weeks.
5. Quasi-steady aero: separation is a first-order lag, not Goman-Khrabrov; no downwash map, no wake skew. Undercuts the post-stall claim. Months; research-grade.
6. No reproducible benchmark suite: doc tables are prose, not a seeded script. About a week.

## 4. Red flags

- Age and bus factor: 48 commits over two days, one author, no PyPI release, no DOI, no CITATION.cff or CONTRIBUTING; a 0.1 to 0.2 reshuffle already.
- The validation number is modest and off-repo: 0.68 of persistence, 1.33x the fitted model; README and docs quote slightly different values (0.68 vs 0.677/0.675).
- Archetypes are "plausible, not validated": a generalization result on them is a claim about Cascade's generative model, not about aircraft.
- float32 hardcoded in `env/weather.py` and `hover_task`; x64 untested.
- Docs drifting: `docs/control.md` points at `cascade/control.py`; `env/baselines.py` imports via the deprecated `cascade.vtol` shim.
- Licensing is clean (MIT); per-coefficient citations in `skywalker_x8.toml` are good provenance.

## 5. One thing I'd ask for first

A single reproducible benchmark script that emits GPU throughput and a seeded PPO-vs-BPTT learning curve on the X8 tracking task.
