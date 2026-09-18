"""Exercise an installed distribution, from outside the source checkout.

Run with the clean environment's Python, e.g. ``python -I /path/to/release_smoke.py``.
Use ``--core-only`` to prove MuJoCo is absent, or ``--viz`` to render an actual frame.
This is a packaging/workflow smoke check; the full test suite remains a separate gate.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.resources
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import cascade
from cascade import env


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_tree(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        require(bool(np.all(np.isfinite(np.asarray(leaf)))), "nonfinite simulation result")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--core-only", action="store_true")
    mode.add_argument("--viz", action="store_true")
    parser.add_argument("--expected-version")
    args = parser.parse_args()

    package = Path(cascade.__file__).resolve()
    require(
        package.is_relative_to(Path(sys.prefix).resolve()), "package is outside this environment"
    )
    require("site-packages" in package.parts, "smoke check requires a noneditable installation")
    version = importlib.metadata.version("cascade-flight")
    if args.expected_version:
        require(version == args.expected_version, f"unexpected installed version {version}")
    if args.core_only:
        require(importlib.util.find_spec("mujoco") is None, "core environment contains MuJoCo")

    for module_name in (
        "cascade",
        "cascade.env",
        "cascade.control",
        "cascade.design",
        "cascade.analysis",
        "cascade.experiments",
    ):
        module = importlib.import_module(module_name)
        for name in module.__all__:
            getattr(module, name)
    resources = importlib.resources.files("cascade")
    require(resources.joinpath("py.typed").is_file(), "wheel omitted py.typed")
    fixtures = (
        "aerobatic_reference",
        "tailsitter_reference",
        "skywalker_x8",
        "skywalker_x8_panels",
    )
    for name in fixtures:
        require(resources.joinpath("aircraft", name + ".toml").is_file(), f"missing {name}.toml")
        cascade.validate_model(getattr(cascade, name + "_spec")().to_model())

    dt = 0.002
    environment = cascade.standard_environment()
    for factory in (cascade.aerobatic_reference, cascade.skywalker_x8):
        model = factory()
        initial = cascade.zero_state(model, altitude=50.0, forward_speed=12.0)
        controls = cascade.repeat_control(cascade.zero_control(model), 4)
        final, trajectory = jax.jit(cascade.rollout)(model, initial, controls, environment, dt)
        finite_tree((final, trajectory))
        require(trajectory.rigid_body.position.shape == (4, 3), "rollout shape mismatch")

    reference = env.ReferenceFlight(initial, cascade.zero_control(model), environment)
    task = env.tracking_task(12.0, 50.0, 0.0)
    config = env.EpisodeConfig(horizon_steps=4)
    episode, obs = jax.jit(lambda key: env.reset(model, config, task, reference, key))(
        jax.random.PRNGKey(7)
    )
    require(obs.shape == (env.observation_size(model),), "reset observation shape mismatch")
    result = jax.jit(lambda state, action: env.step(model, config, task, reference, state, action))(
        episode, jnp.zeros(env.action_size(model))
    )
    finite_tree(result)

    with tempfile.TemporaryDirectory() as temporary:
        record = cascade.stamp(model=model, seed=7, purpose="release smoke")
        require(record["git_commit"] is None, "installed package inherited a repository commit")
        path = cascade.save_trajectory(
            Path(temporary) / "flight", trajectory, dt, t0_s=dt, controls=controls, stamp=record
        )
        loaded, loaded_controls, metadata = cascade.load_trajectory(path)
        np.testing.assert_allclose(loaded.rigid_body.position, trajectory.rigid_body.position)
        np.testing.assert_array_equal(loaded_controls.propeller, controls.propeller)
        require(metadata["t0_s"] == dt, "trajectory time origin mismatch")
        from cascade.experiments import Experiment, Scenario
        from cascade.viz import trajectory_report

        experiment = Experiment(
            "installed-smoke",
            (
                Scenario(
                    "nominal", cascade.aerobatic_reference_spec(), env.tracking_task(12.0, 50.0)
                ),
            ),
        )
        manifest = experiment.save(Path(temporary) / "manifest.json")
        require(Experiment.load(manifest).sha256 == experiment.sha256, "manifest mismatch")
        report = trajectory_report(path, Path(temporary) / "report.html")
        require("flight-data" in report.read_text(), "inspector template missing or unusable")

    if args.viz:
        from cascade.viz import Scene

        spec = cascade.aerobatic_reference_spec()
        scene = Scene(spec, width=160, height=120)
        try:
            scene.pose(cascade.zero_state(spec.to_model(), altitude=5.0))
            frame = scene.frame()
            require(frame.shape == (120, 160, 3), "rendered frame shape mismatch")
            require(frame.dtype == np.uint8 and np.ptp(frame) > 0, "rendered frame is empty")
        finally:
            scene.close()
    if args.core_only:
        require("mujoco" not in sys.modules, "core smoke imported MuJoCo")
    print(
        json.dumps(
            {
                "version": version,
                "package": str(package),
                "python": sys.version.split()[0],
                "jax": jax.__version__,
                "backend": jax.default_backend(),
                "x64": bool(jax.config.jax_enable_x64),
                "mode": "viz" if args.viz else "core-only" if args.core_only else "core",
                "result": "passed",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
