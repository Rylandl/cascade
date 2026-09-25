"""Headless JSBSim adapter with explicit units, state mapping and internal dynamics."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ._xml import FT, SLUG, property_name, write_model

JSBSIM_VERSION = "1.3.1"


class JSBSimAircraft:
    """Execute generated XML loads and JSBSim's independent six-DOF propagation.

    Actuator and separation ODEs use a host-side forward Euler step. JSBSim also uses
    Euler propagation, so the campaign must demonstrate convergence against Cascade's
    RK4 with successively smaller timesteps. No Cascade dynamics are called here.
    The nonrotating, large spherical planet approximates Cascade's flat NED world.
    """

    def __init__(self, spec, directory, *, dt=0.00125, density=1.225):
        os.environ.setdefault("JSBSIM_DEBUG", "0")
        try:
            import jsbsim
        except ImportError as error:
            raise ImportError(
                "Install cascade-flight[validation] to run JSBSim comparisons"
            ) from error

        if jsbsim.__version__ != JSBSIM_VERSION:
            raise RuntimeError(f"comparison requires jsbsim=={JSBSIM_VERSION}")
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(density) or density <= 0:
            raise ValueError("dt and density must be finite and positive")
        self.spec, self.dt, self.density = spec, float(dt), float(density)
        directory = Path(directory).resolve()
        self.xml_path = write_model(spec, directory)
        self.fdm = f = jsbsim.FGFDMExec(str(directory))
        f.set_debug_level(0)
        if not f.load_model("comparison") or not f.load_planet(
            str(directory / "planet.xml"), False
        ):
            raise RuntimeError("JSBSim failed to load the comparison model")
        f["simulation/gravity-model"] = 0
        f["simulation/gravitational-torque"] = 0
        for group in ("rate", "position"):
            for kind in ("rotational", "translational"):
                f[f"simulation/integrator/{group}/{kind}"] = 1
        f.set_dt(self.dt)
        self.surface = np.zeros(len(spec.surfaces))
        self.motor = np.zeros(len(spec.propellers))
        self.separation = np.zeros(len(spec.surfaces))
        self.origin = np.zeros(3)
        self.rotation_origin = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0.0]])

    def _set(self, key, value):
        self.fdm[property_name("cmp/" + key)] = float(value)

    def _get(self, key):
        return self.fdm[property_name("cmp/" + key)]

    def _sync(self):
        self._set("rho", self.density)
        for name in ("surface", "motor", "separation"):
            for i, value in enumerate(getattr(self, name)):
                self._set(f"{name}/{i}", value)

    def refresh(self):
        self._sync()
        self.fdm.suspend_integration()
        try:
            if not self.fdm.run():
                raise RuntimeError("JSBSim suspended evaluation failed")
        finally:
            self.fdm.resume_integration()

    def initialize(self, state, *, equilibrate=False, control=None):
        """Initialize from Cascade-shaped state; frames are NED/FRD, quaternion xyzw.

        The initial horizontal location becomes the tangent origin. Initial altitude
        must be 1000 m for the campaign's gravity normalization.
        """
        rb = state.rigid_body
        position = np.asarray(rb.position, float)
        if not np.isclose(position[2], -1000, atol=1e-5):
            raise ValueError("comparison initial altitude must be 1000 m")
        rotation = Rotation.from_quat(np.asarray(rb.attitude, float))
        yaw, pitch, roll = rotation.as_euler("ZYX")
        body_velocity = rotation.inv().apply(np.asarray(rb.velocity, float))
        f = self.fdm
        for name, value in zip(("phi", "theta", "psi-true"), (roll, pitch, yaw), strict=True):
            f[f"ic/{name}-rad"] = value
        f["ic/lat-gc-deg"], f["ic/long-gc-deg"] = 0, 0
        f["ic/h-sl-ft"] = -position[2] / FT
        for axis, value in zip("uvw", body_velocity, strict=True):
            f[f"ic/{axis}-fps"] = value / FT
        for axis, value in zip("pqr", np.asarray(rb.angular_velocity), strict=True):
            f[f"ic/{axis}-rad_sec"] = value
        self.surface = np.array(state.actuators.surface_deflection, float)
        self.motor = np.array(state.actuators.propeller_speed, float)
        self.separation = np.array(state.aero.separation, float)
        self._sync()
        if not f.run_ic():
            raise RuntimeError("JSBSim initial condition failed")
        self.initial_position = position.copy()
        self.origin = np.array([f[f"position/ecef-{x}-ft"] * FT for x in "xyz"])
        self.refresh()
        if equilibrate:
            if control is None:
                raise ValueError("control is required for equilibrium initialization")
            self.surface, self.motor = self.targets(control)
            for _ in range(100):
                self.refresh()
                target = self.equilibrium()
                error = np.max(np.abs(target - self.separation))
                self.separation = target
                if error < 1e-12:
                    break
            else:
                raise RuntimeError("JSBSim separation equilibrium failed to converge")
            self.refresh()
        # Test actual mass, rather than assuming the XML's weight convention.
        if not np.isclose(f["inertia/mass-slugs"] * SLUG, self.spec.mass_kg, rtol=1e-7):
            raise RuntimeError("JSBSim mass conversion mismatch")

    def targets(self, control):
        surface = np.array(
            [
                np.clip(
                    s.actuator_bias_rad + np.dot(s.control_map_rad, control.channel),
                    -s.actuator_limit_rad,
                    s.actuator_limit_rad,
                )
                for s in self.spec.surfaces
            ]
        )
        motor = np.array(
            [
                p.speed_min_rad_s + np.clip(t, 0, 1) * (p.speed_max_rad_s - p.speed_min_rad_s)
                for p, t in zip(self.spec.propellers, control.propeller, strict=True)
            ]
        )
        return surface, motor

    def equilibrium(self):
        return np.array([self._get(f"equilibrium/{i}") for i in range(len(self.surface))])

    def loads(self):
        return {
            name: np.array([self._get(f"{name}/{i}") for i in range(3)])
            for name in (
                "force",
                "moment",
                "aero-force",
                "aero-moment",
                "propulsion-force",
                "propulsion-moment",
            )
        }

    def accelerations(self):
        f = self.fdm
        angular = np.array([f[f"accelerations/{x}dot-rad_sec2"] for x in "pqr"])
        body_derivative = np.array([f[f"accelerations/{x}dot-ft_sec2"] * FT for x in "uvw"])
        velocity = np.array([f[f"velocities/{x}-fps"] * FT for x in "uvw"])
        rates = np.array([f[f"velocities/{x}-rad_sec"] for x in "pqr"])
        q = self.snapshot()[6:10]
        return Rotation.from_quat(q).apply(body_derivative + np.cross(rates, velocity)), angular

    def snapshot(self):
        """Return position/velocity NED, body-to-NED xyzw quaternion, body FRD rates."""
        f = self.fdm
        ecef = np.array([f[f"position/ecef-{x}-ft"] * FT for x in "xyz"])
        # At latitude/longitude zero, NED basis vectors are +Z, +Y, -X.
        position = (ecef - self.origin) @ self.rotation_origin + self.initial_position
        lat, lon = f["position/lat-gc-rad"], f["position/long-gc-rad"]
        sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
        local_to_ecef = np.array(
            [[-sl * co, -so, -cl * co], [-sl * so, co, -cl * so], [cl, 0, -sl]]
        )
        local_to_origin = self.rotation_origin.T @ local_to_ecef
        velocity = local_to_origin @ np.array(
            [f[f"velocities/v-{x}-fps"] * FT for x in ("north", "east", "down")]
        )
        local_rotation = Rotation.from_euler(
            "ZYX", [f["attitude/psi-rad"], f["attitude/theta-rad"], f["attitude/phi-rad"]]
        )
        quaternion = (Rotation.from_matrix(local_to_origin) * local_rotation).as_quat()
        rates = np.array([f[f"velocities/{x}-rad_sec"] for x in "pqr"])
        return np.concatenate((position, velocity, quaternion, rates))

    def step(self, control):
        target_surface, target_motor = self.targets(control)
        surface_rate = np.array(
            [
                s.actuator_rate_limit_rad_s
                * np.tanh(
                    (target - self.surface[i])
                    / (s.actuator_time_constant_s * s.actuator_rate_limit_rad_s)
                )
                for i, (s, target) in enumerate(
                    zip(self.spec.surfaces, target_surface, strict=True)
                )
            ]
        )
        motor_rate = np.array(
            [
                p.acceleration_limit_rad_s2
                * np.tanh(
                    (target - self.motor[i]) / (p.time_constant_s * p.acceleration_limit_rad_s2)
                )
                for i, (p, target) in enumerate(
                    zip(self.spec.propellers, target_motor, strict=True)
                )
            ]
        )
        separation_rate = np.array(
            [self._get(f"separation-rate/{i}") for i in range(len(self.surface))]
        )
        self.surface += self.dt * surface_rate
        self.motor += self.dt * motor_rate
        self.separation += self.dt * separation_rate
        limits = np.array([s.actuator_limit_rad for s in self.spec.surfaces])
        self.surface = np.clip(self.surface, -limits, limits)
        self.motor = np.maximum(self.motor, 0)
        self.separation = np.clip(self.separation, 0, 1)
        self._sync()
        if not self.fdm.run():
            raise RuntimeError("JSBSim propagation failed")
        result = self.snapshot()
        if not np.isfinite(result).all():
            raise FloatingPointError("JSBSim produced a nonfinite state")
        return result
