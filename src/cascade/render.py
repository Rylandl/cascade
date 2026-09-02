"""Render Cascade flights with MuJoCo: kinematic playback of a trajectory through the MJCF
that :mod:`cascade.geometry` writes, with flaps and propellers moving and panels coloured by
their separation state, encoded to MP4 by ffmpeg.

MuJoCo is an optional dependency (``cascade-flight[viz]``); ffmpeg must be on the path for
video. Physics stays in JAX: MuJoCo only draws.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from cascade.canonical import rigid_body_to_canonical
from cascade.geometry import aircraft_parts, mjcf_string
from cascade.spec import AircraftSpec
from cascade.state import AircraftState

ATTACHED_RGBA = np.array([0.82, 0.84, 0.88, 1.0])
SEPARATED_RGBA = np.array([0.9, 0.2, 0.15, 1.0])


class Scene:
    """A MuJoCo model of one aircraft, posed from Cascade states. Posing needs no display;
    the renderer (and its OpenGL context) is created on the first frame."""

    def __init__(self, spec: AircraftSpec, *, width: int = 960, height: int = 540, **mjcf_kwargs):
        import mujoco

        self.mujoco = mujoco
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_string(mjcf_string(spec, **mjcf_kwargs))
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self._renderer = None  # created on the first frame: needs an OpenGL context
        self.parts = aircraft_parts(spec)
        self.surface_geoms: dict[int, list[int]] = {}
        self.flap_joints: dict[int, int] = {}
        self.propeller_joints: dict[int, int] = {}
        for part in self.parts:
            if part.surface is not None:
                geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, part.name)
                self.surface_geoms.setdefault(part.surface, []).append(geom)
                if part.hinge is not None:
                    joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, part.name)
                    self.flap_joints[part.surface] = int(self.model.jnt_qposadr[joint])
            if part.propeller is not None and part.hinge is not None:
                joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, part.name)
                self.propeller_joints[part.propeller] = int(self.model.jnt_qposadr[joint])
        self.propeller_angle = np.zeros(len(spec.propellers))

    @property
    def renderer(self):
        if self._renderer is None:
            self._renderer = self.mujoco.Renderer(self.model, height=self.height, width=self.width)
        return self._renderer

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def pose(
        self, state: AircraftState, dt: float = 0.0, *, colour_separation: bool = True
    ) -> None:
        """Set the free joint from the rigid-body state (NED/FRD to NWU/FLU), the flaps from
        the actuator deflections, spin the propellers by ``dt``, and colour the panels."""

        canonical = np.asarray(rigid_body_to_canonical(state.rigid_body), dtype=float)
        self.data.qpos[0:3] = canonical[0:3]
        self.data.qpos[3:7] = canonical[3:7]
        deflection = np.asarray(state.actuators.surface_deflection, dtype=float)
        for surface, address in self.flap_joints.items():
            self.data.qpos[address] = deflection[surface]
        speed = np.asarray(state.actuators.propeller_speed, dtype=float)
        self.propeller_angle = self.propeller_angle + speed * dt
        for propeller, address in self.propeller_joints.items():
            self.data.qpos[address] = self.propeller_angle[propeller]
        if colour_separation:
            separation = np.clip(np.asarray(state.aero.separation, dtype=float), 0.0, 1.0)
            for surface, geoms in self.surface_geoms.items():
                rgba = ATTACHED_RGBA + separation[surface] * (SEPARATED_RGBA - ATTACHED_RGBA)
                for geom in geoms:
                    self.model.geom_rgba[geom] = rgba
        self.mujoco.mj_forward(self.model, self.data)

    def frame(self, camera: str = "chase") -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()


def _index_leaf(state: AircraftState, index: int) -> AircraftState:
    import jax

    return jax.tree.map(lambda leaf: leaf[index], state)


def render_trajectory(
    spec: AircraftSpec,
    trajectory: AircraftState,
    dt: float,
    path: str | Path,
    *,
    fps: int = 30,
    camera: str = "chase",
    width: int = 960,
    height: int = 540,
    colour_separation: bool = True,
    ground_camera_offset=None,
) -> Path:
    """Encode a time-major trajectory (states after each ``dt`` step) to an MP4 at ``fps``.

    ``camera`` is ``chase`` (behind and above the aircraft, turning with it), ``side``,
    ``follow`` (behind and above in world axes, following the position only: right for a
    tailsitter), or ``ground`` (a fixed point beside the flight path that tracks the aircraft;
    ``ground_camera_offset`` places it relative to the path's centroid, in NWU metres).
    """

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to write video")
    steps = int(trajectory.rigid_body.position.shape[0])
    stride = max(int(round(1.0 / (fps * dt))), 1)
    indices = list(range(0, steps, stride))
    scene = Scene(spec, width=width, height=height)
    if camera == "ground":
        # A fixed point beside the flight path, far enough to keep the whole path in view.
        positions = np.asarray(
            [rigid_body_to_canonical(_index_leaf(trajectory, i).rigid_body)[:3] for i in indices]
        )
        centroid = positions.mean(axis=0)
        extent = max(float(np.max(np.ptp(positions, axis=0))), 10.0 * spec.reference_span_m)
        if ground_camera_offset is None:
            offset = np.array([-0.6, 0.6, 0.35]) * extent
        else:
            offset = np.asarray(ground_camera_offset)
        camera_id = scene.mujoco.mj_name2id(scene.model, scene.mujoco.mjtObj.mjOBJ_CAMERA, "ground")
        scene.model.cam_pos[camera_id] = centroid + offset
        # A telephoto field of view so a distant aircraft still fills a useful part of the
        # frame: about six spans across the picture at the camera's distance.
        distance = float(np.linalg.norm(offset))
        fovy = np.degrees(2.0 * np.arctan(3.0 * spec.reference_span_m / max(distance, 1e-3)))
        scene.model.cam_fovy[camera_id] = float(np.clip(fovy, 2.0, 60.0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index in indices:
            scene.pose(
                _index_leaf(trajectory, index), stride * dt, colour_separation=colour_separation
            )
            process.stdin.write(scene.frame(camera).tobytes())
    finally:
        process.stdin.close()
        process.wait()
        scene.close()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {process.returncode}")
    return path


__all__ = ["Scene", "render_trajectory"]
