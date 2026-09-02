"""Visual geometry from an aircraft specification: boxes for surfaces, discs for propellers,
a pod for the fuselage the spec does not describe, and writers for OBJ and MuJoCo MJCF.

The spec has what a drawing needs: every surface carries a position (its quarter-chord
reference point), a chord, an area (so a span width), and an orientation matrix; every
propeller a position, an axis, and a diameter. Flapped surfaces are split into a fixed part
and a flap hinged at 70% chord; all-moving surfaces hinge at their quarter chord; propellers
spin about their axis. All output is in the canonical FLU body frame (x forward, y left, z
up) with a z-up world, which is what MuJoCo and most renderers expect, converted from the
spec's FRD body frame by flipping y and z.

Zero-area surfaces (the coefficient-backend X8's elevons) draw nothing; a spec with no drawn
surface gets a swept wing outline from its reference span and chord so it is still visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cascade.spec import AircraftSpec

FLIP = np.diag([1.0, -1.0, -1.0])  # FRD -> FLU
FLAP_CHORD_FRACTION = 0.3
PANEL_RGBA = (0.82, 0.84, 0.88, 1.0)
FLAP_RGBA = (0.95, 0.55, 0.2, 1.0)
PROPELLER_RGBA = (0.15, 0.15, 0.15, 0.6)
POD_RGBA = (0.9, 0.9, 0.92, 1.0)


@dataclass(frozen=True)
class Part:
    """One rigid visual element in the FLU body frame.

    ``kind`` is ``box`` (size = half extents), ``cylinder`` (size = radius, half length,
    axis along local z), or ``ellipsoid`` (size = semi-axes). ``hinge`` is the joint axis in
    the part's local frame for parts that move (flaps, all-moving surfaces, propellers), with
    the part's ``position`` at the hinge; ``surface`` and ``propeller`` link back to the spec.
    """

    name: str
    kind: str
    position: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]
    size: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hinge: tuple[float, float, float] | None = None
    surface: int | None = None
    propeller: int | None = None


def _flu_point(p) -> np.ndarray:
    return FLIP @ np.asarray(p, dtype=float)


def _flu_rotation(rotation) -> np.ndarray:
    return FLIP @ np.asarray(rotation, dtype=float) @ FLIP


def _tuple3(v) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _tuple33(m) -> tuple[tuple[float, float, float], ...]:
    return tuple(_tuple3(row) for row in np.asarray(m))


def _frame_with_z(axis: np.ndarray) -> np.ndarray:
    """A rotation whose third column is ``axis``."""

    z = axis / max(np.linalg.norm(axis), 1e-9)
    helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= max(np.linalg.norm(x), 1e-9)
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=1)


def surface_parts(spec: AircraftSpec) -> list[Part]:
    """Boxes for every surface with area: a fixed part and a hinged flap, or one hinged
    all-moving part, or a plain box."""

    parts = []
    for index, surface in enumerate(spec.surfaces):
        if surface.area_m2 <= 0.0 or surface.chord_m <= 0.0:
            continue
        chord = surface.chord_m
        width = surface.area_m2 / chord
        thickness = max(0.05 * chord, 0.002)
        rotation = _flu_rotation(surface.body_from_surface)
        reference = _flu_point(surface.position_m)  # quarter chord, FLU
        flapped = surface.flap_effectiveness > 0.0 and surface.all_moving_fraction < 0.5
        all_moving = surface.all_moving_fraction >= 0.5
        # Local x runs forward along the chord from the quarter chord: leading edge at +c/4,
        # trailing edge at -3c/4. Local y is span (left after the flip), z is the normal.
        if all_moving:
            parts.append(
                Part(
                    name=f"surface_{index}",
                    kind="box",
                    position=_tuple3(reference),
                    rotation=_tuple33(rotation),
                    size=(0.5 * chord, 0.5 * width, 0.5 * thickness),
                    rgba=FLAP_RGBA,
                    offset=(-0.25 * chord, 0.0, 0.0),
                    hinge=(0.0, -1.0, 0.0),
                    surface=index,
                )
            )
            continue
        if flapped:
            fixed_chord = (1.0 - FLAP_CHORD_FRACTION) * chord
            flap_chord = FLAP_CHORD_FRACTION * chord
            fixed_centre = reference + rotation @ np.array([0.25 * chord - 0.5 * fixed_chord, 0, 0])
            hinge_point = reference + rotation @ np.array([0.25 * chord - fixed_chord, 0, 0])
            parts.append(
                Part(
                    name=f"surface_{index}",
                    kind="box",
                    position=_tuple3(fixed_centre),
                    rotation=_tuple33(rotation),
                    size=(0.5 * fixed_chord, 0.5 * width, 0.5 * thickness),
                    rgba=PANEL_RGBA,
                    surface=index,
                )
            )
            parts.append(
                Part(
                    name=f"flap_{index}",
                    kind="box",
                    position=_tuple3(hinge_point),
                    rotation=_tuple33(rotation),
                    size=(0.5 * flap_chord, 0.5 * width, 0.4 * thickness),
                    rgba=FLAP_RGBA,
                    offset=(-0.5 * flap_chord, 0.0, 0.0),
                    hinge=(0.0, -1.0, 0.0),
                    surface=index,
                )
            )
            continue
        centre = reference + rotation @ np.array([-0.25 * chord, 0.0, 0.0])
        parts.append(
            Part(
                name=f"surface_{index}",
                kind="box",
                position=_tuple3(centre),
                rotation=_tuple33(rotation),
                size=(0.5 * chord, 0.5 * width, 0.5 * thickness),
                rgba=PANEL_RGBA,
                surface=index,
            )
        )
    return parts


def fallback_wing_parts(spec: AircraftSpec, sweep_rad: float = 0.45) -> list[Part]:
    """A swept wing outline from the reference span and chord, for specs whose surfaces have
    no area to draw (coefficient-backend aircraft)."""

    half = 0.5 * spec.reference_span_m
    chord = spec.reference_chord_m
    parts = []
    for side, sign in (("left", 1.0), ("right", -1.0)):  # FLU: +y is left
        yaw = -sign * sweep_rad
        rotation = np.array(
            [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0, 0, 1]]
        )
        centre = np.array([-0.5 * half * math.tan(sweep_rad), sign * 0.5 * half, 0.0])
        parts.append(
            Part(
                name=f"{side}_wing",
                kind="box",
                position=_tuple3(centre),
                rotation=_tuple33(rotation),
                size=(0.5 * chord, 0.5 * half, max(0.03 * chord, 0.002)),
                rgba=PANEL_RGBA,
            )
        )
    return parts


def propeller_parts(spec: AircraftSpec) -> list[Part]:
    """A spinning disc per propeller (hinge about its axis) and a small motor pod."""

    parts = []
    for index, propeller in enumerate(spec.propellers):
        axis = _flu_point(propeller.direction_body)
        rotation = _frame_with_z(axis)
        position = _flu_point(propeller.position_m)
        radius = 0.5 * propeller.diameter_m
        parts.append(
            Part(
                name=f"propeller_{index}",
                kind="cylinder",
                position=_tuple3(position),
                rotation=_tuple33(rotation),
                size=(radius, 0.004, 0.0),
                rgba=PROPELLER_RGBA,
                hinge=(0.0, 0.0, 1.0),
                propeller=index,
            )
        )
        parts.append(
            Part(
                name=f"motor_{index}",
                kind="cylinder",
                position=_tuple3(position - rotation[:, 2] * 0.15 * radius),
                rotation=_tuple33(rotation),
                size=(0.12 * radius, 0.15 * radius, 0.0),
                rgba=(0.3, 0.3, 0.32, 1.0),
                propeller=index,
            )
        )
    return parts


def pod_part(spec: AircraftSpec, drawn: list[Part]) -> Part:
    """An ellipsoid fuselage spanning the drawn parts along x, for the body the spec omits,
    fat enough to fill the gap between the innermost wing panels."""

    xs = []
    root_gap = None
    for part in drawn:
        rotation = np.asarray(part.rotation)
        centre = np.asarray(part.position) + rotation @ np.asarray(part.offset)
        reach = abs(rotation[0, 0]) * part.size[0] + abs(rotation[0, 1]) * part.size[1]
        xs.extend((centre[0] - reach, centre[0] + reach))
        span_reach = abs(rotation[1, 1]) * part.size[1] + abs(rotation[1, 0]) * part.size[0]
        inner = abs(centre[1]) - span_reach
        if abs(centre[1]) > 1e-6 and abs(rotation[2, 2]) > 0.5:
            root_gap = inner if root_gap is None else min(root_gap, inner)
    for propeller in spec.propellers:
        xs.append(float(_flu_point(propeller.position_m)[0]))
    x_min, x_max = (min(xs), max(xs)) if xs else (-0.5, 0.5)
    length = max(x_max - x_min, 0.05)
    radius = max(min(0.045 * length, 0.25 * spec.reference_chord_m), 0.008)
    if root_gap is not None and root_gap > 0.0:
        radius = max(radius, min(1.05 * root_gap, 0.5 * spec.reference_chord_m))
    return Part(
        name="pod",
        kind="ellipsoid",
        position=(0.5 * (x_min + x_max), 0.0, 0.0),
        rotation=_tuple33(np.eye(3)),
        size=(0.5 * length, radius, radius),
        rgba=POD_RGBA,
    )


def aircraft_parts(spec: AircraftSpec) -> list[Part]:
    """Everything to draw for a spec, in the FLU body frame."""

    surfaces = surface_parts(spec)
    if not surfaces:
        surfaces = fallback_wing_parts(spec)
    propellers = propeller_parts(spec)
    return [*surfaces, *propellers, pod_part(spec, surfaces)]


# --------------------------------------------------------------------------------------------
# OBJ


def _box_vertices(part: Part) -> np.ndarray:
    sx, sy, sz = part.size
    corners = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    rotation = np.asarray(part.rotation)
    return np.asarray(part.position) + (corners + np.asarray(part.offset)) @ rotation.T


def write_obj(spec: AircraftSpec, path: str | Path) -> None:
    """Write the parts as an OBJ mesh (boxes as boxes; cylinders and the pod as boxes of
    their extents), grouped by part name, in the FLU body frame in metres."""

    faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    lines = [f"# cascade geometry for {spec.name}", "o aircraft"]
    offset = 1
    for part in aircraft_parts(spec):
        box = (
            part
            if part.kind == "box"
            else Part(
                **{
                    **part.__dict__,
                    "kind": "box",
                    "size": (
                        part.size[0] if part.kind == "ellipsoid" else part.size[0],
                        part.size[1] if part.kind == "ellipsoid" else part.size[0],
                        part.size[2] if part.kind == "ellipsoid" else part.size[1],
                    ),
                }
            )
        )
        vertices = _box_vertices(box)
        lines.append(f"g {part.name}")
        lines.extend(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in vertices)
        lines.extend(" ".join(["f", *(str(offset + i) for i in face)]) for face in faces)
        offset += len(vertices)
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------------
# MJCF


def _quat_wxyz(rotation) -> tuple[float, float, float, float]:
    m = np.asarray(rotation, dtype=float)
    trace = np.trace(m)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x, y, z = (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return (float(w), float(x), float(y), float(z))


def camera_xyaxes(offset) -> str:
    """MuJoCo ``xyaxes`` for a camera at ``offset`` from its target, looking at the target with
    world z up: the camera looks along its -z, x is its right, y its up."""

    offset = np.asarray(offset, dtype=float)
    z = offset / max(np.linalg.norm(offset), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return _fmt((*x, *y))


def _fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _geom_xml(part: Part, name: str) -> str:
    size = part.size[:2] if part.kind == "cylinder" else part.size
    return (
        f'<geom name="{name}" type="{part.kind}" size="{_fmt(size)}" pos="{_fmt(part.offset)}" '
        f'rgba="{_fmt(part.rgba)}" contype="0" conaffinity="0"/>'
    )


def mjcf_string(
    spec: AircraftSpec, *, ground: bool = True, chase_distance: float | None = None
) -> str:
    """MJCF for kinematic playback: a free body with the parts as geoms, hinged flaps and
    propellers as child bodies with joints, a ground plane, sky, lights, a chase camera on
    the body, a world-aligned follow camera, and a ground camera that tracks it. Gravity is
    off; Cascade drives the pose."""

    parts = aircraft_parts(spec)
    span = spec.reference_span_m
    distance = 2.3 * span if chase_distance is None else chase_distance
    no_contact = 'contype="0" conaffinity="0"'
    lines = [
        f'<mujoco model="{spec.name}">',
        '  <compiler angle="radian"/>',
        '  <option gravity="0 0 0" timestep="0.01"/>',
        "  <visual>",
        '    <global offwidth="1920" offheight="1080"/>',
        '    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>',
        '    <quality shadowsize="4096"/>',
        '    <map znear="0.01" zfar="2000"/>',
        "  </visual>",
        "  <asset>",
        '    <texture type="skybox" builtin="gradient" rgb1="0.62 0.76 0.95" '
        'rgb2="0.15 0.25 0.5" width="256" height="256"/>',
        '    <texture name="grid" type="2d" builtin="checker" rgb1="0.52 0.62 0.45" '
        'rgb2="0.44 0.54 0.38" width="512" height="512"/>',
        '    <material name="grid" texture="grid" texrepeat="60 60" texuniform="true" '
        'reflectance="0.05"/>',
        "  </asset>",
        "  <worldbody>",
        '    <light directional="true" pos="0 0 100" dir="-0.3 0.2 -1" diffuse="0.8 0.8 0.8" '
        'specular="0.2 0.2 0.2" castshadow="true"/>',
    ]
    if ground:
        lines.append(
            f'    <geom name="ground" type="plane" size="0 0 1" material="grid" {no_contact}/>'
        )
    ground_camera = _fmt((-2 * distance, 2 * distance, 0.6 * distance))
    lines.append(
        f'    <camera name="ground" mode="targetbody" target="aircraft" pos="{ground_camera}"/>'
    )
    # A world-aligned follower: keeps its offset from the aircraft's centre of mass but not
    # its attitude, so a hovering or tumbling tailsitter stays framed from behind and above.
    # MuJoCo measures a tracking camera's offset from the initial centre of mass, so the
    # camera is placed relative to where the aircraft body starts (0, 0, 1).
    follow_offset = (-1.4 * distance, 0.9 * distance, 0.6 * distance)
    follow_position = (follow_offset[0], follow_offset[1], follow_offset[2] + 1.0)
    lines.append(
        f'    <camera name="follow" mode="trackcom" pos="{_fmt(follow_position)}" '
        f'xyaxes="{camera_xyaxes(follow_offset)}"/>'
    )
    lines.append('    <body name="aircraft" pos="0 0 1">')
    lines.append('      <freejoint name="root"/>')
    chase_offset = (-distance, 0.0, 0.35 * distance)
    lines.append(
        f'      <camera name="chase" pos="{_fmt(chase_offset)}" '
        f'xyaxes="{camera_xyaxes(chase_offset)}"/>'
    )
    side_offset = (0.0, -1.6 * distance, 0.3 * distance)
    lines.append(
        f'      <camera name="side" pos="{_fmt(side_offset)}" '
        f'xyaxes="{camera_xyaxes(side_offset)}"/>'
    )
    for part in parts:
        quat = _fmt(_quat_wxyz(part.rotation))
        if part.hinge is None:
            rotation = np.asarray(part.rotation)
            centre = _fmt(np.asarray(part.position) + rotation @ np.asarray(part.offset))
            size = _fmt(part.size[:2] if part.kind == "cylinder" else part.size)
            lines.append(
                f'      <geom name="{part.name}" type="{part.kind}" size="{size}" '
                f'pos="{centre}" quat="{quat}" rgba="{_fmt(part.rgba)}" {no_contact}/>'
            )
        else:
            lines.append(
                f'      <body name="{part.name}" pos="{_fmt(part.position)}" quat="{quat}">'
            )
            lines.append(
                f'        <joint name="{part.name}" type="hinge" axis="{_fmt(part.hinge)}" '
                'limited="false"/>'
            )
            lines.append("        " + _geom_xml(part, part.name))
            lines.append("      </body>")
    lines.extend(["    </body>", "  </worldbody>", "</mujoco>"])
    return "\n".join(lines) + "\n"


def write_mjcf(spec: AircraftSpec, path: str | Path, **kwargs) -> None:
    Path(path).write_text(mjcf_string(spec, **kwargs))


__all__ = [
    "Part",
    "aircraft_parts",
    "camera_xyaxes",
    "fallback_wing_parts",
    "mjcf_string",
    "pod_part",
    "propeller_parts",
    "surface_parts",
    "write_mjcf",
    "write_obj",
]
