import os
import sys

import numpy as np
import pytest

import cascade
from cascade.design.archetypes import ConventionalDesign, FlyingWingDesign, design_spec
from cascade.viz.geometry import aircraft_parts, mjcf_string, surface_parts, write_obj

SPECS = {
    "aerobatic": cascade.aerobatic_reference_spec(),
    "x8": cascade.skywalker_x8_spec(),
    "tailsitter": cascade.tailsitter_reference_spec(),
    "flying_wing": design_spec(FlyingWingDesign()),
    "conventional": design_spec(ConventionalDesign(tail="v_tail")),
}


def test_surfaces_become_boxes_in_the_flu_frame():
    spec = SPECS["aerobatic"]
    parts = surface_parts(spec)
    names = [p.name for p in parts]
    # Flapped wings split into a fixed part and a flap; the all-moving tail is one hinged part.
    assert "surface_0" in names and "flap_0" in names
    tail = next(p for p in parts if p.surface == 2)
    assert tail.hinge is not None and tail.name == "surface_2"
    # The fin spans upward: its local span axis maps to +z (up) in FLU.
    fin = next(p for p in parts if p.surface == 3)
    span_axis = np.asarray(fin.rotation)[:, 1]
    assert abs(abs(span_axis[2]) - 1.0) < 1e-6
    # The right wing (FRD +y) sits at negative y in FLU.
    right = next(p for p in parts if p.surface == 1)
    assert right.position[1] < 0.0


def test_every_spec_draws_and_a_coefficient_aircraft_gets_a_wing_outline():
    for spec in SPECS.values():
        parts = aircraft_parts(spec)
        assert parts[-1].name == "pod"
        assert any(p.propeller is not None for p in parts)
        assert all(np.all(np.isfinite(p.position)) for p in parts)
    x8 = aircraft_parts(SPECS["x8"])
    assert {p.name for p in x8} >= {"left_wing", "right_wing"}


def test_obj_and_mjcf_are_written(tmp_path):
    spec = SPECS["flying_wing"]
    write_obj(spec, tmp_path / "wing.obj")
    text = (tmp_path / "wing.obj").read_text()
    assert text.count("\nv ") > 40 and "g pod" in text
    xml = mjcf_string(spec)
    assert '<body name="aircraft"' in xml and 'name="chase"' in xml
    assert xml.count("<joint") >= 6  # elevons and two propellers


def _gl_available() -> bool:
    """MuJoCo aborts the process (no exception) when it cannot create an OpenGL context, so a
    frame is rendered only where a context is known to exist: on request
    (``CASCADE_RENDER_TESTS=1``), with an explicit ``MUJOCO_GL`` backend or a ``DISPLAY`` on
    Linux, or on a macOS session outside CI (GitHub's macOS runners have no window server)."""

    if os.environ.get("CASCADE_RENDER_TESTS") == "1":
        return True
    if os.environ.get("CI"):
        return False
    if sys.platform == "darwin":
        return True
    if os.environ.get("MUJOCO_GL") in {"egl", "osmesa"}:
        return True
    return bool(os.environ.get("DISPLAY"))


@pytest.mark.parametrize("name", list(SPECS))
def test_mjcf_loads_in_mujoco_and_flaps_hinge_trailing_edge_down(name):
    mujoco = pytest.importorskip("mujoco")
    from cascade.viz.render import Scene

    spec = SPECS[name]
    model = mujoco.MjModel.from_xml_string(mjcf_string(spec))
    assert model.nq == 7 + model.njnt - 1
    scene = Scene(spec, width=64, height=48)
    try:
        from cascade.math import quaternion_from_euler

        model_ = spec.to_model()
        state = cascade.zero_state(model_, altitude=20.0)
        scene.pose(state)
        assert abs(float(scene.data.qpos[2]) - 20.0) < 1e-6  # NWU z up

        def world(part_name):
            geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, part_name)
            return scene.data.geom_xpos[geom].copy()

        # A right bank (FRD positive roll) lowers whatever sits at negative FLU y, and a
        # nose-up pitch raises the propeller above the aircraft's centre: the pose is applied.
        propeller = world("propeller_0")
        rolled = state._replace(
            rigid_body=state.rigid_body._replace(
                attitude=quaternion_from_euler(np.deg2rad(30.0), 0.0, 0.0)
            )
        )
        scene.pose(rolled)
        right_side = [
            world(p.name) for p in scene.parts if p.kind == "box" and p.position[1] < -1e-3
        ]
        left_side = [world(p.name) for p in scene.parts if p.kind == "box" and p.position[1] > 1e-3]
        if right_side and left_side:
            assert np.mean([w[2] for w in right_side]) < np.mean([w[2] for w in left_side])
        pitched = state._replace(
            rigid_body=state.rigid_body._replace(
                attitude=quaternion_from_euler(0.0, np.deg2rad(30.0), 0.0)
            )
        )
        scene.pose(pitched)
        nose_up = world("propeller_0")
        forward = propeller[0] - float(scene.data.qpos[0])
        if abs(forward) > 1e-3:
            assert (nose_up[2] - 20.0) * np.sign(forward) > 0.1 * abs(forward)
        if scene.flap_joints:
            surface, address = next(iter(scene.flap_joints.items()))
            part = next(p for p in scene.parts if p.surface == surface and p.hinge is not None)
            geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, part.name)
            scene.data.qpos[address] = 0.0
            mujoco.mj_forward(scene.model, scene.data)
            neutral = scene.data.geom_xpos[geom].copy()
            scene.data.qpos[address] = 0.5
            mujoco.mj_forward(scene.model, scene.data)
            deflected = scene.data.geom_xpos[geom].copy()
            # Positive deflection is trailing-edge down: the flap centre drops (world z up).
            assert deflected[2] < neutral[2]
        if _gl_available():
            frame = scene.frame("chase")
            assert frame.shape == (48, 64, 3)
    finally:
        scene.close()
