import numpy as np
import pytest

import cascade
from cascade.archetypes import ConventionalDesign, FlyingWingDesign, design_spec
from cascade.geometry import aircraft_parts, mjcf_string, surface_parts, write_obj

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


@pytest.mark.parametrize("name", list(SPECS))
def test_mjcf_loads_in_mujoco_and_flaps_hinge_trailing_edge_down(name):
    mujoco = pytest.importorskip("mujoco")
    from cascade.render import Scene

    spec = SPECS[name]
    model = mujoco.MjModel.from_xml_string(mjcf_string(spec))
    assert model.nq == 7 + model.njnt - 1
    try:
        scene = Scene(spec, width=64, height=48)
    except Exception as error:  # no OpenGL context on this machine
        pytest.skip(f"no renderer: {error}")
    try:
        model_ = cascade.load_aircraft_spec if False else None  # noqa: F841
        state = cascade.zero_state(spec.to_model(), altitude=20.0)
        scene.pose(state)
        frame = scene.frame("chase")
        assert frame.shape == (48, 64, 3)
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
    finally:
        scene.close()
