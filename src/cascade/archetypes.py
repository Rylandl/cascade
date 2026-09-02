"""Parametric airframe archetypes: a handful of design decisions to a full aircraft spec.

An archetype maps the choices a designer actually makes (span, aspect ratio, wing loading,
sweep, tail volumes, static margin, control-surface fractions, thrust-to-weight, ...) onto the
panel backend through textbook relations: lift slope from aspect ratio (Helmbold), induced drag
from span efficiency, flap effectiveness and flap moment from chord fraction (thin airfoil),
static wing downwash folded into the tail, propwash weights from disk coverage, inertia from
geometry and a mass split. The designs are plausible, not validated; their purpose is a family
of visibly different airframes whose parameters a learner never sees.

Frames: body FRD with the origin at the centre of mass. Sweep is of the quarter-chord line and
positive aft; washout is positive when the tip flies at lower incidence than the root; reflex is
the section's zero-lift pitching moment (positive nose-up); static margin is the centre of
mass ahead of the neutral point as a fraction of the mean aerodynamic chord.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import jax
import numpy as np

from cascade.spec import AircraftSpec, PropellerSpec, SurfaceSpec

GRAVITY = 9.80665
DENSITY = 1.225
STATIC_THRUST_COEFFICIENT = 0.15  # c_20: T = rho n^2 D^4 c_20 at rest
SECTION_LIFT_SLOPE = 5.7  # 2 pi corrected for thickness and viscosity, per rad
SPAN_EFFICIENCY = 0.85


# --------------------------------------------------------------------------------------------
# Designs


@dataclass(frozen=True)
class FlyingWingDesign:
    """A tailless wing with elevons, optional winglets, and a pusher or a twin tractor layout."""

    span_m: float = 1.0
    aspect_ratio: float = 5.0
    wing_loading_kg_m2: float = 3.0
    sweep_rad: float = 0.35
    taper: float = 0.6
    washout_rad: float = 0.05
    reflex: float = 0.02
    camber: float = 0.05
    static_margin: float = 0.08
    elevon_span_fraction: float = 0.9
    elevon_chord_fraction: float = 0.25
    winglet_area_fraction: float = 0.06
    thrust_to_weight: float = 0.8
    propeller_diameter_fraction: float = 0.2
    motors: Literal["pusher", "twin_tractor"] = "pusher"
    pod_mass_fraction: float = 0.5
    cruise_lift_coefficient: float = 0.4


@dataclass(frozen=True)
class ConventionalDesign:
    """A wing with ailerons, a tail (conventional or V) with elevator and rudder, tractor prop."""

    span_m: float = 1.2
    aspect_ratio: float = 7.0
    wing_loading_kg_m2: float = 4.0
    camber: float = 0.2
    dihedral_rad: float = 0.05
    tail_arm_chords: float = 2.5
    horizontal_tail_volume: float = 0.5
    vertical_tail_volume: float = 0.035
    static_margin: float = 0.12
    aileron_span_fraction: float = 0.45
    aileron_chord_fraction: float = 0.25
    elevator_chord_fraction: float = 0.4
    rudder_chord_fraction: float = 0.4
    tail: Literal["conventional", "v_tail"] = "conventional"
    thrust_to_weight: float = 0.7
    propeller_diameter_fraction: float = 0.22
    pod_mass_fraction: float = 0.55
    cruise_lift_coefficient: float = 0.45


Design = FlyingWingDesign | ConventionalDesign


# --------------------------------------------------------------------------------------------
# Aerodynamic and structural relations


def wing_lift_slope(aspect_ratio: float, sweep_rad: float = 0.0) -> float:
    """Helmbold's finite-wing correction of the section slope, with simple sweep."""

    a0 = SECTION_LIFT_SLOPE * math.cos(sweep_rad)
    ratio = a0 / (math.pi * aspect_ratio)
    return a0 / (math.sqrt(1.0 + ratio**2) + ratio)


def induced_drag_factor(aspect_ratio: float) -> float:
    return 1.0 / (math.pi * aspect_ratio * SPAN_EFFICIENCY)


def flap_effectiveness(chord_fraction: float) -> float:
    """Thin-airfoil ``d alpha_eff / d delta`` for a plain flap, reduced for viscosity."""

    theta = math.acos(2.0 * chord_fraction - 1.0)
    return 0.85 * (1.0 - (theta - math.sin(theta)) / math.pi)


def flap_moment_coefficient(chord_fraction: float) -> float:
    """Thin-airfoil quarter-chord moment per radian of plain-flap deflection (nose-down)."""

    theta = math.acos(2.0 * chord_fraction - 1.0)
    return -0.8 * 0.5 * math.sin(theta) * (1.0 - math.cos(theta))


def rotation_x(angle: float) -> tuple[tuple[float, float, float], ...]:
    c, s = math.cos(angle), math.sin(angle)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def rotation_y(angle: float) -> tuple[tuple[float, float, float], ...]:
    c, s = math.cos(angle), math.sin(angle)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def _matmul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


VERTICAL_UP = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))  # span up, normal to +y


def _surface(
    name: str,
    position: tuple[float, float, float],
    frame,
    area: float,
    chord: float,
    *,
    lift_slope: float,
    camber: float = 0.0,
    moment_zero: float = 0.0,
    induced: float,
    control_map: tuple[float, ...],
    flap_fraction: float = 0.0,
    all_moving: float = 0.0,
    limit_rad: float = 0.5,
    time_constant_s: float = 0.04,
    crossflow: float = 0.15,
) -> SurfaceSpec:
    flapped = flap_fraction > 0.0 and all_moving < 1.0
    return SurfaceSpec(
        name=name,
        position_m=tuple(float(v) for v in position),
        body_from_surface=tuple(tuple(float(v) for v in row) for row in frame),
        area_m2=float(area),
        chord_m=float(chord),
        lift_coefficient_zero=float(camber),
        lift_curve_slope_rad=float(lift_slope),
        drag_coefficient_zero=0.02,
        induced_drag_factor=float(induced),
        moment_coefficient_zero=float(moment_zero),
        moment_coefficient_alpha_rad=0.0,
        stall_angle_rad=0.24,
        stall_width_rad=0.05,
        normal_force_coefficient=1.8,
        edge_drag_coefficient=0.06,
        span_drag_coefficient=float(crossflow),
        separation_time_constant_s=0.06,
        reattachment_time_constant_s=0.12,
        all_moving_fraction=float(all_moving),
        flap_effectiveness=flap_effectiveness(flap_fraction) if flapped else 0.0,
        moment_coefficient_flap_rad=flap_moment_coefficient(flap_fraction) if flapped else 0.0,
        drag_coefficient_flap_rad2=0.08 if flapped else 0.0,
        control_map_rad=tuple(float(v) for v in control_map),
        actuator_bias_rad=0.0,
        actuator_limit_rad=float(limit_rad),
        actuator_time_constant_s=float(time_constant_s),
        actuator_rate_limit_rad_s=10.0,
    )


def _propeller(
    name: str,
    position: tuple[float, float, float],
    diameter: float,
    static_thrust: float,
    cruise_speed_m_s: float,
    spin: float,
    slipstream: tuple[float, ...],
) -> PropellerSpec:
    """A propeller whose full-throttle static thrust is ``static_thrust`` newtons and whose
    pitch lets it cruise: the zero-thrust airspeed at full speed is 1.6 times cruise. A very
    large propeller for its cruise speed gets more speed (and static thrust) rather than an
    impossible pitch."""

    n_max = math.sqrt(static_thrust / (DENSITY * STATIC_THRUST_COEFFICIENT * diameter**4))
    advance_ratio = 1.6 * cruise_speed_m_s / (n_max * diameter)
    if advance_ratio > 1.2:
        n_max *= advance_ratio / 1.2
        advance_ratio = 1.2
    advance_ratio = max(advance_ratio, 0.45)
    return PropellerSpec(
        name=name,
        position_m=tuple(float(v) for v in position),
        direction_body=(1.0, 0.0, 0.0),
        diameter_m=float(diameter),
        thrust_map=(
            (0.0, -STATIC_THRUST_COEFFICIENT / advance_ratio, 0.0),
            (STATIC_THRUST_COEFFICIENT, 0.0, 0.0),
        ),
        torque_coefficient_static=0.0094,
        spin_direction=float(spin),
        slipstream_weights=tuple(float(v) for v in slipstream),
        speed_min_rad_s=0.0,
        speed_max_rad_s=float(2.0 * math.pi * n_max),
        time_constant_s=0.06,
        acceleration_limit_rad_s2=float(2.0 * math.pi * n_max / 0.15),
    )


def _disk_coverage(panel_y0: float, panel_y1: float, prop_y: float, diameter: float) -> float:
    lo, hi = min(panel_y0, panel_y1), max(panel_y0, panel_y1)
    overlap = max(0.0, min(hi, prop_y + 0.5 * diameter) - max(lo, prop_y - 0.5 * diameter))
    return overlap / max(hi - lo, 1e-9)


def _plate_inertia(mass, centre, chord, width, height=0.0):
    """Inertia of a thin plate (chord along x, width along y, height along z) about the origin."""

    x, y, z = centre
    local = np.diag(
        [
            mass * (width**2 + height**2) / 12.0,
            mass * (chord**2 + height**2) / 12.0,
            mass * (chord**2 + width**2) / 12.0,
        ]
    )
    r = np.array([x, y, z])
    return local + mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


def _inertia_from_parts(parts, pod_mass, pod_radius):
    inertia = 0.4 * pod_mass * pod_radius**2 * np.eye(3)
    for mass, centre, chord, width, height in parts:
        inertia += _plate_inertia(mass, centre, chord, width, height)
    inertia = 0.5 * (inertia + inertia.T)
    return tuple(tuple(float(v) for v in row) for row in inertia)


# --------------------------------------------------------------------------------------------
# Flying wing


def flying_wing_spec(design: FlyingWingDesign, name: str = "archetype-flying-wing") -> AircraftSpec:
    b, ar = design.span_m, design.aspect_ratio
    area = b * b / ar
    mass = design.wing_loading_kg_m2 * area
    weight = mass * GRAVITY
    taper = design.taper
    root_chord = 2.0 * area / (b * (1.0 + taper))
    mac = (2.0 / 3.0) * root_chord * (1.0 + taper + taper**2) / (1.0 + taper)
    slope = wing_lift_slope(ar, design.sweep_rad)
    induced = induced_drag_factor(ar)
    half = 0.5 * b
    # Three panels per side, equal in span; the elevon covers the outer fraction of the span.
    edges = [0.0, half / 3.0, 2.0 * half / 3.0, half]
    panels = []
    for i in range(3):
        y0, y1 = edges[i], edges[i + 1]
        y_c = 0.5 * (y0 + y1)
        chord = root_chord * (1.0 - (1.0 - taper) * y_c / half)
        x_qc = -y_c * math.tan(design.sweep_rad)
        incidence = -design.washout_rad * (y_c / half)
        elevon_start = half * (1.0 - design.elevon_span_fraction)
        elevon_share = _disk_coverage(y0, y1, 0.5 * (elevon_start + half), half - elevon_start)
        panels.append((y0, y1, y_c, chord, x_qc, incidence, elevon_share))
    lift_weighted = sum(slope * (y1 - y0) * chord * x for y0, y1, _, chord, x, _, _ in panels)
    lift_total = sum(slope * (y1 - y0) * chord for y0, y1, _, chord, _, _, _ in panels)
    neutral_point = lift_weighted / lift_total
    x_cg = neutral_point + design.static_margin * mac
    surfaces = []
    parts = []
    wing_mass = (1.0 - design.pod_mass_fraction) * mass
    for side, sign in (("left", -1.0), ("right", 1.0)):
        for i, (y0, y1, y_c, chord, x_qc, incidence, elevon_share) in enumerate(panels):
            panel_area = (y1 - y0) * chord
            flap = design.elevon_chord_fraction if elevon_share > 0.05 else 0.0
            gain = 0.5 * elevon_share
            control_map = (sign * gain, gain)  # aileron differential, elevator symmetric
            position = (x_qc - x_cg, sign * y_c, 0.0)
            surfaces.append(
                _surface(
                    f"{side}_wing_{i}",
                    position,
                    rotation_y(incidence),
                    panel_area,
                    chord,
                    lift_slope=slope,
                    camber=design.camber,
                    moment_zero=design.reflex,
                    induced=induced,
                    control_map=control_map,
                    flap_fraction=flap,
                    limit_rad=0.6,
                    time_constant_s=0.03,
                )
            )
            parts.append((wing_mass * panel_area / area, position, chord, y1 - y0, 0.0))
    winglet_area = design.winglet_area_fraction * area
    tip_chord = root_chord * taper
    if winglet_area > 0.0:
        height = winglet_area / (0.6 * tip_chord)
        for side, sign in (("left", -1.0), ("right", 1.0)):
            x_tip = -half * math.tan(design.sweep_rad) - x_cg
            position = (x_tip - 0.1 * tip_chord, sign * half, -0.5 * height)
            surfaces.append(
                _surface(
                    f"{side}_winglet",
                    position,
                    VERTICAL_UP,
                    winglet_area,
                    0.6 * tip_chord,
                    lift_slope=2.5,
                    induced=0.15,
                    control_map=(0.0, 0.0),
                    crossflow=0.35,
                )
            )
            parts.append((0.03 * mass, position, 0.6 * tip_chord, 0.0, height))
    diameter = design.propeller_diameter_fraction * b
    thrust = design.thrust_to_weight * weight
    propellers = []
    if design.motors == "pusher":
        position = (-0.9 * root_chord - x_cg, 0.0, 0.0)
        propellers.append(
            _propeller(
                "pusher",
                position,
                diameter,
                thrust,
                cruise_speed(design),
                1.0,
                tuple(0.0 for _ in surfaces),
            )
        )
        pod_mass = design.pod_mass_fraction * mass
    else:
        y_motor = 0.5 * diameter + 0.05 * half
        for side, sign, spin in (("left", -1.0, 1.0), ("right", 1.0, -1.0)):
            x_motor = -y_motor * math.tan(design.sweep_rad) + 0.3 * root_chord - x_cg
            weights = []
            for surface in surfaces:
                y_c = surface.position_m[1]
                if surface.name.endswith("winglet"):
                    weights.append(0.0)
                    continue
                width = surface.area_m2 / surface.chord_m
                weights.append(
                    1.8
                    * _disk_coverage(y_c - 0.5 * width, y_c + 0.5 * width, sign * y_motor, diameter)
                )
            propellers.append(
                _propeller(
                    f"{side}_motor",
                    (x_motor, sign * y_motor, 0.0),
                    diameter,
                    0.5 * thrust,
                    cruise_speed(design),
                    spin,
                    tuple(weights),
                )
            )
            parts.append((0.05 * mass, (x_motor, sign * y_motor, 0.0), 0.0, 0.0, 0.0))
        pod_mass = (design.pod_mass_fraction - 0.1) * mass
    inertia = _inertia_from_parts(parts, pod_mass, 0.25 * root_chord)
    return AircraftSpec(
        name=name,
        description=f"Flying-wing archetype: {design}",
        mass_kg=float(mass),
        inertia_kg_m2=inertia,
        reference_area_m2=float(area),
        reference_chord_m=float(mac),
        reference_span_m=float(b),
        control_channels=("aileron", "elevator"),
        surfaces=tuple(surfaces),
        propellers=tuple(propellers),
    ).validate()


# --------------------------------------------------------------------------------------------
# Conventional


def conventional_spec(
    design: ConventionalDesign, name: str = "archetype-conventional"
) -> AircraftSpec:
    b, ar = design.span_m, design.aspect_ratio
    area = b * b / ar
    mass = design.wing_loading_kg_m2 * area
    weight = mass * GRAVITY
    chord = area / b
    mac = chord
    slope = wing_lift_slope(ar)
    induced = induced_drag_factor(ar)
    half = 0.5 * b
    tail_arm = design.tail_arm_chords * mac
    horizontal_area = design.horizontal_tail_volume * area * mac / tail_arm
    vertical_area = design.vertical_tail_volume * area * b / tail_arm
    downwash_slope = 2.0 * slope / (math.pi * ar)
    cruise_downwash = 2.0 * cruise_lift_coefficient(design) / (math.pi * ar)
    tail_ar = 4.0
    tail_slope = wing_lift_slope(tail_ar) * (1.0 - downwash_slope)
    fin_slope = wing_lift_slope(1.8)
    # Neutral point from wing (at x = 0, its quarter chord) and tail with downwash.
    tail_x = -tail_arm
    neutral_point = (tail_slope * horizontal_area * tail_x) / (
        slope * area + tail_slope * horizontal_area
    )
    x_cg = neutral_point + design.static_margin * mac
    surfaces = []
    parts = []
    wing_mass = (1.0 - design.pod_mass_fraction - 0.1) * mass
    edges = [0.0, half * (1.0 - design.aileron_span_fraction), half]
    for side, sign in (("left", -1.0), ("right", 1.0)):
        frame = rotation_x(-sign * design.dihedral_rad)
        for i, (y0, y1) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
            y_c = 0.5 * (y0 + y1)
            panel_area = (y1 - y0) * chord
            z_c = -y_c * math.tan(design.dihedral_rad)
            position = (-x_cg, sign * y_c, z_c)
            aileron = i == 1
            surfaces.append(
                _surface(
                    f"{side}_wing_{'outer' if aileron else 'inner'}",
                    position,
                    frame,
                    panel_area,
                    chord,
                    lift_slope=slope,
                    camber=design.camber,
                    moment_zero=-0.25 * design.camber,
                    induced=induced,
                    control_map=(sign * 0.4, 0.0, 0.0) if aileron else (0.0, 0.0, 0.0),
                    flap_fraction=design.aileron_chord_fraction if aileron else 0.0,
                    limit_rad=0.5,
                )
            )
            parts.append((wing_mass * panel_area / area, position, chord, y1 - y0, 0.0))
    tail_position_x = tail_x - x_cg
    tail_chord = math.sqrt(horizontal_area / tail_ar)
    if design.tail == "conventional":
        surfaces.append(
            _surface(
                "horizontal_tail",
                (tail_position_x, 0.0, 0.0),
                rotation_y(-cruise_downwash),
                horizontal_area,
                tail_chord,
                lift_slope=tail_slope,
                induced=induced_drag_factor(tail_ar),
                control_map=(0.0, 0.45, 0.0),
                flap_fraction=design.elevator_chord_fraction,
                limit_rad=0.55,
            )
        )
        fin_chord = math.sqrt(vertical_area / 1.8)
        fin_height = vertical_area / fin_chord
        surfaces.append(
            _surface(
                "vertical_tail",
                (tail_position_x, 0.0, -0.5 * fin_height),
                VERTICAL_UP,
                vertical_area,
                fin_chord,
                lift_slope=fin_slope,
                induced=0.2,
                control_map=(0.0, 0.0, 0.45),
                flap_fraction=design.rudder_chord_fraction,
                limit_rad=0.55,
                crossflow=0.35,
            )
        )
        parts.append(
            (
                0.05 * mass,
                (tail_position_x, 0.0, 0.0),
                tail_chord,
                horizontal_area / tail_chord,
                0.0,
            )
        )
        parts.append(
            (0.03 * mass, (tail_position_x, 0.0, -0.5 * fin_height), fin_chord, 0.0, fin_height)
        )
    else:
        # V-tail: two panels whose projections carry the horizontal and vertical volumes.
        dihedral = math.atan2(math.sqrt(vertical_area), math.sqrt(horizontal_area))
        panel_area = 0.5 * (horizontal_area + vertical_area)
        panel_chord = math.sqrt(2.0 * panel_area / tail_ar)
        span_each = panel_area / panel_chord
        for side, sign in (("left", -1.0), ("right", 1.0)):
            y_c = sign * 0.5 * span_each * math.cos(dihedral)
            z_c = -0.5 * span_each * math.sin(dihedral)
            frame = _matmul(rotation_x(-sign * dihedral), rotation_y(-cruise_downwash))
            surfaces.append(
                _surface(
                    f"{side}_v_tail",
                    (tail_position_x, y_c, z_c),
                    frame,
                    panel_area,
                    panel_chord,
                    lift_slope=tail_slope,
                    induced=induced_drag_factor(tail_ar),
                    control_map=(0.0, 0.45, sign * 0.45),
                    flap_fraction=design.elevator_chord_fraction,
                    limit_rad=0.55,
                    crossflow=0.25,
                )
            )
            parts.append((0.04 * mass, (tail_position_x, y_c, z_c), panel_chord, span_each, 0.0))
    diameter = design.propeller_diameter_fraction * b
    weights = []
    for surface in surfaces:
        y_c = surface.position_m[1]
        if "tail" in surface.name:
            weights.append(1.2)
        else:
            width = surface.area_m2 / surface.chord_m
            weights.append(
                0.6 * _disk_coverage(y_c - 0.5 * width, y_c + 0.5 * width, 0.0, diameter)
            )
    nose = (0.6 * chord - x_cg, 0.0, 0.0)
    propellers = (
        _propeller(
            "nose_propeller",
            nose,
            diameter,
            design.thrust_to_weight * weight,
            cruise_speed(design),
            1.0,
            tuple(weights),
        ),
    )
    parts.append((0.1 * mass, nose, 0.0, 0.0, 0.0))
    inertia = _inertia_from_parts(parts, design.pod_mass_fraction * mass, 0.3 * chord)
    return AircraftSpec(
        name=name,
        description=f"Conventional archetype: {design}",
        mass_kg=float(mass),
        inertia_kg_m2=inertia,
        reference_area_m2=float(area),
        reference_chord_m=float(mac),
        reference_span_m=float(b),
        control_channels=("aileron", "elevator", "rudder"),
        surfaces=tuple(surfaces),
        propellers=propellers,
    ).validate()


def design_spec(design: Design, name: str | None = None) -> AircraftSpec:
    if isinstance(design, FlyingWingDesign):
        return flying_wing_spec(design, name or "archetype-flying-wing")
    if isinstance(design, ConventionalDesign):
        return conventional_spec(design, name or "archetype-conventional")
    raise TypeError(f"unknown design {type(design).__name__}")


def cruise_lift_coefficient(design: Design) -> float:
    """The design's cruise lift coefficient, capped at 60% of the wing's stall lift so a
    low-slope wing (low aspect ratio, high sweep) is not asked to cruise on the edge of stall."""

    sweep = getattr(design, "sweep_rad", 0.0)
    washout = getattr(design, "washout_rad", 0.0)
    slope = wing_lift_slope(design.aspect_ratio, sweep)
    stall_lift = design.camber + slope * (0.24 - 0.5 * washout)
    return min(design.cruise_lift_coefficient, 0.6 * stall_lift)


def cruise_speed(design: Design) -> float:
    """Design cruise speed from wing loading and the (capped) cruise lift coefficient."""

    return math.sqrt(
        2.0 * design.wing_loading_kg_m2 * GRAVITY / (DENSITY * cruise_lift_coefficient(design))
    )


# --------------------------------------------------------------------------------------------
# Validation


@dataclass(frozen=True)
class DesignReport:
    """What a sampled design is like at its cruise trim, and whether it passes."""

    valid: bool
    reasons: tuple[str, ...]
    cruise_speed_m_s: float
    angle_of_attack_rad: float
    throttle: float
    channels: tuple[float, ...]
    pitch_authority_rad_s2: float
    roll_authority_rad_s2: float
    yaw_authority_rad_s2: float
    fastest_unstable_time_constant_s: float | None
    short_period_hz: float
    short_period_damping: float


def control_authority(model, state, control, environment) -> np.ndarray:
    """Angular acceleration per unit channel command, ``(3, C)``, with the surfaces placed at
    the command's steady deflection (so actuator lag does not hide the authority)."""

    from cascade.dynamics import evaluate_dynamics

    def acceleration(channel):
        deflection = model.actuators.surface_map @ channel + model.actuators.surface_bias
        placed = state._replace(actuators=state.actuators._replace(surface_deflection=deflection))
        result = evaluate_dynamics(model, placed, control._replace(channel=channel), environment)
        return result.derivative.rigid_body.angular_velocity

    return np.asarray(jax.jacfwd(acceleration)(control.channel))


def validate_design(
    design: Design,
    spec: AircraftSpec | None = None,
    *,
    altitude_m: float = 50.0,
    timestep_s: float = 0.005,
) -> DesignReport:
    """Trim the design at cruise, linearise, and apply the flyability checks.

    A design passes when it trims within its limits and below stall, has a pitch, roll, and
    (if it has a yaw channel) yaw authority above a floor, and has no unstable mode faster
    than a spiral (time constant under 2 s).
    """

    from cascade.analysis import (
        StraightFlightCondition,
        linearize_step,
        stability_modes,
        trim_straight_flight,
    )
    from cascade.initialization import standard_environment

    spec = design_spec(design) if spec is None else spec
    model = spec.to_model()
    environment = standard_environment()
    speed = cruise_speed(design)
    trim = trim_straight_flight(
        model, StraightFlightCondition(speed, altitude_m=altitude_m), environment=environment
    )
    reasons = []
    channels = tuple(float(v) for v in trim.control.channel)
    throttle = float(np.max(np.asarray(trim.control.propeller)))
    stall = float(np.min(np.asarray(model.surfaces.stall_angle)))
    if not trim.success:
        reasons.append("no cruise trim")
    if trim.angle_of_attack_rad > stall - math.radians(3.0):
        reasons.append("cruise within 3 deg of stall")
    if not 0.05 < throttle < 0.9:
        reasons.append("cruise throttle outside (0.05, 0.9)")
    if any(abs(v) > 0.6 for v in channels):
        reasons.append("cruise needs more than 0.6 of a channel")
    linearization = linearize_step(model, trim.state, trim.control, environment, timestep_s)
    authority = control_authority(model, trim.state, trim.control, environment)
    channel_names = spec.control_channels

    def axis_authority(role: str, axis: int) -> float:
        if role not in channel_names:
            return 0.0
        return abs(float(authority[axis, channel_names.index(role)]))

    roll_authority = axis_authority("aileron", 0)
    pitch_authority = axis_authority("elevator", 1)
    yaw_authority = axis_authority("rudder", 2)
    if pitch_authority < 15.0:
        reasons.append("pitch authority below 15 rad/s^2 per unit elevator")
    if roll_authority < 15.0:
        reasons.append("roll authority below 15 rad/s^2 per unit aileron")
    if "rudder" in channel_names and yaw_authority < 3.0:
        reasons.append("yaw authority below 3 rad/s^2 per unit rudder")
    modes = stability_modes(linearization)
    unstable = [m.time_constant_s for m in modes if not m.stable and m.time_constant_s is not None]
    fastest = min(unstable) if unstable else None
    if fastest is not None and fastest < 2.0:
        reasons.append(f"unstable mode with time constant {fastest:.2f} s")
    oscillatory = [m for m in modes if m.damping_ratio is not None and m.frequency_hz > 0.3]
    short_period = max(oscillatory, key=lambda m: m.frequency_hz) if oscillatory else None
    return DesignReport(
        valid=not reasons,
        reasons=tuple(reasons),
        cruise_speed_m_s=speed,
        angle_of_attack_rad=float(trim.angle_of_attack_rad),
        throttle=throttle,
        channels=channels,
        pitch_authority_rad_s2=pitch_authority,
        roll_authority_rad_s2=roll_authority,
        yaw_authority_rad_s2=yaw_authority,
        fastest_unstable_time_constant_s=fastest,
        short_period_hz=float(short_period.frequency_hz) if short_period else 0.0,
        short_period_damping=float(short_period.damping_ratio) if short_period else 0.0,
    )


# --------------------------------------------------------------------------------------------
# Sampling


FLYING_WING_RANGES = {
    "span_m": (0.6, 2.0),
    "aspect_ratio": (3.5, 8.0),
    "wing_loading_kg_m2": (1.5, 8.0),
    "sweep_rad": (0.15, 0.6),
    "taper": (0.4, 1.0),
    "washout_rad": (0.0, 0.1),
    "reflex": (0.0, 0.04),
    "camber": (0.0, 0.1),
    "static_margin": (0.03, 0.15),
    "elevon_span_fraction": (0.5, 1.0),
    "elevon_chord_fraction": (0.15, 0.35),
    "winglet_area_fraction": (0.0, 0.1),
    "thrust_to_weight": (0.5, 1.5),
    "propeller_diameter_fraction": (0.12, 0.3),
    "pod_mass_fraction": (0.35, 0.65),
    "cruise_lift_coefficient": (0.25, 0.5),
}

CONVENTIONAL_RANGES = {
    "span_m": (0.8, 3.0),
    "aspect_ratio": (5.0, 12.0),
    "wing_loading_kg_m2": (2.0, 12.0),
    "camber": (0.0, 0.4),
    "dihedral_rad": (0.0, 0.12),
    "tail_arm_chords": (1.8, 4.0),
    "horizontal_tail_volume": (0.35, 0.8),
    "vertical_tail_volume": (0.02, 0.06),
    "static_margin": (0.05, 0.2),
    "aileron_span_fraction": (0.3, 0.6),
    "aileron_chord_fraction": (0.15, 0.35),
    "elevator_chord_fraction": (0.25, 0.5),
    "rudder_chord_fraction": (0.25, 0.5),
    "thrust_to_weight": (0.4, 1.2),
    "propeller_diameter_fraction": (0.15, 0.3),
    "pod_mass_fraction": (0.4, 0.7),
    "cruise_lift_coefficient": (0.3, 0.6),
}


def sample_design(archetype: type[Design], key: jax.Array, ranges=None) -> Design:
    """Draw one design uniformly within the archetype's ranges (a dict overrides ranges)."""

    base = FLYING_WING_RANGES if archetype is FlyingWingDesign else CONVENTIONAL_RANGES
    bounds = dict(base, **(ranges or {}))
    names = sorted(bounds)
    draws = jax.random.uniform(key, (len(names) + 1,))
    values = {}
    for index, field in enumerate(names):
        lo, hi = bounds[field]
        values[field] = float(lo + (hi - lo) * draws[index])
    if archetype is FlyingWingDesign:
        values["motors"] = "pusher" if float(draws[-1]) < 0.5 else "twin_tractor"
    else:
        values["tail"] = "conventional" if float(draws[-1]) < 0.7 else "v_tail"
    return archetype(**values)


def sample_designs(
    archetype: type[Design], key: jax.Array, count: int, ranges=None
) -> list[Design]:
    return [sample_design(archetype, k, ranges) for k in jax.random.split(key, count)]


__all__ = [
    "CONVENTIONAL_RANGES",
    "ConventionalDesign",
    "Design",
    "DesignReport",
    "FLYING_WING_RANGES",
    "FlyingWingDesign",
    "control_authority",
    "conventional_spec",
    "cruise_lift_coefficient",
    "cruise_speed",
    "design_spec",
    "flap_effectiveness",
    "flap_moment_coefficient",
    "flying_wing_spec",
    "induced_drag_factor",
    "replace",
    "sample_design",
    "sample_designs",
    "validate_design",
    "wing_lift_slope",
]
