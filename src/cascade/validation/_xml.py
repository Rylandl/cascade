"""Independent aircraft equations expressed as native JSBSim XML functions.

Only specification data is shared with Cascade. No Cascade force, derivative, trim,
or integration function is used to compute the reference loads. This is a second
implementation of the same equations, not an independently identified aircraft.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from cascade.spec import AircraftSpec

FT = 0.3048
LBF = 4.4482216152605
SLUG = 14.59390293720636
RADIUS_M = 1e9
GRAVITY = 9.80665


@dataclass(frozen=True)
class Expr:
    xml: str

    def __add__(self, other):
        return op("sum", self, other)

    __radd__ = __add__

    def __mul__(self, other):
        return op("product", self, other)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return op("difference", self, other)

    def __rsub__(self, other):
        return op("difference", other, self)

    def __truediv__(self, other):
        return op("quotient", self, other)

    def __rtruediv__(self, other):
        return op("quotient", other, self)

    def __pow__(self, other):
        return op("pow", self, other)


def expr(value):
    return value if isinstance(value, Expr) else Expr(f"<value>{float(value):.17g}</value>")


def property_name(name):
    return re.sub(r"/(?=\d)", "/n", name)


def prop(name):
    return Expr(f"<property>{property_name(name)}</property>")


def op(name, *args):
    return Expr(f"<{name}>" + "".join(expr(x).xml for x in args) + f"</{name}>")


def smooth_abs(x):
    return op("sqrt", x * x + 1e-12)


def sigmoid(x):
    # Bound exponent arguments; this also avoids overflow on reversed-flow fixtures.
    return 1 / (1 + op("exp", op("min", 700, op("max", -700, -x))))


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def dot(a, b):
    return sum((expr(x) * y for x, y in zip(a, b, strict=True)), expr(0))


def cross(a, b):
    return [expr(a[j]) * b[k] - expr(a[k]) * b[j] for j, k in [(1, 2), (2, 0), (0, 1)]]


def add(a, b):
    return [expr(x) + y for x, y in zip(a, b, strict=True)]


class Equations:
    def __init__(self, root):
        self.root = root

    def save(self, name, value):
        name = property_name("cmp/" + name)
        element = ET.SubElement(self.root, "function", name=name)
        element.append(ET.fromstring(expr(value).xml))
        return prop(name)

    def vector(self, name, values):
        return [self.save(f"{name}/{i}", v) for i, v in enumerate(values)]

    def angles(self, name, velocity):
        u, v, w = velocity
        planar2 = self.save(name + "/planar2", u * u + w * w)
        speed = self.save(name + "/speed", op("sqrt", planar2 + v * v + 1e-8))
        alpha = self.save(name + "/alpha", op("atan2", w, u + 1e-3 * op("exp", -planar2 / 1e-6)))
        beta = self.save(name + "/beta", op("atan2", v, op("sqrt", planar2 + 1e-8)))
        return alpha, beta, speed, planar2


def surface_coefficients(e, name, s, alpha, delta, separation):
    flap = (1 - s.all_moving_fraction) * delta
    a = e.save(
        name + "/bounded-alpha", 3 * s.stall_angle_rad * tanh(alpha / (3 * s.stall_angle_rad))
    )
    cl = e.save(
        name + "/cl-attached",
        s.lift_coefficient_zero + s.lift_curve_slope_rad * (a + s.flap_effectiveness * flap),
    )
    cd = (
        s.drag_coefficient_zero
        + s.induced_drag_factor * cl**2
        + s.drag_coefficient_flap_rad2 * flap**2
    )
    cm = (
        s.moment_coefficient_zero
        + s.moment_coefficient_alpha_rad * a
        + s.moment_coefficient_flap_rad * flap
    )
    sine = e.save(name + "/sine", op("sin", alpha + s.flap_effectiveness * flap))
    normal = e.save(name + "/normal", s.normal_force_coefficient * sine * smooth_abs(sine))
    clean_sine = op("sin", alpha)
    clean_normal = s.normal_force_coefficient * clean_sine * smooth_abs(clean_sine)
    slope = s.lift_curve_slope_rad * s.flap_effectiveness
    arm = -s.moment_coefficient_flap_rad / slope if slope > 0 else 0
    cm_sep = -arm * (normal - clean_normal) - s.separated_center_of_pressure * clean_normal
    separated = e.save(
        name + "/blend",
        1
        - (1 - separation)
        * (1 - sigmoid((smooth_abs(alpha) - 2 * s.stall_angle_rad) / s.stall_width_rad)),
    )
    cosine = op("cos", alpha + s.flap_effectiveness * flap)
    return (
        e.save(name + "/cl", (1 - separated) * cl + separated * normal * cosine),
        e.save(
            name + "/cd",
            (1 - separated) * cd
            + separated
            * (
                s.normal_force_coefficient * smooth_abs(sine) ** 3
                + s.edge_drag_coefficient * cosine**2
            ),
        ),
        e.save(name + "/cm", (1 - separated) * cm + separated * cm_sep),
    )


def aircraft_xml(spec: AircraftSpec) -> str:
    """Generate body-axis SI equations, converting to JSBSim units only at the axes."""
    spec.validate()
    root = ET.Element("fdm_config", name=spec.name, version="2.0", release="ALPHA")
    metrics = ET.SubElement(root, "metrics")
    for tag, value, unit in [
        ("wingarea", spec.reference_area_m2, "M2"),
        ("wingspan", spec.reference_span_m, "M"),
        ("chord", spec.reference_chord_m, "M"),
    ]:
        ET.SubElement(metrics, tag, unit=unit).text = str(value)
    for name in ("AERORP", "EYEPOINT", "VRP"):
        loc = ET.SubElement(metrics, "location", name=name, unit="M")
        for axis in "xyz":
            ET.SubElement(loc, axis).text = "0"
    mass = ET.SubElement(root, "mass_balance", negated_crossproduct_inertia="true")
    # JSBSim's default XML convention includes the structural-to-body transform.
    for tag, i, j, sign in [
        ("ixx", 0, 0, 1),
        ("iyy", 1, 1, 1),
        ("izz", 2, 2, 1),
        ("ixy", 0, 1, -1),
        ("ixz", 0, 2, 1),
        ("iyz", 1, 2, -1),
    ]:
        ET.SubElement(mass, tag, unit="SLUG*FT2").text = str(
            sign * spec.inertia_kg_m2[i][j] / (SLUG * FT**2)
        )
    # JSBSim converts weight to mass using its fixed slug-to-pound constant.
    ET.SubElement(mass, "emptywt", unit="LBS").text = str(spec.mass_kg / SLUG * 32.174049)
    loc = ET.SubElement(mass, "location", name="CG", unit="M")
    for axis in "xyz":
        ET.SubElement(loc, axis).text = "0"
    ET.SubElement(root, "ground_reactions")
    inputs = ET.SubElement(root, "flight_control", name="comparison-inputs")
    for name in [
        "rho",
        *[f"surface/{i}" for i in range(len(spec.surfaces))],
        *[f"separation/{i}" for i in range(len(spec.surfaces))],
        *[f"motor/{i}" for i in range(len(spec.propellers))],
    ]:
        ET.SubElement(
            inputs, "property", value="1.225" if name == "rho" else "0"
        ).text = property_name("cmp/" + name)
    aero = ET.SubElement(root, "aerodynamics")
    e = Equations(aero)
    velocity = e.vector("velocity", [prop(f"velocities/{x}-aero-fps") * FT for x in "uvw"])
    rates = [prop(f"velocities/{x}-rad_sec") for x in "pqr"]
    rho = prop("cmp/rho")
    force = [expr(0)] * 3
    moment = [expr(0)] * 3
    wakes = []
    for i, p in enumerate(spec.propellers):
        name = f"prop/{i}"
        n = prop(f"cmp/motor/{i}") / (2 * math.pi)
        axial = e.save(
            name + "/axial", dot(add(velocity, cross(rates, p.position_m)), p.direction_body)
        )
        thrust_density = e.save(
            name + "/thrust-density",
            p.diameter_m**4
            * sum(
                (
                    p.thrust_map[j][k] * n ** (j + 1) * (axial / p.diameter_m) ** k
                    for j in range(2)
                    for k in range(3)
                ),
                expr(0),
            ),
        )
        f = e.vector(name + "/force", [rho * thrust_density * x for x in p.direction_body])
        torque = -p.spin_direction * rho * n**2 * p.diameter_m**5 * p.torque_coefficient_static
        force = add(force, f)
        moment = add(moment, add(cross(p.position_m, f), [torque * x for x in p.direction_body]))
        k = thrust_density / (0.5 * math.pi * p.diameter_m**2)
        wakes.append(
            e.save(
                name + "/wake",
                2
                * k
                / (op("sqrt", op("max", axial**2 + 4 * k, 0) + 1e-12) + smooth_abs(axial) + 1e-6),
            )
        )
    propulsion_force = e.vector("propulsion-force", force)
    propulsion_moment = e.vector("propulsion-moment", moment)
    force, moment = [expr(0)] * 3, [expr(0)] * 3
    frames, angles, pressures, local_velocities, upstream = [], [], [], [], []
    for i, s in enumerate(spec.surfaces):
        name = f"panel/{i}"
        delta = prop(f"cmp/surface/{i}")
        theta = s.all_moving_fraction * delta
        c, sn = op("cos", theta), op("sin", theta)
        ry = [[c, 0, sn], [0, 1, 0], [-sn, 0, c]]
        frame = [
            [
                e.save(
                    f"{name}/frame/{j}/{k}",
                    dot(s.body_from_surface[j], [ry[t][k] for t in range(3)]),
                )
                for k in range(3)
            ]
            for j in range(3)
        ]
        frames.append(frame)
        wash = [
            sum(
                (
                    wakes[j] * p.slipstream_weights[i] * p.direction_body[k]
                    for j, p in enumerate(spec.propellers)
                ),
                expr(0),
            )
            for k in range(3)
        ]
        vb = add(add(velocity, cross(rates, s.position_m)), wash)
        local = e.vector(
            name + "/local", [dot([frame[j][k] for j in range(3)], vb) for k in range(3)]
        )
        alpha, _, _, planar2 = e.angles(name, local)
        angles.append(alpha)
        pressures.append(e.save(name + "/q", 0.5 * rho * planar2))
        local_velocities.append(local)
        upstream.append(
            surface_coefficients(
                e, name + "/upstream", s, alpha, delta, prop(f"cmp/separation/{i}")
            )[0]
        )
    for i, s in enumerate(spec.surfaces):
        name = f"panel/{i}"
        downwash = dot(spec.downwash_map[i], upstream) if spec.downwash_map is not None else expr(0)
        alpha = e.save(name + "/effective-alpha", angles[i] - downwash)
        sep = prop(f"cmp/separation/{i}")
        equilibrium = e.save(
            f"equilibrium/{i}", sigmoid((smooth_abs(alpha) - s.stall_angle_rad) / s.stall_width_rad)
        )
        separating = (1 + tanh((equilibrium - sep) / 0.05)) * 0.5
        tau = (
            separating * s.separation_time_constant_s
            + (1 - separating) * s.reattachment_time_constant_s
        )
        e.save(f"separation-rate/{i}", (equilibrium - sep) / tau)
        cl, cd, cm = surface_coefficients(
            e, name + "/effective", s, alpha, prop(f"cmp/surface/{i}"), sep
        )
        scale = pressures[i] * s.area_m2
        f_local = [
            scale * (-cd * op("cos", alpha) + cl * op("sin", alpha)),
            -0.5
            * rho
            * s.area_m2
            * s.span_drag_coefficient
            * local_velocities[i][1]
            * smooth_abs(local_velocities[i][1]),
            scale * (-cd * op("sin", alpha) - cl * op("cos", alpha)),
        ]
        f = e.vector(name + "/force", [dot(row, f_local) for row in frames[i]])
        m = e.vector(
            name + "/moment",
            add(cross(s.position_m, f), [row[1] * scale * s.chord_m * cm for row in frames[i]]),
        )
        force, moment = add(force, f), add(moment, m)
    if spec.body is not None:
        b = spec.body
        alpha, beta, speed, _ = e.angles(
            "body", add(velocity, cross(rates, b.reference_position_m))
        )
        aileron, elevator, rudder = [
            dot(row, [prop(f"cmp/surface/{i}") for i in range(len(spec.surfaces))])
            for row in b.deflection_map
        ]
        attached = 1 - sigmoid((smooth_abs(alpha) - b.stall_angle_rad) / b.stall_width_rad)
        sn, cs = op("sin", alpha), op("cos", alpha)
        plate = b.normal_force_coefficient * sn * smooth_abs(sn)
        qhat = rates[1] * spec.reference_chord_m / (2 * speed)
        phat, rhat = [rates[j] * spec.reference_span_m / (2 * speed) for j in (0, 2)]
        cl = (
            attached * (b.lift.zero + b.lift.alpha_rad * alpha)
            + (1 - attached) * plate * cs
            + b.lift.elevator_rad * elevator
            + b.lift.q * qhat
        )
        cd = (
            attached * (b.drag.zero + b.drag.alpha_rad * alpha + b.drag.alpha_sq_rad2 * alpha**2)
            + (1 - attached) * b.normal_force_coefficient * smooth_abs(sn) ** 3
            + b.drag.beta_rad * beta
            + b.drag.beta_sq_rad2 * beta**2
            + b.drag.elevator_sq_rad2 * elevator**2
            + b.drag.q * qhat
        )

        def lateral(c):
            return (
                c.zero
                + c.beta_rad * beta
                + c.p * phat
                + c.r * rhat
                + c.aileron_rad * aileron
                + c.rudder_rad * rudder
            )

        cy = lateral(b.side)
        pitch = (
            attached * (b.pitch.zero + b.pitch.alpha_rad * alpha)
            + (1 - attached) * b.pitch_flat_plate * sn * smooth_abs(sn)
            + b.pitch.elevator_rad * elevator
            + b.pitch.q * qhat
        )
        qS = 0.5 * rho * speed**2 * spec.reference_area_m2
        # Resolve the orthonormal wind basis explicitly, including sideslip.
        drag_axis = [cs * op("cos", beta), op("sin", beta), sn * op("cos", beta)]
        side_axis = [-cs * op("sin", beta), op("cos", beta), -sn * op("sin", beta)]
        lift_axis = [sn, 0, -cs]
        f = e.vector(
            "body/force",
            [qS * dot([-cd, cy, cl], [drag_axis[j], side_axis[j], lift_axis[j]]) for j in range(3)],
        )
        m = e.vector(
            "body/moment",
            add(
                [
                    qS * spec.reference_span_m * lateral(b.roll),
                    qS * spec.reference_chord_m * pitch,
                    qS * spec.reference_span_m * lateral(b.yaw),
                ],
                cross(b.reference_position_m, f),
            ),
        )
        force, moment = add(force, f), add(moment, m)
    force = e.vector("aero-force", force)
    moment = e.vector("aero-moment", moment)
    total_force = e.vector("force", add(force, propulsion_force))
    total_moment = e.vector("moment", add(moment, propulsion_moment))
    # The harness applies both aero and propulsion in the aerodynamic axis section.
    for name, value, unit in zip(
        ("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        total_force + total_moment,
        [LBF] * 3 + [LBF * FT] * 3,
        strict=True,
    ):
        axis = ET.SubElement(aero, "axis", name=name)
        function = ET.SubElement(axis, "function", name="cmp/output/" + name)
        function.append(ET.fromstring((value / unit).xml))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def write_model(spec: AircraftSpec, directory: Path) -> Path:
    directory = Path(directory)
    folder = directory / "aircraft" / "comparison"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "comparison.xml"
    path.write_text(aircraft_xml(spec))
    planet = ET.Element("planet", name="nonrotating-flat-limit")
    for tag, value, unit in [
        ("semimajor_axis", RADIUS_M, "M"),
        ("semiminor_axis", RADIUS_M, "M"),
        ("rotation_rate", 0, "RAD/SEC"),
        ("GM", GRAVITY * (RADIUS_M + 1000) ** 2 / FT**3, "FT3/SEC2"),
    ]:
        ET.SubElement(planet, tag, unit=unit).text = str(value)
    ET.SubElement(planet, "J2").text = "0"
    (directory / "planet.xml").write_text(ET.tostring(planet, encoding="unicode"))
    return path
