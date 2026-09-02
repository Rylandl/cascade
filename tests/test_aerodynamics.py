import jax
import jax.numpy as jnp

from cascade.aerodynamics import (
    aerodynamic_coefficients,
    aerodynamics,
    deflected_surface_frames,
    propulsion,
    separation_derivative,
    surface_air_data,
)
from cascade.analysis import aerodynamic_sweep
from cascade.initialization import standard_environment, zero_state
from cascade.math import quaternion_rotate_inverse, rotation_y
from cascade.reference import aerobatic_reference
from cascade.state import ActuatorState, AeroState, ControlInput


def test_separated_flat_plate_coefficients_cover_full_envelope():
    model = aerobatic_reference()
    alpha = jnp.array([0.0, jnp.pi / 4.0, jnp.pi / 2.0, -jnp.pi / 4.0])[:, None]
    aero_state = AeroState(separation=jnp.ones((4, model.n_surfaces)))

    lift, drag, moment = aerodynamic_coefficients(model, aero_state, alpha, jnp.zeros_like(alpha))

    assert jnp.allclose(lift[0], 0.0, atol=1e-5)
    assert jnp.allclose(lift[2], 0.0, atol=1e-5)
    assert jnp.allclose(lift[1], -lift[3], atol=1e-5)
    assert jnp.all(drag >= 0.0)
    assert jnp.all(drag[2] > drag[0])
    assert jnp.allclose(moment, 0.0)

    # A flap on a separated plate shifts its incidence by the flap effectiveness times the angle.
    shifted = model.surfaces.flap_effectiveness * 0.3
    flapped, _, _ = aerodynamic_coefficients(model, aero_state, alpha, jnp.full_like(alpha, 0.3))
    rotated, _, _ = aerodynamic_coefficients(
        model, aero_state, alpha + shifted, jnp.zeros_like(alpha)
    )
    assert jnp.allclose(flapped[:, 0], rotated[:, 0], atol=1e-6)


def test_separation_equilibrium_and_lag_have_expected_direction():
    model = aerobatic_reference()
    environment = standard_environment(batch_shape=(2,))
    state = zero_state(model, batch_shape=(2,), forward_speed=12.0)
    alpha = jnp.array([0.0, 40.0 * jnp.pi / 180.0])
    velocity = jnp.stack(
        (12.0 * jnp.cos(alpha), jnp.zeros_like(alpha), 12.0 * jnp.sin(alpha)), axis=-1
    )
    state = state._replace(rigid_body=state.rigid_body._replace(velocity=velocity))
    air_velocity_body = quaternion_rotate_inverse(
        state.rigid_body.attitude, state.rigid_body.velocity
    )
    propeller_result = propulsion(model, state, environment, air_velocity_body)

    air, _ = surface_air_data(
        model, state, environment, air_velocity_body, propeller_result.induced_velocity
    )
    rate = separation_derivative(model, state.aero, air.separation_equilibrium)

    assert jnp.all(air.separation_equilibrium[0, :3] < 0.01)
    assert jnp.all(air.separation_equilibrium[1, :3] > 0.99)
    assert jnp.all(rate[1] > 0.0)


def test_propwash_reaches_mapped_tail_at_zero_freestream():
    model = aerobatic_reference()
    environment = standard_environment()
    state = zero_state(model)
    state = state._replace(
        actuators=ActuatorState(
            surface_deflection=state.actuators.surface_deflection,
            propeller_speed=jnp.array([700.0]),
        )
    )
    propeller_result = propulsion(model, state, environment, jnp.zeros(3))
    result = aerodynamics(
        model,
        state,
        environment,
        jnp.zeros(3),
        propeller_result.induced_velocity,
    )

    assert result.air.dynamic_pressure[2] > result.air.dynamic_pressure[0]
    assert result.air.dynamic_pressure[3] > result.air.dynamic_pressure[0]


def test_spanwise_flow_produces_opposing_crossflow_drag():
    model = aerobatic_reference()
    environment = standard_environment()
    state = zero_state(model)
    air_velocity_body = jnp.array([0.0, 10.0, 0.0])
    propeller_result = propulsion(model, state, environment, air_velocity_body)
    result = aerodynamics(
        model,
        state,
        environment,
        air_velocity_body,
        propeller_result.induced_velocity,
    )

    assert result.force_body[1] < 0.0


def test_roll_moment_is_monotonic_in_aileron_command():
    model = aerobatic_reference()
    aileron = jnp.linspace(0.0, 1.0, 11)
    control = ControlInput(
        propeller=jnp.zeros((11, 1)),
        channel=jnp.stack((aileron, jnp.zeros(11), jnp.zeros(11)), axis=-1),
    )
    sweep = aerodynamic_sweep(
        model, jnp.full((11,), jnp.deg2rad(5.0)), airspeed_m_s=12.0, control=control
    )

    assert jnp.all(jnp.diff(sweep.moment_coefficient_body[:, 0]) > 0.0)


def test_flap_surfaces_keep_their_frame_and_all_moving_surfaces_rotate():
    model = aerobatic_reference()
    deflection = jnp.full((model.n_surfaces,), 0.3)

    frames = deflected_surface_frames(model, deflection)

    # Wings carry flap-type ailerons: the geometric frame is unchanged by deflection.
    assert jnp.allclose(frames[0], model.surfaces.body_from_surface[0], atol=1e-6)
    # The horizontal tail is a stabilator and rotates as a whole.
    expected_tail = model.surfaces.body_from_surface[2] @ rotation_y(jnp.array(0.3))
    assert jnp.allclose(frames[2], expected_tail, atol=1e-6)


def test_flap_deflection_shifts_attached_lift_and_adds_nose_down_moment():
    model = aerobatic_reference()
    attached = AeroState(separation=jnp.zeros((model.n_surfaces,)))
    alpha = jnp.full((model.n_surfaces,), jnp.deg2rad(5.0))
    deflection = jnp.full((model.n_surfaces,), 0.2)

    lift_0, drag_0, moment_0 = aerodynamic_coefficients(model, attached, alpha, jnp.zeros(4))
    lift_1, drag_1, moment_1 = aerodynamic_coefficients(model, attached, alpha, deflection)

    surfaces = model.surfaces
    expected_lift_shift = surfaces.lift_curve_slope[0] * surfaces.flap_effectiveness[0] * 0.2
    # Relative tolerance: the tiny forced-separation blend at 5 deg scales the shift by ~1 - 5e-5.
    assert jnp.allclose(lift_1[0] - lift_0[0], expected_lift_shift, rtol=1e-3)
    assert moment_1[0] < moment_0[0]
    assert drag_1[0] > drag_0[0]
    # The all-moving tail has no flap share; at the same local alpha nothing changes.
    assert jnp.allclose(lift_1[2], lift_0[2])
    assert jnp.allclose(moment_1[2], moment_0[2])


def test_thrust_falls_with_airspeed_scales_with_density_and_windmills():
    model = aerobatic_reference()
    speed_max = float(model.actuators.propeller_speed_max[0])
    diameter = float(model.propellers.diameter[0])
    # The fixture map is the linear C_T(J) law: thrust crosses zero at V_a = J_0 n D.
    thrust_map = model.propellers.thrust_map[0]
    zero_thrust_speed = float(
        -thrust_map[1, 0] * speed_max / (2.0 * jnp.pi) * diameter / thrust_map[0, 1]
    )
    airspeeds = jnp.array([0.0, 6.0, 12.0, zero_thrust_speed, 30.0])
    state = zero_state(model, batch_shape=(5,))
    state = state._replace(
        actuators=ActuatorState(
            surface_deflection=state.actuators.surface_deflection,
            propeller_speed=jnp.full((5, 1), speed_max),
        )
    )
    air_velocity = jnp.stack((airspeeds, jnp.zeros(5), jnp.zeros(5)), axis=-1)

    result = propulsion(model, state, standard_environment(batch_shape=(5,)), air_velocity)
    dense = propulsion(
        model, state, standard_environment(batch_shape=(5,), density=2.45), air_velocity
    )

    thrust = result.thrust[:, 0]
    induced = result.induced_velocity[:, 0]
    assert jnp.allclose(thrust[0], 23.5, rtol=0.02)
    assert jnp.all(jnp.diff(thrust) < 0.0)
    assert jnp.abs(thrust[3]) < 1e-3 * thrust[0]
    assert thrust[4] < 0.0
    disk_area = 0.25 * jnp.pi * diameter**2
    assert jnp.allclose(induced[0], jnp.sqrt(thrust[0] / (2.0 * 1.225 * disk_area)), rtol=1e-3)
    assert induced[2] < induced[0]
    assert induced[4] < 0.0
    assert jnp.allclose(dense.thrust, 2.0 * result.thrust, rtol=1e-5)


def test_propulsion_gradients_are_finite_at_zero_speed_and_zero_airspeed():
    model = aerobatic_reference()
    environment = standard_environment()

    def total(propeller_speed, air_velocity):
        state = zero_state(model)
        state = state._replace(
            actuators=ActuatorState(
                surface_deflection=state.actuators.surface_deflection,
                propeller_speed=propeller_speed,
            )
        )
        result = propulsion(model, state, environment, air_velocity)
        return jnp.sum(result.thrust) + jnp.sum(result.induced_velocity)

    gradients = jax.grad(total, argnums=(0, 1))(jnp.zeros(1), jnp.zeros(3))
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradients)


def test_stalled_flap_load_keeps_the_attached_moment_arm():
    model = aerobatic_reference()
    surfaces = model.surfaces
    alpha = jnp.full((1, model.n_surfaces), jnp.pi / 4.0)
    aero_state = AeroState(separation=jnp.ones((1, model.n_surfaces)))
    deflection = jnp.full_like(alpha, 0.3)

    _, _, moment = aerodynamic_coefficients(model, aero_state, alpha, deflection)

    flap_share = (1.0 - surfaces.all_moving_fraction) * 0.3
    shifted = alpha + surfaces.flap_effectiveness * flap_share

    def normal(a):
        return surfaces.normal_force_coefficient * jnp.sin(a) * jnp.abs(jnp.sin(a))

    flap_lift_slope = surfaces.lift_curve_slope * surfaces.flap_effectiveness
    arm = -surfaces.moment_coefficient_flap / flap_lift_slope
    expected = -jnp.nan_to_num(arm) * (normal(shifted) - normal(alpha))
    assert jnp.allclose(moment, expected, atol=1e-6)
    # Trailing-edge-down on a flapped wing: more normal force on the flap, nose-down moment.
    flapped = (surfaces.all_moving_fraction < 1.0) & (surfaces.flap_effectiveness > 0.0)
    assert jnp.all(moment[0][flapped] < 0.0)
    # An all-moving surface has no flap share and therefore no flap moment.
    assert jnp.allclose(moment[0][~flapped], 0.0, atol=1e-6)


def test_downwash_map_reduces_a_downstream_surface_incidence_and_lift():
    from dataclasses import replace

    from cascade.dynamics import evaluate_dynamics
    from cascade.reference import aerobatic_reference_spec
    from cascade.spec import load_aircraft_spec, save_aircraft_spec

    spec = aerobatic_reference_spec()
    names = [s.name for s in spec.surfaces]
    tail = names.index("horizontal_tail")
    count = len(names)
    table = [[0.0] * count for _ in range(count)]
    for wing in ("left_wing", "right_wing"):
        table[tail][names.index(wing)] = 0.15
    washed = replace(spec, downwash_map=tuple(tuple(r) for r in table))
    plain, downwashed = spec.to_model(), washed.to_model()
    environment = standard_environment()
    state = zero_state(plain, altitude=50.0, forward_speed=12.0)
    state = state._replace(
        rigid_body=state.rigid_body._replace(
            attitude=jnp.array([0.0, jnp.sin(0.05), 0.0, jnp.cos(0.05)])
        )
    )
    control = ControlInput(propeller=jnp.array([0.5]), channel=jnp.zeros(3))
    before = evaluate_dynamics(plain, state, control, environment).aerodynamics
    after = evaluate_dynamics(downwashed, state, control, environment).aerodynamics
    # The tail's effective incidence and lift drop; the wings are untouched.
    assert float(after.air.angle_of_attack[tail]) < float(before.air.angle_of_attack[tail])
    assert float(after.force_per_surface[tail, 2]) > float(before.force_per_surface[tail, 2])
    for wing in ("left_wing", "right_wing"):
        index = names.index(wing)
        assert jnp.allclose(after.force_per_surface[index], before.force_per_surface[index])
    # Less tail download means a less nose-down moment.
    assert float(after.moment_body[1]) > float(before.moment_body[1])
    # The table survives the TOML round trip, and its absence means zeros.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "washed.toml"
        save_aircraft_spec(washed, path)
        loaded = load_aircraft_spec(path)
        assert loaded.downwash_map == washed.downwash_map
    assert float(jnp.max(jnp.abs(plain.surfaces.downwash_map))) == 0.0
