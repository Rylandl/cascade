from dataclasses import replace

import jax.numpy as jnp

from cascade.dynamics import evaluate_dynamics
from cascade.initialization import standard_environment, zero_control, zero_state
from cascade.model import (
    BodyModel,
    DragCoefficients,
    LateralCoefficients,
    LongitudinalCoefficients,
    validate_model,
)
from cascade.reference import aerobatic_reference, aerobatic_reference_spec
from cascade.spec import (
    BodySpec,
    DragCoefficientSpec,
    LateralCoefficientSpec,
    LongitudinalCoefficientSpec,
    load_aircraft_spec,
    save_aircraft_spec,
)
from cascade.state import ActuatorState, ControlInput

RHO = 1.225


def coefficient_only_model():
    """Reference geometry with all surface areas zeroed and a hand-picked body block."""

    base = aerobatic_reference()
    scalar = jnp.asarray
    body = BodyModel(
        lift=LongitudinalCoefficients(scalar(0.1), scalar(4.0), scalar(3.0), scalar(0.3)),
        drag=DragCoefficients(
            scalar(0.02),
            scalar(0.05),
            scalar(1.0),
            scalar(-0.01),
            scalar(0.2),
            scalar(0.0),
            scalar(0.06),
        ),
        side=LateralCoefficients(
            scalar(0.0), scalar(-0.2), scalar(-0.1), scalar(0.08), scalar(0.04), scalar(0.1)
        ),
        roll=LateralCoefficients(
            scalar(0.0), scalar(-0.08), scalar(-0.4), scalar(0.05), scalar(0.12), scalar(0.01)
        ),
        pitch=LongitudinalCoefficients(scalar(0.03), scalar(-0.2), scalar(-1.3), scalar(-0.2)),
        yaw=LateralCoefficients(
            scalar(0.0), scalar(0.03), scalar(0.004), scalar(-0.05), scalar(-0.003), scalar(-0.1)
        ),
        stall_angle=scalar(0.3),
        stall_width=scalar(0.02),
        normal_force_coefficient=scalar(2.0),
        pitch_flat_plate=scalar(-0.2),
        # aileron = left wing angle, elevator = tail angle, rudder = fin angle
        deflection_map=jnp.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ),
    )
    surfaces = base.surfaces._replace(area=jnp.zeros_like(base.surfaces.area))
    return validate_model(base._replace(surfaces=surfaces, body=body))


def evaluate(model, air_velocity_body, rates=(0.0, 0.0, 0.0), deflections=(0.0, 0.0, 0.0, 0.0)):
    state = zero_state(model)
    state = state._replace(
        rigid_body=state.rigid_body._replace(
            velocity=jnp.asarray(air_velocity_body), angular_velocity=jnp.asarray(rates)
        ),
        actuators=ActuatorState(
            surface_deflection=jnp.asarray(deflections),
            propeller_speed=state.actuators.propeller_speed,
        ),
    )
    return evaluate_dynamics(model, state, zero_control(model), standard_environment())


def test_body_block_matches_hand_computed_forces_and_moments():
    model = coefficient_only_model()
    area, chord, span = (
        float(model.reference_area),
        float(model.reference_chord),
        float(model.reference_span),
    )
    speed = 12.0
    dynamic_pressure = 0.5 * RHO * speed**2

    # Level flow at zero incidence: only the constant coefficients act.
    level = evaluate(model, (speed, 0.0, 0.0)).aerodynamics
    expected_force = dynamic_pressure * area * jnp.array([-0.02, 0.0, -0.1])
    expected_moment = dynamic_pressure * area * jnp.array([span * 0.0, chord * 0.03, span * 0.0])
    assert jnp.allclose(level.force_body, expected_force, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(level.moment_body, expected_moment, rtol=1e-4, atol=1e-4)

    # Pure angle of attack below stall: lift and drag rotate with the wind frame.
    alpha = 0.1
    pitched = evaluate(model, (speed * jnp.cos(alpha), 0.0, speed * jnp.sin(alpha))).aerodynamics
    lift = dynamic_pressure * area * (0.1 + 4.0 * alpha)
    drag = dynamic_pressure * area * (0.02 + 0.05 * alpha + 1.0 * alpha**2)
    expected = jnp.array(
        [
            -drag * jnp.cos(alpha) + lift * jnp.sin(alpha),
            0.0,
            -drag * jnp.sin(alpha) - lift * jnp.cos(alpha),
        ]
    )
    assert jnp.allclose(pitched.force_body, expected, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(
        pitched.moment_body[1], dynamic_pressure * area * chord * (0.03 - 0.2 * alpha), rtol=1e-3
    )

    # Positive sideslip (air from the right) gives a force to the left, plus drag along the
    # sideslip direction, and the lateral moments.
    beta = 0.05
    slipped = evaluate(model, (speed * jnp.cos(beta), speed * jnp.sin(beta), 0.0)).aerodynamics
    side = dynamic_pressure * area * (-0.2 * beta)
    slip_drag = dynamic_pressure * area * (0.02 - 0.01 * beta + 0.2 * beta**2)
    assert jnp.allclose(
        slipped.force_body[1], side * jnp.cos(beta) - slip_drag * jnp.sin(beta), rtol=1e-3
    )
    assert jnp.allclose(
        slipped.moment_body[0], dynamic_pressure * area * span * (-0.08 * beta), rtol=1e-3
    )
    assert jnp.allclose(
        slipped.moment_body[2], dynamic_pressure * area * span * (0.03 * beta), rtol=1e-3
    )

    # Pitch rate: the non-dimensional rate c q / (2 V_a) scales the damping derivative.
    rate = 0.5
    damped = evaluate(model, (speed, 0.0, 0.0), rates=(0.0, rate, 0.0)).aerodynamics
    pitch_increment = dynamic_pressure * area * chord * (-1.3 * chord * rate / (2.0 * speed))
    assert jnp.allclose(damped.moment_body[1] - level.moment_body[1], pitch_increment, rtol=1e-3)

    # Elevator through the deflection map: the tail surface angle is the generalized elevator.
    elevator = 0.1
    trimmed = evaluate(model, (speed, 0.0, 0.0), deflections=(0.0, 0.0, elevator, 0.0)).aerodynamics
    assert jnp.allclose(
        trimmed.force_body[2] - level.force_body[2],
        -dynamic_pressure * area * 0.3 * elevator,
        rtol=1e-3,
    )
    assert jnp.allclose(
        trimmed.moment_body[1] - level.moment_body[1],
        dynamic_pressure * area * chord * (-0.2 * elevator),
        rtol=1e-3,
    )


def test_body_block_is_finite_and_bounded_across_the_full_envelope():
    model = coefficient_only_model()
    alpha = jnp.linspace(-jnp.pi, jnp.pi, 73)
    velocity = jnp.stack((12.0 * jnp.cos(alpha), jnp.zeros(73), 12.0 * jnp.sin(alpha)), axis=-1)
    state = zero_state(model, batch_shape=(73,))
    state = state._replace(rigid_body=state.rigid_body._replace(velocity=velocity))

    environment = standard_environment(batch_shape=(73,))
    result = evaluate_dynamics(model, state, zero_control(model, batch_shape=(73,)), environment)

    assert jnp.all(jnp.isfinite(result.aerodynamics.force_body))
    assert jnp.all(jnp.isfinite(result.aerodynamics.moment_body))
    lift_coefficient = result.aerodynamics.body.coefficients[:, 0]
    assert jnp.all(jnp.abs(lift_coefficient) < 2.5)
    # Deep post-stall lift follows the flat plate: zero at 90 degrees, antisymmetric.
    assert jnp.abs(lift_coefficient[54]) < 0.05
    assert jnp.allclose(lift_coefficient[45], -lift_coefficient[27], atol=0.05)


def test_component_only_aircraft_has_a_silent_body_block():
    model = aerobatic_reference()
    assert all(
        bool(jnp.all(leaf == 0.0))
        for name, leaf in zip(BodyModel._fields, model.body, strict=True)
        if name not in ("stall_angle", "stall_width")
        for leaf in ([leaf] if not isinstance(leaf, tuple) else list(leaf))
    )
    control = ControlInput(propeller=jnp.array([0.5]), channel=jnp.array([0.2, -0.1, 0.05]))
    # Attached and deep post-stall flow alike: an empty body must contribute nothing.
    for alpha in (0.0, jnp.deg2rad(40.0)):
        state = zero_state(model)
        state = state._replace(
            rigid_body=state.rigid_body._replace(
                velocity=12.0 * jnp.array([jnp.cos(alpha), 0.0, jnp.sin(alpha)])
            )
        )
        result = evaluate_dynamics(model, state, control, standard_environment())
        assert jnp.allclose(result.aerodynamics.body.force_body, 0.0)
        assert jnp.allclose(result.aerodynamics.body.moment_body, 0.0)


def test_body_spec_round_trips_through_toml(tmp_path):
    base = aerobatic_reference_spec()
    body = BodySpec(
        lift=LongitudinalCoefficientSpec(0.1, 4.0, 3.0, 0.3),
        drag=DragCoefficientSpec(0.02, 0.05, 1.0, -0.01, 0.2, 0.0, 0.06),
        side=LateralCoefficientSpec(0.0, -0.2, -0.1, 0.08, 0.04, 0.1),
        roll=LateralCoefficientSpec(0.0, -0.08, -0.4, 0.05, 0.12, 0.01),
        pitch=LongitudinalCoefficientSpec(0.03, -0.2, -1.3, -0.2),
        yaw=LateralCoefficientSpec(0.0, 0.03, 0.004, -0.05, -0.003, -0.1),
        stall_angle_rad=0.3,
        stall_width_rad=0.02,
        normal_force_coefficient=2.0,
        pitch_flat_plate=-0.2,
        deflection_map=((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    spec = replace(base, body=body)
    path = tmp_path / "with_body.toml"

    save_aircraft_spec(spec, path)
    loaded = load_aircraft_spec(path)

    assert loaded == spec
    assert float(loaded.to_model().body.lift.alpha) == 4.0
