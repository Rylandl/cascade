import jax
import jax.numpy as jnp
import pytest

from cascade.analysis import aerodynamic_sweep, velocity_from_air_angles
from cascade.reference import aerobatic_reference


def test_air_angle_conversion_uses_frd_sign_conventions():
    velocity = velocity_from_air_angles(
        jnp.array(10.0), jnp.deg2rad(jnp.array(30.0)), jnp.deg2rad(jnp.array(20.0))
    )

    assert velocity[0] > 0.0
    assert velocity[1] > 0.0
    assert velocity[2] > 0.0
    assert jnp.allclose(jnp.linalg.norm(velocity), 10.0)


def test_aerodynamic_sweep_is_finite_through_the_full_envelope():
    model = aerobatic_reference()
    alpha = jnp.linspace(-jnp.pi, jnp.pi, 73)
    sweep = aerodynamic_sweep(model, alpha, airspeed_m_s=12.0)

    assert sweep.force_coefficient_body.shape == (73, 3)
    assert sweep.moment_coefficient_body.shape == (73, 3)
    assert sweep.surface_separation.shape == (73, model.n_surfaces)
    assert jnp.all(jnp.isfinite(sweep.force_coefficient_body))
    assert jnp.all(jnp.isfinite(sweep.moment_coefficient_body))
    assert jnp.all((sweep.surface_separation >= 0.0) & (sweep.surface_separation <= 1.0))

    ninety_degrees = 54
    zero_degrees = 36
    # The wing and horizontal tail see 90-degree chordwise incidence. The vertical tail correctly
    # sees that same body-z flow along its span, so its separation coordinate need not match.
    assert jnp.all(sweep.surface_separation[ninety_degrees, :3] > 0.99)
    assert jnp.linalg.norm(sweep.force_coefficient_body[ninety_degrees]) > 5.0 * jnp.linalg.norm(
        sweep.force_coefficient_body[zero_degrees]
    )


def test_aerodynamic_sweep_rejects_zero_speed():
    with pytest.raises(ValueError, match="airspeed"):
        aerodynamic_sweep(aerobatic_reference(), jnp.array(0.0), airspeed_m_s=0.0)


def test_aerodynamic_sweep_is_differentiable_in_angle_of_attack():
    model = aerobatic_reference()

    def force_coefficient(alpha):
        return aerodynamic_sweep(model, alpha, airspeed_m_s=12.0).force_coefficient_body

    jacobian = jax.jacfwd(force_coefficient)(jnp.array(0.1))
    assert jacobian.shape == (3,)
    assert jnp.all(jnp.isfinite(jacobian))


def test_pitch_rate_damps_the_pitching_moment_of_the_aerobatic_reference():
    model = aerobatic_reference()
    alpha = jnp.deg2rad(jnp.array(3.0))
    baseline = aerodynamic_sweep(model, alpha, airspeed_m_s=15.0)
    pitching = aerodynamic_sweep(
        model,
        alpha,
        airspeed_m_s=15.0,
        angular_velocity_rad_s=jnp.array([0.0, 1.0, 0.0]),
    )

    # A positive pitch rate should add a damping (nose-down, negative) increment to C_m.
    assert pitching.moment_coefficient_body[1] < baseline.moment_coefficient_body[1]
