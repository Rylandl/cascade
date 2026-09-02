"""Fit the component-panel Skywalker X8 to the published whole-aircraft coefficient table.

The scientific question: can a component-panel model of the X8 flying wing, with its static
coefficients fitted to the published coefficient polynomial (Gryte et al., ICUAS 2018, wind-tunnel
column), reproduce the published rate derivatives (C_lp, C_mq, C_nr, C_Lq, C_Yp, C_Yr, C_lr, C_np)
from geometry alone? Rate derivatives are never fitted; they are the test.

This script fits the shared per-panel static coefficients of ``skywalker_x8_panels.toml`` (lift,
drag, and pitching-moment polynomials, plus the outer panels' flap terms and the winglets' lift
slope and zero drag) so that ``aerodynamic_sweep`` of the panel model matches the body-block sweep
of ``skywalker_x8_spec()`` over an alpha/beta/aileron/elevator grid at 18 m/s and zero rates.
Positions and everything that affects rate derivatives are left untouched. It then evaluates both
models' rate derivatives from small body rates and reports the comparison; the panel numbers are
predictions from geometry, reported honestly whatever they are.

Run ``uv run python scripts/fit_x8_panels.py``. It rewrites the fitted values into
``src/cascade/aircraft/skywalker_x8_panels.toml``. See ``docs/skywalker-x8.md``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from cascade.analysis import aerodynamic_sweep
from cascade.model import AircraftModel
from cascade.reference import skywalker_x8_panels_spec, skywalker_x8_spec
from cascade.spec import AircraftSpec, save_aircraft_spec
from cascade.state import ControlInput

TOML_PATH = Path(__file__).resolve().parent.parent / "src/cascade/aircraft/skywalker_x8_panels.toml"

COEFFICIENT_NAMES = ("CX", "CY", "CZ", "Cl", "Cm", "Cn")

# Fixed narrative half of the spec description; the fit-results half is appended fresh on every
# run (from this constant, not from whatever the file currently contains) so re-running the
# script is idempotent instead of accumulating duplicate suffixes.
BASE_DESCRIPTION = (
    "Skywalker X8 flying wing assembled from component surfaces (no [body] table). Geometry: "
    "span 2.1 m, reference area 0.75 m2, MAC 0.357 m, 30 deg leading-edge/quarter-chord sweep, "
    "CG ~0.44 m aft of the nose, elevons on the outer panels, large tip winglets acting as "
    "vertical stabilizers (Gryte et al. ICUAS 2018; Winter et al. EUCASS 2019; Reinhardt et al. "
    "ICUAS 2022). Mass, inertia, reference block, and propeller are copied from "
    "skywalker_x8.toml. Static per-panel coefficients are fitted by scripts/fit_x8_panels.py to "
    "reproduce the published body-block static polynomial (Gryte et al. wind-tunnel column); "
    "positions and all rate-affecting quantities are geometry, not fitted, so C_lp, C_mq, C_nr, "
    "C_Lq, C_Yp, C_Yr, C_lr, C_np are predictions to compare against the published XFLR5 values. "
    "See docs/skywalker-x8.md for the fit residuals and rate-derivative comparison."
)

# Panel index groups in the [[surfaces]] order of skywalker_x8_panels.toml.
CENTER = 0
INNER = (1, 2)
OUTER = (3, 4)
WINGLET = (5, 6)

N_PARAMS = 23


def fit_grid() -> tuple[jax.Array, jax.Array, ControlInput]:
    """Alpha in [-8, 10] deg (7), beta in [-8, 8] deg (5), aileron/elevator in {-0.2, 0, 0.2}."""

    alpha_deg = np.linspace(-8.0, 10.0, 7)
    beta_deg = np.linspace(-8.0, 8.0, 5)
    channel_vals = np.array([-0.2, 0.0, 0.2])
    combinations = np.array(
        [
            (a, b, aileron, elevator)
            for a in np.deg2rad(alpha_deg)
            for b in np.deg2rad(beta_deg)
            for aileron in channel_vals
            for elevator in channel_vals
        ]
    )
    alpha, beta, aileron, elevator = (jnp.asarray(column) for column in combinations.T)
    size = alpha.shape[0]
    control = ControlInput(
        propeller=jnp.zeros((size, 1)), channel=jnp.stack((aileron, elevator), axis=-1)
    )
    return alpha, beta, control


def target_coefficients(alpha: jax.Array, beta: jax.Array, control: ControlInput) -> jax.Array:
    """The published body-block coefficients over the fit grid, at 18 m/s and zero rates."""

    target_model = skywalker_x8_spec().to_model()
    sweep = aerodynamic_sweep(
        target_model, alpha, airspeed_m_s=18.0, sideslip_rad=beta, control=control
    )
    return jnp.concatenate((sweep.force_coefficient_body, sweep.moment_coefficient_body), axis=-1)


def apply_params(base_model: AircraftModel, params: jax.Array) -> AircraftModel:
    """Replace the fitted coefficient leaves of the panel surfaces, keeping the static topology."""

    (
        c_lift0,
        c_log_a,
        c_log_d0,
        c_log_k,
        c_cm0,
        c_cma,
        i_lift0,
        i_log_a,
        i_log_d0,
        i_log_k,
        i_cm0,
        i_cma,
        o_lift0,
        o_log_a,
        o_log_d0,
        o_log_k,
        o_cm0,
        o_cma,
        o_log_tau,
        o_cmflap,
        o_log_dflap,
        w_log_a,
        w_log_d0,
    ) = params

    surfaces = base_model.surfaces
    lift0 = surfaces.lift_coefficient_zero.at[CENTER].set(c_lift0)
    lift0 = lift0.at[INNER[0] : INNER[1] + 1].set(i_lift0).at[OUTER[0] : OUTER[1] + 1].set(o_lift0)

    slope = surfaces.lift_curve_slope.at[CENTER].set(jnp.exp(c_log_a))
    slope = (
        slope.at[INNER[0] : INNER[1] + 1]
        .set(jnp.exp(i_log_a))
        .at[OUTER[0] : OUTER[1] + 1]
        .set(jnp.exp(o_log_a))
        .at[WINGLET[0] : WINGLET[1] + 1]
        .set(jnp.exp(w_log_a))
    )

    drag0 = surfaces.drag_coefficient_zero.at[CENTER].set(jnp.exp(c_log_d0))
    drag0 = (
        drag0.at[INNER[0] : INNER[1] + 1]
        .set(jnp.exp(i_log_d0))
        .at[OUTER[0] : OUTER[1] + 1]
        .set(jnp.exp(o_log_d0))
        .at[WINGLET[0] : WINGLET[1] + 1]
        .set(jnp.exp(w_log_d0))
    )

    induced = surfaces.induced_drag_factor.at[CENTER].set(jnp.exp(c_log_k))
    induced = (
        induced.at[INNER[0] : INNER[1] + 1]
        .set(jnp.exp(i_log_k))
        .at[OUTER[0] : OUTER[1] + 1]
        .set(jnp.exp(o_log_k))
    )

    cm0 = surfaces.moment_coefficient_zero.at[CENTER].set(c_cm0)
    cm0 = cm0.at[INNER[0] : INNER[1] + 1].set(i_cm0).at[OUTER[0] : OUTER[1] + 1].set(o_cm0)

    cma = surfaces.moment_coefficient_alpha.at[CENTER].set(c_cma)
    cma = cma.at[INNER[0] : INNER[1] + 1].set(i_cma).at[OUTER[0] : OUTER[1] + 1].set(o_cma)

    tau = surfaces.flap_effectiveness.at[OUTER[0] : OUTER[1] + 1].set(jnp.exp(o_log_tau))
    cmflap = surfaces.moment_coefficient_flap.at[OUTER[0] : OUTER[1] + 1].set(o_cmflap)
    dflap = surfaces.drag_coefficient_flap.at[OUTER[0] : OUTER[1] + 1].set(jnp.exp(o_log_dflap))

    new_surfaces = surfaces._replace(
        lift_coefficient_zero=lift0,
        lift_curve_slope=slope,
        drag_coefficient_zero=drag0,
        induced_drag_factor=induced,
        moment_coefficient_zero=cm0,
        moment_coefficient_alpha=cma,
        flap_effectiveness=tau,
        moment_coefficient_flap=cmflap,
        drag_coefficient_flap=dflap,
    )
    return base_model._replace(surfaces=new_surfaces)


# Regularization weight, in each parameter's own (possibly log) units: a panel's own lift0/cm0
# feed C_m both directly and through moment-arm times force, so several panels can trade off
# against each other at near-equal data residual, and an unregularized bounded solve saturates
# at whatever box it is given (checked at three box widths; all three pinned most parameters to
# an edge). A small quadratic pull toward the initial guess picks the minimum-norm point among
# those near-equal-residual solutions instead, keeping every panel's fitted coefficients close to
# an ordinary small-UAV wing section unless the data genuinely demands otherwise.
REGULARIZATION_WEIGHT = 0.35


def _residual(
    params: jax.Array,
    base_model: AircraftModel,
    alpha: jax.Array,
    beta: jax.Array,
    control: ControlInput,
    target: jax.Array,
    anchor: jax.Array,
) -> jax.Array:
    model = apply_params(base_model, params)
    sweep = aerodynamic_sweep(model, alpha, airspeed_m_s=18.0, sideslip_rad=beta, control=control)
    candidate = jnp.concatenate(
        (sweep.force_coefficient_body, sweep.moment_coefficient_body), axis=-1
    )
    data_residual = (candidate - target).reshape(-1)
    prior_residual = REGULARIZATION_WEIGHT * (params - anchor)
    return jnp.concatenate((data_residual, prior_residual))


_compiled_residual = jax.jit(_residual)
_compiled_jacobian = jax.jit(jax.jacfwd(_residual, argnums=0))


def initial_guess() -> np.ndarray:
    return np.array(
        [
            0.05,
            math.log(4.0),
            math.log(0.02),
            math.log(0.05),
            0.0,
            -0.05,  # center
            0.05,
            math.log(4.5),
            math.log(0.02),
            math.log(0.05),
            0.0,
            -0.05,  # inner
            0.05,
            math.log(4.0),
            math.log(0.02),
            math.log(0.05),
            0.0,
            -0.05,  # outer
            math.log(0.5),
            -0.3,
            math.log(0.08),  # outer flap
            math.log(3.6),
            math.log(0.02),  # winglet
        ]
    )


def parameter_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Physically plausible ranges for every parameter, in its own (possibly log) space.

    The fit grid alone under-determines the 23 free coefficients: a panel's lift0/lift-slope feed
    C_m both directly (its own moment coefficients) and indirectly (moment-arm times force at its
    position), so several panels can trade off against each other at near-equal residual. An
    unbounded solve runs away to unphysical values (a lift-curve slope over 10 / rad, a flap
    effectiveness above 1, an intrinsic moment-curve slope over 1 / rad — an order of magnitude
    past any of Cascade's other, physically identified surfaces). Bounding each parameter to the
    range spanned by Cascade's other surface specifications (`aerobatic_reference.toml`) keeps
    every panel's own airfoil-like coefficients plausible while still fitting the data within that
    range; both endpoints of an exp parametrized entry stay positive, so this composes with the
    exp reparametrization rather than replacing it.
    """

    lift0 = (-0.3, 0.3)
    log_slope = (math.log(1.5), math.log(2.0 * math.pi))  # 2 pi: thin-airfoil theory's own ceiling
    log_drag0 = (math.log(0.005), math.log(0.08))
    log_induced = (math.log(0.01), math.log(0.2))
    cm0 = (-0.3, 0.3)
    cma = (-1.5, 1.5)
    panel = (lift0, log_slope, log_drag0, log_induced, cm0, cma)
    outer_extra = ((math.log(0.2), math.log(1.0)), (-1.0, 1.0), (math.log(0.01), math.log(0.3)))
    winglet = ((math.log(1.5), math.log(6.0)), (math.log(0.005), math.log(0.08)))

    bounds = panel + panel + panel + outer_extra + winglet
    lower = np.array([pair[0] for pair in bounds])
    upper = np.array([pair[1] for pair in bounds])
    return lower, upper


def fit() -> tuple[np.ndarray, np.ndarray, AircraftModel]:
    alpha, beta, control = fit_grid()
    target = target_coefficients(alpha, beta, control)
    base_model = skywalker_x8_panels_spec().to_model()
    anchor = jnp.asarray(initial_guess())

    def scipy_residual(x: np.ndarray) -> np.ndarray:
        value = _compiled_residual(jnp.asarray(x), base_model, alpha, beta, control, target, anchor)
        return np.asarray(jax.device_get(value), dtype=float)

    def scipy_jacobian(x: np.ndarray) -> np.ndarray:
        value = _compiled_jacobian(jnp.asarray(x), base_model, alpha, beta, control, target, anchor)
        return np.asarray(jax.device_get(value), dtype=float)

    lower, upper = parameter_bounds()
    result = least_squares(
        scipy_residual,
        initial_guess(),
        jac=scipy_jacobian,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=500,
    )
    fitted_model = apply_params(base_model, jnp.asarray(result.x))
    return result.x, result.fun, fitted_model


def residual_rms_table(residual: np.ndarray) -> dict[str, float]:
    """RMS per coefficient over the data residuals only, dropping the trailing prior residuals."""

    data_residual = residual[:-N_PARAMS]
    per_point = data_residual.reshape(-1, len(COEFFICIENT_NAMES))
    return {
        name: float(np.sqrt(np.mean(np.square(per_point[:, index]))))
        for index, name in enumerate(COEFFICIENT_NAMES)
    }


def write_fitted_spec(params: np.ndarray, rms: dict[str, float]) -> AircraftSpec:
    (
        c_lift0,
        c_log_a,
        c_log_d0,
        c_log_k,
        c_cm0,
        c_cma,
        i_lift0,
        i_log_a,
        i_log_d0,
        i_log_k,
        i_cm0,
        i_cma,
        o_lift0,
        o_log_a,
        o_log_d0,
        o_log_k,
        o_cm0,
        o_cma,
        o_log_tau,
        o_cmflap,
        o_log_dflap,
        w_log_a,
        w_log_d0,
    ) = (float(value) for value in params)

    updates = {
        "center_body": dict(
            lift_coefficient_zero=c_lift0,
            lift_curve_slope_rad=math.exp(c_log_a),
            drag_coefficient_zero=math.exp(c_log_d0),
            induced_drag_factor=math.exp(c_log_k),
            moment_coefficient_zero=c_cm0,
            moment_coefficient_alpha_rad=c_cma,
        ),
    }
    inner = dict(
        lift_coefficient_zero=i_lift0,
        lift_curve_slope_rad=math.exp(i_log_a),
        drag_coefficient_zero=math.exp(i_log_d0),
        induced_drag_factor=math.exp(i_log_k),
        moment_coefficient_zero=i_cm0,
        moment_coefficient_alpha_rad=i_cma,
    )
    outer = dict(
        lift_coefficient_zero=o_lift0,
        lift_curve_slope_rad=math.exp(o_log_a),
        drag_coefficient_zero=math.exp(o_log_d0),
        induced_drag_factor=math.exp(o_log_k),
        moment_coefficient_zero=o_cm0,
        moment_coefficient_alpha_rad=o_cma,
        flap_effectiveness=math.exp(o_log_tau),
        moment_coefficient_flap_rad=o_cmflap,
        drag_coefficient_flap_rad2=math.exp(o_log_dflap),
    )
    winglet = dict(lift_curve_slope_rad=math.exp(w_log_a), drag_coefficient_zero=math.exp(w_log_d0))
    updates.update(
        left_inner=inner,
        right_inner=inner,
        left_outer=outer,
        right_outer=outer,
        left_winglet=winglet,
        right_winglet=winglet,
    )

    spec = skywalker_x8_panels_spec()
    surfaces = tuple(replace(surface, **updates[surface.name]) for surface in spec.surfaces)
    rms_text = ", ".join(f"{name} {value:.4f}" for name, value in rms.items())
    description = f"{BASE_DESCRIPTION} Fit residual RMS: {rms_text}."
    return AircraftSpec(
        name=spec.name,
        description=description,
        mass_kg=spec.mass_kg,
        inertia_kg_m2=spec.inertia_kg_m2,
        reference_area_m2=spec.reference_area_m2,
        reference_chord_m=spec.reference_chord_m,
        reference_span_m=spec.reference_span_m,
        control_channels=spec.control_channels,
        surfaces=surfaces,
        propellers=spec.propellers,
        body=spec.body,
        schema_version=spec.schema_version,
    )


PUBLISHED_RATES = {
    "C_lp": -0.404,
    "C_lr": 0.0555,
    "C_np": 0.00437,
    "C_nr": -0.012,
    "C_Yp": -0.137,
    "C_Yr": 0.0839,
    "C_Lq": 3.87,
    "C_mq": -1.3,
}


def wind_axis_lift(force_coefficient_body: jax.Array, alpha: float) -> jax.Array:
    """C_L from body-axis C_X, C_Z at the (small) sideslip-free angle of attack used here."""

    axial, normal = force_coefficient_body[..., 0], force_coefficient_body[..., 2]
    return axial * jnp.sin(alpha) - normal * jnp.cos(alpha)


def rate_derivatives(model: AircraftModel, alpha_rad: float, speed: float, rate: float) -> dict:
    """Non-dimensional rate derivatives from one-at-a-time p, q, r perturbations."""

    b, c = 2.1, 0.357
    baseline = aerodynamic_sweep(model, jnp.asarray(alpha_rad), airspeed_m_s=speed)
    cl0, cm0, cn0 = (float(v) for v in baseline.moment_coefficient_body)
    cy0 = float(baseline.force_coefficient_body[1])
    CL0 = float(wind_axis_lift(baseline.force_coefficient_body, alpha_rad))

    def perturbed(p: float = 0.0, q: float = 0.0, r: float = 0.0):
        sweep = aerodynamic_sweep(
            model,
            jnp.asarray(alpha_rad),
            airspeed_m_s=speed,
            angular_velocity_rad_s=jnp.array([p, q, r]),
        )
        cl, cm, cn = (float(v) for v in sweep.moment_coefficient_body)
        cy = float(sweep.force_coefficient_body[1])
        CL = float(wind_axis_lift(sweep.force_coefficient_body, alpha_rad))
        return cl, cm, cn, cy, CL

    p_hat = b * rate / (2.0 * speed)
    q_hat = c * rate / (2.0 * speed)
    r_hat = b * rate / (2.0 * speed)

    cl_p, _, cn_p, cy_p, _ = perturbed(p=rate)
    _, cm_q, _, _, CL_q = perturbed(q=rate)
    cl_r, _, cn_r, cy_r, _ = perturbed(r=rate)

    return {
        "C_lp": (cl_p - cl0) / p_hat,
        "C_np": (cn_p - cn0) / p_hat,
        "C_Yp": (cy_p - cy0) / p_hat,
        "C_mq": (cm_q - cm0) / q_hat,
        "C_Lq": (CL_q - CL0) / q_hat,
        "C_lr": (cl_r - cl0) / r_hat,
        "C_nr": (cn_r - cn0) / r_hat,
        "C_Yr": (cy_r - cy0) / r_hat,
    }


def main() -> None:
    params, residual, fitted_model = fit()
    rms = residual_rms_table(residual)

    print("Fit residual RMS per coefficient (candidate - target, over the 315-point grid):")
    print("  " + "  ".join(f"{name}: {value:.4f}" for name, value in rms.items()))

    fitted_spec = write_fitted_spec(params, rms)
    save_aircraft_spec(fitted_spec, TOML_PATH)
    print(f"\nWrote fitted coefficients to {TOML_PATH}")

    target_model = skywalker_x8_spec().to_model()
    alpha_rad, speed, rate = math.radians(3.0), 18.0, 0.2
    target_rates = rate_derivatives(target_model, alpha_rad, speed, rate)
    panel_rates = rate_derivatives(fitted_model, alpha_rad, speed, rate)

    print("\nRate-derivative comparison at alpha=3 deg, 18 m/s, p=q=r=0.2 rad/s (one at a time):")
    header = f"{'derivative':>10s}  {'published':>10s}  {'target model':>13s}  {'panel model':>12s}"
    print(header)
    for name in PUBLISHED_RATES:
        print(
            f"{name:>10s}  {PUBLISHED_RATES[name]:10.4f}  {target_rates[name]:13.4f}  "
            f"{panel_rates[name]:12.4f}"
        )


if __name__ == "__main__":
    main()
