"""Temperature-pressure profiles for ultra-hot Jupiters."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def isothermal_profile(art: object, T0: float) -> jnp.ndarray:
    """Isothermal temperature profile."""
    return T0 * jnp.ones_like(art.pressure)

def numpyro_isothermal(art: object, T_low: float, T_high: float) -> jnp.ndarray:
    """Isothermal profile with NumPyro sampling."""
    T0 = numpyro.sample("T0", dist.Uniform(T_low, T_high))
    return isothermal_profile(art, T0)


def gradient_profile(art: object, T_bottom: float, T_top: float) -> jnp.ndarray:
    """Linear temperature gradient from bottom to top of atmosphere."""
    log_p = jnp.log10(art.pressure)
    log_p_min, log_p_max = log_p.min(), log_p.max()
    # Linear interpolation in log-pressure
    frac = (log_p - log_p_min) / (log_p_max - log_p_min)
    return T_bottom + (T_top - T_bottom) * frac


def numpyro_gradient(art: object, T_low: float, T_high: float) -> jnp.ndarray:
    """Linear gradient profile with NumPyro sampling."""
    T_bottom = numpyro.sample("T_bottom", dist.Uniform(T_low, T_high))
    T_top = numpyro.sample("T_top", dist.Uniform(T_low, T_high))
    return gradient_profile(art, T_bottom, T_top)


def guillot_profile(
    pressure_bar: jnp.ndarray,
    g_cgs: float,
    Tirr: float,
    Tint: float,
    kappa_ir_cgs: float,
    gamma: float,
) -> jnp.ndarray:
    """Grey radiative-equilibrium profile (Guillot 2010 form).

    Physics convention:
      τ(P) = κ_IR * P / g with P in dyn/cm² (1 bar = 1e6 dyn/cm²), g in cm/s².

      T⁴ = (3/4) Tint⁴ (2/3 + τ)
         + (3/4) Tirr⁴ [ 2/3 + 1/(√3 γ) + (γ/√3 - 1/(√3 γ)) exp(-√3 γ τ) ].
    """
    P_cgs = pressure_bar * 1.0e6
    tau = kappa_ir_cgs * P_cgs / jnp.clip(g_cgs, 1.0e-20, None)

    sqrt3 = jnp.sqrt(3.0)
    exp_term = jnp.exp(-sqrt3 * gamma * tau)
    bracket = (
        (2.0 / 3.0)
        + (1.0 / (sqrt3 * gamma))
        + ((gamma / sqrt3) - (1.0 / (sqrt3 * gamma))) * exp_term
    )

    T4 = (3.0 / 4.0) * (Tint**4) * (2.0 / 3.0 + tau) + (3.0 / 4.0) * (Tirr**4) * bracket
    return jnp.power(jnp.clip(T4, 0.0, None), 0.25)


def madhu_seager_profile(
    art: object,
    T_deep: float,
    T_high: float,
    P_trans: float,
    delta_P: float,
) -> jnp.ndarray:
    """Madhusudhan & Seager (2009) smoothly-varying profile with inversions."""
    log_p = jnp.log10(art.pressure)
    log_p_trans = jnp.log10(P_trans)

    alpha = (log_p - log_p_trans) / delta_P
    f_transition = 0.5 * (1.0 + jnp.tanh(alpha))

    Tarr = T_high + (T_deep - T_high) * f_transition
    return Tarr

def numpyro_madhu_seager(art: object, T_low: float, T_high: float) -> jnp.ndarray:
    """Madhusudhan-Seager profile with NumPyro sampling."""
    T_deep = numpyro.sample("T_deep", dist.Uniform(T_low, T_high))
    T_high = numpyro.sample("T_high", dist.Uniform(T_low, T_high))
    log_P_trans = numpyro.sample("log_P_trans", dist.Uniform(-8, 2))
    delta_P = numpyro.sample("delta_P", dist.Uniform(0.1, 3.0))

    P_trans = 10**log_P_trans
    return madhu_seager_profile(art, T_deep, T_high, P_trans, delta_P)




# TODO: this requires strong priors and careful tuning to work

def free_temperature_profile(
    art: object,
    n_layers: int = 5,
    T_low: float = 1000,
    T_high: float = 4000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Free temperature profile with piecewise linear interpolation."""
    log_p = jnp.log10(art.pressure)
    log_p_nodes = jnp.linspace(log_p.min(), log_p.max(), n_layers)

    T_nodes = []
    for i in range(n_layers):
        T_i = numpyro.sample(f"T_node_{i}", dist.Uniform(T_low, T_high))
        T_nodes.append(T_i)
    T_nodes = jnp.array(T_nodes)

    Tarr = jnp.interp(log_p, log_p_nodes, T_nodes)
    return Tarr, T_nodes


def numpyro_free_temperature(
    art: object,
    n_layers: int = 5,
    T_low: float = 1000,
    T_high: float = 4000,
) -> jnp.ndarray:
    """Free temperature profile with NumPyro sampling."""
    return free_temperature_profile(art, n_layers, T_low, T_high)[0]


def _validate_pressure_bar(pressure_bar: jnp.ndarray) -> jnp.ndarray:
    p = jnp.asarray(pressure_bar)
    if p.ndim != 1:
        raise ValueError("pressure_bar must be 1D")
    if jnp.any(~jnp.isfinite(p)) or jnp.any(p <= 0.0):
        raise ValueError("pressure_bar must be finite and > 0 everywhere")
    return p


def pspline_knots_profile_on_grid(
    pressure_bar: jnp.ndarray,
    T_knots: jnp.ndarray,
    *,
    pressure_eval_bar: jnp.ndarray,
    n_knots: int,  # Pass explicitly to avoid JAX tracer issues
) -> jnp.ndarray:
    """Interpolate knot temperatures (even in log10 P) onto an arbitrary pressure grid.

    Knots are evenly spaced in log10(pressure_bar) range.
    Evaluated at log10(pressure_eval_bar).
    Interpolation is linear in log10(P) (JAX-friendly).

    Note: n_knots must be passed explicitly as a concrete int to avoid JAX abstract
    tracing issues during MCMC sampling (T_knots.shape[0] returns a tracer).
    """
    p_ref = _validate_pressure_bar(pressure_bar)
    p_eval = _validate_pressure_bar(pressure_eval_bar)

    # Ensure T_knots is 1D JAX array
    T_knots_arr = jnp.atleast_1d(jnp.asarray(T_knots))
    if T_knots_arr.ndim > 1:
        T_knots_arr = T_knots_arr.ravel()

    if n_knots < 3:
        raise ValueError("need at least 3 knots to define second differences")

    logp_min = jnp.log10(p_ref.min())
    logp_max = jnp.log10(p_ref.max())
    logp_knots = jnp.linspace(logp_min, logp_max, n_knots)  # Use concrete n_knots

    logp_eval = jnp.log10(p_eval)
    T_eval = jnp.interp(logp_eval, logp_knots, T_knots_arr)
    return T_eval


def numpyro_pspline_knots_on_art_grid(
    art: object,
    *,
    n_knots: int = 15, #must be >= 3
    T_low: float = 100.0,
    T_high: float = 6000.0,
    inv_gamma_a: float = 1.0,
    inv_gamma_b: float = 5.0e-5,
) -> jnp.ndarray:
    """Line+2015-style TP prior with ART-grid evaluation.

    Physics convention: pressure is in bar; we use log10(P) for knot placement
    and interpolation. Roughness penalty is on discrete 2nd differences of knot T.

    Returns
    -------
    Tarr : jnp.ndarray
        Temperature on the ART pressure grid (same length as art.pressure).
    """

    p_bar = _validate_pressure_bar(art.pressure)

    # Sample knot temperatures individually (like free_temperature_profile)
    T_knots = []
    for i in range(n_knots):
        T_i = numpyro.sample(f"T_knot_{i}", dist.Uniform(T_low, T_high))
        T_knots.append(T_i)

    # Stack into array - use jnp.stack instead of jnp.array for better tracer handling
    T_knots = jnp.stack(T_knots)

    # Smoothness hyperparameter γ
    gamma = numpyro.sample("gamma", dist.InverseGamma(inv_gamma_a, inv_gamma_b))

    # Discrete second differences on knots
    d2 = T_knots[2:] - 2.0 * T_knots[1:-1] + T_knots[:-2]

    # log p(T | γ) up to an additive constant
    # Use concrete n_knots instead of d2.shape[0] to avoid JAX tracer issues during MCMC
    n_d2 = n_knots - 2  # d2 has length n_knots - 2
    logp = -0.5 * gamma * jnp.sum(d2**2) + 0.5 * n_d2 * jnp.log(gamma)
    numpyro.factor("tp_smoothness", logp)

    # Inline interpolation instead of calling another function
    logp_min = jnp.log10(p_bar.min())
    logp_max = jnp.log10(p_bar.max())
    logp_knots = jnp.linspace(logp_min, logp_max, n_knots)
    logp_grid = jnp.log10(p_bar)

    Tarr = jnp.interp(logp_grid, logp_knots, T_knots)
    return Tarr
