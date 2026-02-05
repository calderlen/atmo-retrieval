import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def isothermal_profile(art: object, T0: float) -> jnp.ndarray:
    return T0 * jnp.ones_like(art.pressure)

def numpyro_isothermal(art: object, T_low: float, T_high: float) -> jnp.ndarray:
    T0 = numpyro.sample("T0", dist.Uniform(T_low, T_high))
    return isothermal_profile(art, T0)


def gradient_profile(art: object, T_bottom: float, T_top: float) -> jnp.ndarray:
    log_p = jnp.log10(art.pressure)
    log_p_min, log_p_max = log_p.min(), log_p.max()
    # Linear interpolation in log-pressure
    frac = (log_p - log_p_min) / (log_p_max - log_p_min)
    return T_bottom + (T_top - T_bottom) * frac


def numpyro_gradient(art: object, T_low: float, T_high: float) -> jnp.ndarray:
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
    log_p = jnp.log10(art.pressure)
    log_p_trans = jnp.log10(P_trans)

    alpha = (log_p - log_p_trans) / delta_P
    f_transition = 0.5 * (1.0 + jnp.tanh(alpha))

    Tarr = T_high + (T_deep - T_high) * f_transition
    return Tarr

def numpyro_madhu_seager(art: object, T_low: float, T_high: float) -> jnp.ndarray:
    T_deep = numpyro.sample("T_deep", dist.Uniform(T_low, T_high))
    T_high = numpyro.sample("T_high", dist.Uniform(T_low, T_high))
    log_P_trans = numpyro.sample("log_P_trans", dist.Uniform(-8, 2))
    delta_P = numpyro.sample("delta_P", dist.Uniform(0.1, 3.0))

    P_trans = 10**log_P_trans
    return madhu_seager_profile(art, T_deep, T_high, P_trans, delta_P)




def free_temperature_profile(
    art: object,
    n_layers: int = 5,
    T_low: float = 1000,
    T_high: float = 4000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
) -> jnp.ndarray:
    p_ref = _validate_pressure_bar(pressure_bar)
    p_eval = _validate_pressure_bar(pressure_eval_bar)
    T_knots = jnp.asarray(T_knots)
    if T_knots.ndim != 1:
        raise ValueError("T_knots must be 1D")
    if T_knots.size < 3:
        raise ValueError("need at least 3 knots to define second differences")

    logp_min = jnp.log10(jnp.min(p_ref))
    logp_max = jnp.log10(jnp.max(p_ref))
    logp_knots = jnp.linspace(logp_min, logp_max, T_knots.size)

    logp_eval = jnp.log10(p_eval)
    T_eval = jnp.interp(logp_eval, logp_knots, T_knots)
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

    p_bar = _validate_pressure_bar(art.pressure)

    # Knot temperatures
    T_knots = jnp.stack(
        [numpyro.sample(f"T_{i}", dist.Uniform(T_low, T_high)) for i in range(n_knots)]
    )

    # Smoothness hyperparameter γ
    gamma = numpyro.sample("gamma", dist.InverseGamma(inv_gamma_a, inv_gamma_b))

    # Discrete second differences on knots
    d2 = T_knots[2:] - 2.0 * T_knots[1:-1] + T_knots[:-2]
    n = d2.size

    # log p(T | γ) up to an additive constant
    logp = -0.5 * gamma * jnp.sum(d2**2) + 0.5 * n * jnp.log(gamma)
    numpyro.factor("tp_smoothness", logp)

    # Evaluate directly on ART grid (no n_fine)
    Tarr = pspline_knots_profile_on_grid(
        pressure_bar=p_bar,
        T_knots=T_knots,
        pressure_eval_bar=p_bar,
    )
    return Tarr

def _gp_kernel_rbf(x: jnp.ndarray, amp: jnp.ndarray, ell: jnp.ndarray) -> jnp.ndarray:
    dx = x[:, None] - x[None, :]
    return (amp**2) * jnp.exp(-0.5 * (dx / jnp.clip(ell, 1e-12, None)) ** 2)


def _gp_kernel_matern32(x: jnp.ndarray, amp: jnp.ndarray, ell: jnp.ndarray) -> jnp.ndarray:
    dx = jnp.abs(x[:, None] - x[None, :])
    r = jnp.sqrt(3.0) * dx / jnp.clip(ell, 1e-12, None)
    return (amp**2) * (1.0 + r) * jnp.exp(-r)


def numpyro_gp_temperature(
    art: object,
    *,
    T_low: float = 400.0,
    T_high: float = 6000.0,
    mean_kind: str = "isothermal",   # {"isothermal","linear"}
    kernel: str = "matern32",        # {"matern32","rbf"}
    # Hyperpriors (tune these):
    amp_scale: float = 800.0,        # K, typical vertical variation
    ell_loc: float = 0.7,            # in dex of log10P, correlation length
    ell_scale: float = 0.5,          # in dex, spread of lengthscale prior
    jitter: float = 1.0e-6,          # numerical stability (in K^2 after scaling)
    obs_nugget: float = 1.0,         # K, tiny extra diagonal "nugget" for robustness
) -> jnp.ndarray:
    # x = log10(P/bar) on ART grid
    x = jnp.log10(jnp.asarray(art.pressure))

    # Optional: center/scale x for numerics (still interpretable because ell is in dex pre-scaling).
    # Keep a copy of original-dex coordinate for ell interpretation:
    x_dex = x
    x = (x - jnp.mean(x)) / (jnp.std(x) + 1e-12)

    # Mean function
    if mean_kind == "isothermal":
        T0 = numpyro.sample("T0", dist.Uniform(T_low, T_high))
        mu = T0 * jnp.ones_like(x)
    elif mean_kind == "linear":
        # Linear mean in logP (in dex space)
        T_mid = numpyro.sample("T_mid", dist.Uniform(T_low, T_high))
        dT_dlogP = numpyro.sample("dT_dlogP", dist.Normal(0.0, 1500.0))  # K per dex (broad)
        mu = T_mid + dT_dlogP * (x_dex - jnp.mean(x_dex))
    else:
        raise ValueError("mean_kind must be 'isothermal' or 'linear'.")

    # GP hyperparameters
    amp = numpyro.sample("gp_amp", dist.HalfNormal(amp_scale))  # K
    # Put ell in dex units (more interpretable), then map to standardized-x units
    ell_dex = numpyro.sample("gp_ell_dex", dist.LogNormal(jnp.log(ell_loc), ell_scale))
    # Convert dex-lengthscale to standardized-x lengthscale:
    # x_std = (x_dex - mean)/std, so Δx_std = Δx_dex / std_dex
    std_dex = jnp.std(x_dex) + 1e-12
    ell = ell_dex / std_dex

    # Kernel matrix
    if kernel == "rbf":
        K = _gp_kernel_rbf(x, amp, ell)
    elif kernel == "matern32":
        K = _gp_kernel_matern32(x, amp, ell)
    else:
        raise ValueError("kernel must be 'matern32' or 'rbf'.")

    # Diagonal stabilization:
    # - jitter is dimensionless here; multiply by amp^2 for scale-consistent stabilization
    # - obs_nugget adds a small K to diagonal to avoid near-singular cases in practice
    n = x.size
    K = K + (jitter * (amp**2) + (obs_nugget**2)) * jnp.eye(n)

    # Sample T ~ N(mu, K)
    Tarr = numpyro.sample("Tarr", dist.MultivariateNormal(loc=mu, covariance_matrix=K))

    # Optional hard bounds (keeps RT stable; remove if you want tails)
    Tarr = jnp.clip(Tarr, T_low, T_high)

    return Tarr
