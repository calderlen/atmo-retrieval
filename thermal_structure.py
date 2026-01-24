"""Temperature-pressure profiles for ultra-hot Jupiters."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def isothermal_profile(art: object, T0: float) -> jnp.ndarray:
    """Isothermal temperature profile."""
    return T0 * jnp.ones_like(art.pressure)


def gradient_profile(art: object, T_btm: float, T_top: float) -> jnp.ndarray:
    """Linear temperature gradient in log(P) space."""
    log_p = jnp.log10(art.pressure)
    log_p_btm = jnp.log10(art.pressure[-1])
    log_p_top = jnp.log10(art.pressure[0])

    Tarr = T_top + (T_btm - T_top) * (log_p - log_p_top) / (log_p_btm - log_p_top)
    return Tarr


def guillot_profile(
    art: object,
    T_irr: float,
    T_int: float,
    kappa_ir: float,
    gamma: float,
) -> jnp.ndarray:
    """Guillot (2010) temperature profile with thermal inversion."""
    # Approximation using pressure as proxy for optical depth
    tau = art.pressure / 1e-3

    T4_eff = T_int**4 + T_irr**4 * (
        2.0 / 3.0 + 2.0 / 3.0 / gamma * (1.0 + gamma * tau / 2.0 - gamma * tau)
    )

    Tarr = jnp.power(jnp.clip(T4_eff, 0, None), 0.25)
    return Tarr


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


def free_temperature_profile(
    art: object,
    n_layers: int = 5,
    Tlow: float = 1000,
    Thigh: float = 4000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Free temperature profile with piecewise linear interpolation."""
    log_p = jnp.log10(art.pressure)
    log_p_nodes = jnp.linspace(log_p.min(), log_p.max(), n_layers)

    T_nodes = []
    for i in range(n_layers):
        T_i = numpyro.sample(f"T_node_{i}", dist.Uniform(Tlow, Thigh))
        T_nodes.append(T_i)
    T_nodes = jnp.array(T_nodes)

    Tarr = jnp.interp(log_p, log_p_nodes, T_nodes)
    return Tarr, T_nodes


def numpyro_isothermal(art: object, Tlow: float, Thigh: float) -> jnp.ndarray:
    """Isothermal profile with NumPyro sampling."""
    T0 = numpyro.sample("T0", dist.Uniform(Tlow, Thigh))
    return isothermal_profile(art, T0)


def numpyro_gradient(art: object, Tlow: float, Thigh: float) -> jnp.ndarray:
    """Gradient profile with NumPyro sampling."""
    T_btm = numpyro.sample("T_btm", dist.Uniform(Tlow, Thigh))
    T_top = numpyro.sample("T_top", dist.Uniform(Tlow, Thigh))
    return gradient_profile(art, T_btm, T_top)


def numpyro_madhu_seager(art: object, Tlow: float, Thigh: float) -> jnp.ndarray:
    """Madhusudhan-Seager profile with NumPyro sampling."""
    T_deep = numpyro.sample("T_deep", dist.Uniform(Tlow, Thigh))
    T_high = numpyro.sample("T_high", dist.Uniform(Tlow, Thigh))
    log_P_trans = numpyro.sample("log_P_trans", dist.Uniform(-8, 2))
    delta_P = numpyro.sample("delta_P", dist.Uniform(0.1, 3.0))

    P_trans = 10**log_P_trans
    return madhu_seager_profile(art, T_deep, T_high, P_trans, delta_P)


def numpyro_free_temperature(
    art: object,
    n_layers: int = 5,
    Tlow: float = 1000,
    Thigh: float = 4000,
) -> jnp.ndarray:
    """Free temperature profile with NumPyro sampling."""
    return free_temperature_profile(art, n_layers, Tlow, Thigh)[0]
