"""Temperature-pressure profiles for ultra-hot Jupiters."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def isothermal_profile(art: object, T0: float) -> jnp.ndarray:
    """Isothermal temperature profile."""
    return T0 * jnp.ones_like(art.pressure)

def numpyro_isothermal(art: object, Tlow: float, Thigh: float) -> jnp.ndarray:
    """Isothermal profile with NumPyro sampling."""
    T0 = numpyro.sample("T0", dist.Uniform(Tlow, Thigh))
    return isothermal_profile(art, T0)


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

def numpyro_madhu_seager(art: object, Tlow: float, Thigh: float) -> jnp.ndarray:
    """Madhusudhan-Seager profile with NumPyro sampling."""
    T_deep = numpyro.sample("T_deep", dist.Uniform(Tlow, Thigh))
    T_high = numpyro.sample("T_high", dist.Uniform(Tlow, Thigh))
    log_P_trans = numpyro.sample("log_P_trans", dist.Uniform(-8, 2))
    delta_P = numpyro.sample("delta_P", dist.Uniform(0.1, 3.0))

    P_trans = 10**log_P_trans
    return madhu_seager_profile(art, T_deep, T_high, P_trans, delta_P)




# TODO: this requires strong priors and careful tuning to work

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


def numpyro_free_temperature(
    art: object,
    n_layers: int = 5,
    Tlow: float = 1000,
    Thigh: float = 4000,
) -> jnp.ndarray:
    """Free temperature profile with NumPyro sampling."""
    return free_temperature_profile(art, n_layers, Tlow, Thigh)[0]
