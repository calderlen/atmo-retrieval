"""
Thermal Structure Module
=========================

Temperature-pressure profiles for ultra-hot Jupiters.
Includes isothermal, gradient, Guillot, and free temperature profiles.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


# TODO: probably all of these parameters are needed but may be useful

def isothermal_profile(art, T0):
    """
    Isothermal temperature profile.

    Parameters
    ----------
    art : ArtEmisPure or ArtTransPure
        Atmospheric RT object
    T0 : float
        Constant temperature [K]

    Returns
    -------
    Tarr : jnp.ndarray
        Temperature array [K]
    """
    return T0 * jnp.ones_like(art.pressure)


def gradient_profile(art, T_btm, T_top):
    """
    Linear temperature gradient in log(P) space.

    Parameters
    ----------
    art : ArtEmisPure or ArtTransPure
        Atmospheric RT object
    T_btm : float
        Temperature at bottom pressure [K]
    T_top : float
        Temperature at top pressure [K]

    Returns
    -------
    Tarr : jnp.ndarray
        Temperature array [K]
    """
    log_p = jnp.log10(art.pressure)
    log_p_btm = jnp.log10(art.pressure[-1])
    log_p_top = jnp.log10(art.pressure[0])

    # Linear interpolation in log(P) space
    Tarr = T_top + (T_btm - T_top) * (log_p - log_p_top) / (log_p_btm - log_p_top)
    return Tarr


def guillot_profile(art, T_irr, T_int, kappa_ir, gamma):
    """
    Guillot (2010) temperature profile with thermal inversion.

    Parameters
    ----------
    art : ArtEmisPure or ArtTransPure
        Atmospheric RT object
    T_irr : float
        Irradiation temperature [K]
    T_int : float
        Internal temperature [K]
    kappa_ir : float
        IR opacity [cm^2/g]
    gamma : float
        Ratio of visible to IR opacity

    Returns
    -------
    Tarr : jnp.ndarray
        Temperature array [K]
    """
    # Simplified Guillot profile
    # tau is optical depth (needs to be computed from pressure)
    # This is a placeholder - full implementation requires integration

    # Approximation using pressure as proxy for optical depth
    tau = art.pressure / 1e-3  # Rough approximation

    # Guillot formula (simplified)
    T4_eff = T_int**4 + T_irr**4 * (
        2.0 / 3.0 + 2.0 / 3.0 / gamma * (1.0 + gamma * tau / 2.0 - gamma * tau)
    )

    Tarr = jnp.power(jnp.clip(T4_eff, 0, None), 0.25)
    return Tarr


def madhu_seager_profile(art, T_deep, T_high, P_trans, delta_P):
    """
    Madhusudhan & Seager (2009) smoothly-varying profile.

    Allows for thermal inversions.

    Parameters
    ----------
    art : ArtEmisPure or ArtTransPure
        Atmospheric RT object
    T_deep : float
        Temperature at deep atmosphere [K]
    T_high : float
        Temperature at upper atmosphere [K]
    P_trans : float
        Transition pressure [bar]
    delta_P : float
        Width of transition in log(P)

    Returns
    -------
    Tarr : jnp.ndarray
        Temperature array [K]
    """
    log_p = jnp.log10(art.pressure)
    log_p_trans = jnp.log10(P_trans)

    # Sigmoid transition
    alpha = (log_p - log_p_trans) / delta_P
    f_transition = 0.5 * (1.0 + jnp.tanh(alpha))

    Tarr = T_high + (T_deep - T_high) * f_transition
    return Tarr


def free_temperature_profile(art, n_layers=5, Tlow=1000, Thigh=4000):
    """
    Free temperature profile with piecewise linear interpolation.

    Parameters
    ----------
    art : ArtEmisPure or ArtTransPure
        Atmospheric RT object
    n_layers : int
        Number of free temperature nodes
    Tlow : float
        Minimum allowed temperature [K]
    Thigh : float
        Maximum allowed temperature [K]

    Returns
    -------
    Tarr : jnp.ndarray
        Temperature array [K]
    T_nodes : jnp.ndarray
        Temperature node values
    """
    # Sample temperatures at fixed pressure levels
    log_p = jnp.log10(art.pressure)
    log_p_nodes = jnp.linspace(log_p.min(), log_p.max(), n_layers)

    T_nodes = []
    for i in range(n_layers):
        T_i = numpyro.sample(f"T_node_{i}", dist.Uniform(Tlow, Thigh))
        T_nodes.append(T_i)
    T_nodes = jnp.array(T_nodes)

    # Interpolate between nodes
    Tarr = jnp.interp(log_p, log_p_nodes, T_nodes)

    return Tarr, T_nodes


def numpyro_isothermal(art, Tlow, Thigh):
    """
    Isothermal profile with NumPyro sampling.

    Parameters
    ----------
    art : ArtEmisPure or ArtTransPure
        Atmospheric RT object
    Tlow : float
        Lower temperature bound [K]
    Thigh : float
        Upper temperature bound [K]

    Returns
    -------
    Tarr : jnp.ndarray
        Temperature array [K]
    """
    T0 = numpyro.sample("T0", dist.Uniform(Tlow, Thigh))
    return isothermal_profile(art, T0)


def numpyro_gradient(art, Tlow, Thigh):
    """
    Gradient profile with NumPyro sampling.

    Parameters
    ----------
    art : ArtEmisPure or ArtTransPure
        Atmospheric RT object
    Tlow : float
        Lower temperature bound [K]
    Thigh : float
        Upper temperature bound [K]

    Returns
    -------
    Tarr : jnp.ndarray
        Temperature array [K]
    """
    T_btm = numpyro.sample("T_btm", dist.Uniform(Tlow, Thigh))
    T_top = numpyro.sample("T_top", dist.Uniform(Tlow, Thigh))
    return gradient_profile(art, T_btm, T_top)


def numpyro_madhu_seager(art, Tlow, Thigh):
    """
    Madhusudhan-Seager profile with NumPyro sampling.

    Parameters
    ----------
    art : ArtEmisPure or ArtTransPure
        Atmospheric RT object
    Tlow : float
        Lower temperature bound [K]
    Thigh : float
        Upper temperature bound [K]

    Returns
    -------
    Tarr : jnp.ndarray
        Temperature array [K]
    """
    T_deep = numpyro.sample("T_deep", dist.Uniform(Tlow, Thigh))
    T_high = numpyro.sample("T_high", dist.Uniform(Tlow, Thigh))
    log_P_trans = numpyro.sample("log_P_trans", dist.Uniform(-8, 2))
    delta_P = numpyro.sample("delta_P", dist.Uniform(0.1, 3.0))

    P_trans = 10**log_P_trans
    return madhu_seager_profile(art, T_deep, T_high, P_trans, delta_P)


def numpyro_free_temperature(art, n_layers=5, Tlow=1000, Thigh=4000):
    """
    Free temperature profile with NumPyro sampling.

    Parameters
    ----------
    art : ArtEmisPure or ArtTransPure
        Atmospheric RT object
    n_layers : int
        Number of temperature nodes
    Tlow : float
        Lower temperature bound [K]
    Thigh : float
        Upper temperature bound [K]

    Returns
    -------
    Tarr : jnp.ndarray
        Temperature array [K]
    """
    return free_temperature_profile(art, n_layers, Tlow, Thigh)[0]
