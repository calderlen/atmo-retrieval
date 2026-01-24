"""
Forward Model Module
====================

NumPyro probabilistic model for transmission spectrum retrieval.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from exojax.database import molinfo
from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.constants import RJ, Rs, MJ


def create_transmission_model(
    art,
    opa_mols,
    opa_cias,
    molmass_arr,
    nu_grid,
    sop_rot,
    sop_inst,
    beta_inst,
    inst_nus,
    period_day,
    Mp_mean,
    Mp_std,
    Rstar_mean,
    Rstar_std,
    Tlow,
    Thigh,
    pressure_top,
    pressure_btm,
    nlayer,
    cloud_width,
    cloud_integrated_tau,
):
    """
    Create NumPyro transmission spectrum model.

    This returns a function that can be used with NumPyro's inference algorithms.

    Parameters
    ----------
    art : ArtTransPure
        Atmospheric radiative transfer object
    opa_mols : dict
        Molecular opacities
    opa_cias : dict
        CIA opacities
    molmass_arr : jnp.ndarray
        Molecular masses
    nu_grid : jnp.ndarray
        Wavenumber grid
    sop_rot : SopRotation
        Rotation operator
    sop_inst : SopInstProfile
        Instrumental profile operator
    beta_inst : float
        Instrumental Gaussian width
    inst_nus : jnp.ndarray
        Instrument wavenumber grid
    period_day : float
        Orbital period [days]
    Mp_mean : float
        Planet mass prior mean [M_J]
    Mp_std : float
        Planet mass prior std [M_J]
    Rstar_mean : float
        Stellar radius prior mean [R_Sun]
    Rstar_std : float
        Stellar radius prior std [R_Sun]
    Tlow : float
        Low temperature [K]
    Thigh : float
        High temperature [K]
    pressure_top : float
        Top pressure [bar]
    pressure_btm : float
        Bottom pressure [bar]
    nlayer : int
        Number of atmospheric layers
    cloud_width : float
        Cloud width in log10(P)
    cloud_integrated_tau : float
        Integrated cloud optical depth

    Returns
    -------
    model : callable
        NumPyro model function
    """

    def model_c(rp_mean, rp_std):
        """NumPyro model: forward spectral model + priors."""

        # Planet/star parameters
        Mp = numpyro.sample("Mp", dist.TruncatedNormal(Mp_mean, Mp_std, low=0)) * MJ
        Rstar = (
            numpyro.sample("Rs", dist.TruncatedNormal(Rstar_mean, Rstar_std, low=0)) * Rs
        )
        radius_btm = numpyro.sample("Radius_btm", dist.Uniform(1.0, 1.5)) * RJ
        RV = numpyro.sample("RV", dist.Uniform(-200, 0))

        # Atmospheric composition
        vmr_arr = []
        for mol in opa_mols:
            logVMR = numpyro.sample(f"logVMR_{mol}", dist.Uniform(-15, 0))
            vmr_arr.append(art.constant_mmr_profile(jnp.power(10.0, logVMR)))
        vmr_arr = jnp.array(vmr_arr)

        vmr_tot = jnp.clip(jnp.sum(vmr_arr, axis=0), 0.0, 1.0)
        vmrH2 = (1.0 - vmr_tot) * 6.0 / 7.0
        vmrHe = (1.0 - vmr_tot) * 1.0 / 7.0

        mmw = (
            molinfo.molmass_isotope("H2") * vmrH2
            + molinfo.molmass_isotope("He", db_HIT=False) * vmrHe
            + jnp.dot(molmass_arr, vmr_arr)
        )

        # Temperature structure (isothermal)
        T0 = numpyro.sample("T0", dist.Uniform(Tlow, Thigh))
        Tarr = T0 * jnp.ones_like(art.pressure)

        # Grey cloud deck
        logP_cloud = numpyro.sample("logP_cloud", dist.Uniform(-11, 1))

        dtau_c = (
            cloud_integrated_tau
            * ((jnp.log10(pressure_btm) - jnp.log10(pressure_top)) / nlayer)
            / cloud_width
        )
        pressure_arr = jnp.log10(art.pressure)
        cloud_profile = (pressure_arr[:, None] - logP_cloud) / cloud_width
        dtau_cloud = (
            dtau_c / jnp.sqrt(jnp.pi) * jnp.exp(-jnp.clip(cloud_profile**2, -50, 50))
        )
        dtau_cloud = jnp.broadcast_to(dtau_cloud, (pressure_arr.size, nu_grid.size))

        # Gravity profile
        gravity_btm = gravity_jupiter(radius_btm / RJ, Mp / MJ)
        gravity = art.gravity_profile(Tarr, mmw, radius_btm, gravity_btm)

        # Opacity summation
        dtau = dtau_cloud

        # CIA
        for molA, molB in [("H2", "H2"), ("H2", "He")]:
            logacia_matrix = opa_cias[molA + molB].logacia_matrix(Tarr)
            vmrX, vmrY = (vmrH2, vmrH2) if molB == "H2" else (vmrH2, vmrHe)
            dtau += art.opacity_profile_cia(
                logacia_matrix, Tarr, vmrX, vmrY, mmw[:, None], gravity
            )

        # Line opacity
        for i, mol in enumerate(opa_mols):
            xsmatrix = opa_mols[mol].xsmatrix(Tarr, art.pressure)
            dtau += art.opacity_profile_xs(xsmatrix, vmr_arr[i], mmw[:, None], gravity)

        # Radiative transfer
        rp2 = art.run(dtau, Tarr, mmw, radius_btm, gravity_btm)

        # Broadening kernels
        vsini = 2 * jnp.pi * radius_btm / (period_day * 86400) / 1e5  # km/s
        u1 = u2 = 0.0  # limb darkening
        Frot = sop_rot.rigid_rotation(rp2, vsini, u1, u2)
        Frot_inst = sop_inst.ipgauss(Frot, beta_inst)
        Rp2_sample = sop_inst.sampling(Frot_inst, RV, inst_nus)

        mu = jnp.sqrt(Rp2_sample) * (radius_btm / Rstar)

        # Likelihood
        numpyro.deterministic("rp_mu", mu[::-1])
        numpyro.sample("rp", dist.Normal(mu[::-1], rp_std), obs=rp_mean)

    return model_c
