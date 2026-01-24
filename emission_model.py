"""NumPyro model for emission spectrum retrieval (dayside/nightside)."""

from typing import Callable
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from exojax.database import molinfo
from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.constants import RJ, Rs, MJ
from exojax.postproc.specop import SopRotation, SopInstProfile
from exojax.opacity.premodit.api import OpaPremodit
from exojax.opacity.opacont import OpaCIA


def create_emission_model(
    art: object,
    opa_mols: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    molmass_arr: jnp.ndarray,
    nu_grid: jnp.ndarray,
    sop_rot: SopRotation,
    sop_inst: SopInstProfile,
    beta_inst: float,
    inst_nus: jnp.ndarray,
    period_day: float,
    Mp_mean: float,
    Mp_std: float,
    Rstar_mean: float,
    Rstar_std: float,
    Tstar: float,
    Tlow: float,
    Thigh: float,
    pressure_top: float,
    pressure_btm: float,
    nlayer: int,
    cloud_width: float,
    cloud_integrated_tau: float,
    temperature_profile: str = "madhu_seager",
    opa_atoms: dict | None = None,
) -> Callable:
    """Create NumPyro emission spectrum model."""

    def model_c(fp_mean: jnp.ndarray, fp_std: jnp.ndarray) -> None:
        """NumPyro emission model."""

        # Planet/star parameters
        Mp = numpyro.sample("Mp", dist.TruncatedNormal(Mp_mean, Mp_std, low=0)) * MJ
        Rstar = (
            numpyro.sample("Rs", dist.TruncatedNormal(Rstar_mean, Rstar_std, low=0)) * Rs
        )
        radius_btm = numpyro.sample("Radius_btm", dist.Uniform(1.0, 2.5)) * RJ
        RV = numpyro.sample("RV", dist.Uniform(-200, 200))

        # Atmospheric composition
        vmr_arr = []
        for mol in opa_mols:
            logVMR = numpyro.sample(f"logVMR_{mol}", dist.Uniform(-15, 0))
            vmr_arr.append(art.constant_mmr_profile(jnp.power(10.0, logVMR)))
        vmr_arr = jnp.array(vmr_arr)

        # Atomic abundances
        if opa_atoms is not None:
            atom_vmr_arr = []
            for atom in opa_atoms:
                logVMR_atom = numpyro.sample(f"logVMR_{atom}", dist.Uniform(-15, 0))
                atom_vmr_arr.append(art.constant_mmr_profile(jnp.power(10.0, logVMR_atom)))
            atom_vmr_arr = jnp.array(atom_vmr_arr)
            vmr_tot = jnp.clip(jnp.sum(vmr_arr, axis=0) + jnp.sum(atom_vmr_arr, axis=0), 0.0, 1.0)
        else:
            vmr_tot = jnp.clip(jnp.sum(vmr_arr, axis=0), 0.0, 1.0)

        vmrH2 = (1.0 - vmr_tot) * 6.0 / 7.0
        vmrHe = (1.0 - vmr_tot) * 1.0 / 7.0

        mmw = (
            molinfo.molmass_isotope("H2") * vmrH2
            + molinfo.molmass_isotope("He", db_HIT=False) * vmrHe
            + jnp.dot(molmass_arr, vmr_arr)
        )

        # Temperature structure
        if temperature_profile == "isothermal":
            T0 = numpyro.sample("T0", dist.Uniform(Tlow, Thigh))
            Tarr = T0 * jnp.ones_like(art.pressure)
        elif temperature_profile == "gradient":
            from thermal_structure import numpyro_gradient
            Tarr = numpyro_gradient(art, Tlow, Thigh)
        elif temperature_profile == "madhu_seager":
            from thermal_structure import numpyro_madhu_seager
            Tarr = numpyro_madhu_seager(art, Tlow, Thigh)
        elif temperature_profile == "free":
            from thermal_structure import numpyro_free_temperature
            Tarr = numpyro_free_temperature(art, n_layers=5, Tlow=Tlow, Thigh=Thigh)
        else:
            raise ValueError(f"Unknown temperature profile: {temperature_profile}")

        # Grey cloud deck
        logP_cloud = numpyro.sample("logP_cloud", dist.Uniform(-11, 2))

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

        # Molecular line opacity
        for i, mol in enumerate(opa_mols):
            xsmatrix = opa_mols[mol].xsmatrix(Tarr, art.pressure)
            dtau += art.opacity_profile_xs(xsmatrix, vmr_arr[i], mmw[:, None], gravity)

        # Atomic line opacity
        if opa_atoms is not None:
            for i, atom in enumerate(opa_atoms):
                xsmatrix = opa_atoms[atom].xsmatrix(Tarr, art.pressure)
                dtau += art.opacity_profile_xs(xsmatrix, atom_vmr_arr[i], mmw[:, None], gravity)

        # Radiative transfer (emission)
        F0 = art.run(dtau, Tarr, mmw, radius_btm, gravity_btm)

        # Broadening kernels
        vsini = 2 * jnp.pi * radius_btm / (period_day * 86400) / 1e5  # km/s
        u1 = u2 = 0.0  # limb darkening (planet)
        Frot = sop_rot.rigid_rotation(F0, vsini, u1, u2)
        Frot_inst = sop_inst.ipgauss(Frot, beta_inst)
        Fp_sample = sop_inst.sampling(Frot_inst, RV, inst_nus)

        # Planet-to-star flux ratio (Fp/Fs)
        from exojax.spec.planck import piBarr
        Fs_star = piBarr(nu_grid, Tstar)  # Stellar flux
        Fs_sample = sop_inst.sampling(Fs_star, RV, inst_nus)

        # Flux ratio accounting for radius ratio
        mu = Fp_sample / Fs_sample * (radius_btm / Rstar) ** 2

        # Likelihood
        numpyro.deterministic("fp_mu", mu[::-1])
        numpyro.sample("fp", dist.Normal(mu[::-1], fp_std), obs=fp_mean)

    return model_c
