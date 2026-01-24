
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import numpyro
from jax import random



def fspec(T0, alpha, mmr, radius_btm, gravity_btm, RV):
    """ computes planet radius sqaure spectrum

    Args:
        T0 (float): temperature at 1 bar
        alpha (float): power law index of temperature
        mmr (float): Mass mixing ratio of CO
        radius_btm (float): radius at the bottom in cm
        gravity_btm (float): gravity at the bottom in cm/s2
        RV (float): radial velocity in km/s

    Returns:
        _type_: _description_
    """

    Tarr = art.powerlaw_temperature(T0, alpha)
    gravity = art.gravity_profile(Tarr, mmw, radius_btm, gravity_btm)

    #molecule
    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    mmr_arr = art.constant_mmr_profile(mmr)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, molmass, gravity)
    #continuum
    logacia_matrix = opacia.logacia_matrix(Tarr)
    dtaucH2H2 = art.opacity_profile_cia(logacia_matrix, Tarr, vmrH2, vmrH2,
                                        mmw[:, None], gravity)
    #total tau
    dtau = dtau + dtaucH2H2
    Rp2 = art.run(dtau, Tarr, mmw, radius_btm, gravity_btm)
    Rp2_inst = sop_inst.ipgauss(Rp2, beta_inst)

    mu = sop_inst.sampling(Rp2_inst, RV, nu_obs)
    return mu



def model_prob(spectrum):

    #atmospheric/spectral model parameters priors
    logg = numpyro.sample('logg', dist.Uniform(3.0, 4.0))
    RV = numpyro.sample('RV', dist.Uniform(35.0, 45.0))
    mmr = numpyro.sample('MMR', dist.Uniform(0.0, 0.015))
    T0 = numpyro.sample('T0', dist.Uniform(1000.0, 1500.0))
    alpha = numpyro.sample('alpha', dist.Uniform(0.05, 0.2))
    radius_btm = numpyro.sample('rb', dist.Normal(1.0,0.05))

    mu = fspec(T0, alpha, mmr, radius_btm*RJ, 10**logg, RV)

    #noise model parameters priors
    sigmain = numpyro.sample('sigmain', dist.Exponential(1000.0))

    numpyro.sample('spectrum', dist.Normal(mu, sigmain), obs=spectrum)





rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 500, 1000
#kernel = NUTS(model_prob, forward_mode_differentiation=True)
kernel = NUTS(model_prob, forward_mode_differentiation=False)



mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_, spectrum=Fobs)
mcmc.print_summary()
