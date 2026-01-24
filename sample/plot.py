


from exojax.plot.atmplot import plottau
plottau(nu_grid, dtau_chord, Tarr, art.pressure)

# The above spectrum is called “raw spectrum” in ExoJAX. The effects applied to the raw spectrum is handled in ExoJAX by the spectral operator (sop).

from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive
import jax.numpy as jnp

# SAMPLING
posterior_sample = mcmc.get_samples()
pred = Predictive(model_prob, posterior_sample, return_sites=['spectrum'])
predictions = pred(rng_key_, spectrum=None)
median_mu1 = jnp.median(predictions['spectrum'], axis=0)
hpdi_mu1 = hpdi(predictions['spectrum'], 0.9)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 4.5))
ax.plot(nu_obs, median_mu1, color='C1')
ax.fill_between(nu_obs,
                hpdi_mu1[0],
                hpdi_mu1[1],
                alpha=0.3,
                interpolate=True,
                color='C1',
                label='90% area')
ax.errorbar(nu_obs, Fobs, noise, fmt=".", label="mock spectrum", color="black",alpha=0.5)
plt.xlabel('wavenumber (cm-1)', fontsize=16)
plt.legend(fontsize=14)
plt.tick_params(labelsize=14)
plt.show()



import arviz
pararr = ['T0', 'alpha', 'logg', 'MMR', 'radius_btm', 'RV']
arviz.plot_pair(arviz.from_numpyro(mcmc),
                kind='kde',
                divergences=False,
                marginals=True)
plt.show()

#Correlated noise can be introduced using a Gaussian process, and parameter estimation can be performed using SVI or Nested Sampling, just as in the case of emission spectra. See below for details.