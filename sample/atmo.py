from jax import config
import jax.numpy as jnp

from exojax.utils.grids import wavenumber_grid
from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.constants import RJ

from exojax.database.contdb  import CdbCIA
from exojax.database.exomol.api import MdbExomol

from exojax.opacity import OpaPremodit, OpaCIA

from exojax.rt import ArtTransPure

from exojax.rt.chord import chord_geometric_matrix
from exojax.rt.chord import chord_optical_depth


config.update("jax_enable_x64", True)

"""
how ExoJAX works; 
(1) loading databases (*db):
    (1.1) mdb - CO
    (1.2) cdb - CIA
(2) calculating opacity (opa)
(3) running atmospheric radiative transfer (art)
(4) applying operations on the spectrum (sop)

"""

nu_grid, wav, resolution = wavenumber_grid(lambda_min,
                                            lambda_max,
                                            3500, 
                                            unit="AA",
                                            xsmode="premodit") #what's 3500, AA, premodit?


# (1.1) loading ExoMol databases
mdb = MdbExomol(".database/CO/12C-16O/Li2015", nurange=nu_grid)

# (1.2)

# (2) calculating opacity (opa)

molmass = mdb.molmass # we use molmass later
snap = mdb.to_snapshot() # extract snapshot from mdb
del mdb # save the memory
opa = OpaPremodit.from_snapshot(snap, nu_grid, auto_trange=[500.0, 1500.0], dit_grid_resolution=1.0)

# calculating cross-section for two different temperatures, 500 and 1500 K for P=1.0 bar
P = 1.0  # bar
T_1 = 500.0  # K
xsv_1 = opa.xsvector(T_1, P)  # cm2

T_2 = 1500.0  # K
xsv_2 = opa.xsvector(T_2, P)  # cm2


# (3) running atmospheric radiative transfer (art)

art = ArtTransPure(
    pressure_btm=1.0e1,
    pressure_top=1.0e-11,
    nlayer=200,
)

art.change_temperature_range(500.0, 1500.0)
Tarr = art.powerlaw_temperature(1200.0, 0.1) # power law model

mmr_profile = art.constant_mmr_profile(0.01)  # constant mmr profile

gravity_btm = gravity_jupiter(1.0, 1.0)
radius_btm = RJ

mmw = 2.33*jnp.ones_like(art.pressure)  # mean molecular weight of the atmosphere
gravity = art.gravity_profile(Tarr, mmw, radius_btm, gravity_btm)


# cia database
cdb = CdbCIA(".database/H2-H2_2011.cia", nurange=nu_grid)
opacia = OpaCIA(cdb, nu_grid=nu_grid)


#Before running the radiative transfer, we need cross sections for layers, called xsmatrix for CO and logacia_matrix for CIA (strictly speaking, the latter is not cross section but coefficient because CIA intensity is proportional density square). See here for the details.

xsmatrix = opa.xsmatrix(Tarr, art.pressure)
logacia_matrix = opacia.logacia_matrix(Tarr)


# convert the cross-section matrix to the optical depth profile
dtau_CO = art.opacity_profile_xs(xsmatrix, mmr_profile, molmass, gravity)
vmrH2 = 0.855  # VMR of H2
dtaucia = art.opacity_profile_cia(logacia_matrix, Tarr, vmrH2, vmrH2, mmw[:, None], gravity)


# example of addition of two opacity sources
dtau_total = dtau_CO + dtaucia

# To examine the contribution of each atmospheric layer to the transmission spectrum, one can, for example, look at the optical depth along the chord direction. This can be done as follows:

normalized_height, normalized_radius_lower = art.atmosphere_height(Tarr, mmw, radius_btm, gravity_btm)
cgm = chord_geometric_matrix(normalized_height, normalized_radius_lower)
dtau_chord = chord_optical_depth(cgm, dtau)



# PLOTTING, MOVE TO PLOT.PY LATER
from exojax.plot.atmplot import plottau
plottau(nu_grid, dtau_chord, Tarr, art.pressure)

# The above spectrum is called “raw spectrum” in ExoJAX. The effects applied to the raw spectrum is handled in ExoJAX by the spectral operator (sop).


# (4) applying operations on the spectrum (sop)
# instrumental profile, Doppler velocity shift and so on, any operation on spectra.
from exojax.postproc.specop import SopInstProfile
from exojax.utils.instfunc import resolution_to_gaussian_std



# so need to convolve the raw spectrum with the instrumental profile of PEPSI, probably observing mode dependent, 


sop_inst = SopInstProfile(nu_grid, vrmax=1000.0) # i dont know what SopInstProfile is but need the PEPSI one

RV = 40.0  # km/s, sample radial velocity shift?
resolution_inst = 30000.0 # sample resolution to be changed later
beta_inst = resolution_to_gaussian_std(resolution_inst) # idk
Rp2_inst = sop_inst.ipgauss(Rp2, beta_inst) # idk
nu_obs = nu_grid[::5][:-50] # simulate observed wavenumber grid


from numpy.random import normal
noise = 0.001
Fobs = sop_inst.sampling(Rp2_inst, RV, nu_obs) + normal(0.0, noise, len(nu_obs))


