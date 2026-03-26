class: model
def __init__():
**OBJECTS**
    - high resolution transmission spectroscopy -- hrts
            - PEPSI/LBT
    - high resolution emission spectroscopy -- hres
            - PEPSI/LBT
    - low resolution transmission spectroscopy -- lrts
            - HST
            - JWST
            - ?
    - low resolution emission spectroscopy -- lres
            - HST
            - JWST
            - ?
    - photometry -- phot
        - TESS; call tess_proc, also probably rename tess_proc 


**CLASS ATTIRBUTES**
- Kp
- Vsys
- dRV
- Mp
- Rstar
- Rp
- g_ref
- Tarr
- atmospheric composition


**INSTANCE ATTRIBUTES**
- hrts
    - model:
    - 
- hres
    - 
- lrts
    - 
- lres
    - 
- phot
    -

**METHODS**

- dta=compute_opacity
    - using exojax atmospheric radiative transfer object, opacity of molecules, opacity of atoms, CIA opacity, wavelength grid, temperature array, mmr_mols, mmr_atoms, vmrHR_profile, vmrHe_profile, mmw_profile, g

- create_model
    - depending on what instances are initialized, we sum their log likelihoods together. so if hrts+lrts are enabled we sum those loglik terms together. this keeps things modular
    - think all this function should really do is sum those terms together
    - now its not that simple because there are several modes: phase-resolved and full-transit, i think create_model can also handle this, splitting up the phase-resolved vs full-transit modes based on what is passed via cli/config






**current model.py setup**
- imports
- RetrievalModelConfig class
- helper functions
- planet_rv_kms: compute rv 
- sysrem_filter model: undo sysrem changes?
- check_grid_resolution, just 