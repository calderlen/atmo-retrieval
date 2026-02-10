- TODO: implement sysrem



- TODO: need to subtract off the stellar spectrum and stuff, so look how atmo-analysis does it
- TODO: i guess molecfit was already run on the tellurics so you don't need to model them out? check the do_molecfit logic in old code
- TODO: determine what sort of atmospheric chemistry model you're going to use
    - free chemistry: sample log Xi directly as parameters and run NUTS/HCM/nested sampling
        - obtain species-specific constraints
        - don't assume equilibrium; cases against equilibrium include UHJ ionization, day/night chemistry, vertical mixing
        - minimal assumptions
    - equilibrium chemistry retrieval inside JAX
        - ExoGibbs to recover elemental ratios in a physically-coupled way
        - FastChem2/3 to be used as a comparator or to precompute a grid interpolated in JAX
- TODO: remaining retrieval parameters
    - log P_0 (reference pressure)
    - gray opacity/haze amplitude
- TODO: model secondary retrieval parameters
    - ionization partiiton, retrieving the ions and neutral species separately OR ionization fraciton OR retrieve them separately, moving this up to core retrieval parameters
    - MMW/metallicity/(C/O)?
- TODO: what not to try to retrieve, make sure to avoid this
     - free T-P profile
     - log g?

- TODO: add joint retreival?

- TODO: look up `tau` (ingress/egress duration) values for planet ephemerides
    - Currently only KELT-20b Duck24 has a real value (tau = 0.02007 days)
    - All others have tau = NaN and will error until filled in
    - Need values for: Lund17, Singh24, WASP-76b, KELT-9b, WASP-12b, WASP-33b, WASP-18b, WASP-189b, MASCARA-1b, TOI-1431b, TOI-1518b
    - Sources: ExoFOP, discovery papers, or fit from TESS lightcurves

