"""
KELT-20b Ultra-Hot Jupiter Retrieval Configuration
===================================================

System parameters, paths, and configuration for PEPSI observations.
"""

import os
from exojax.utils.constants import RJ, Rs, MJ
from uncertainties import ufloat

# TODO: probably using dicts, or whatever is faster, need to make which set of system parameters used something selectable via cli. need to make dicts for say the k20b system with each published set of parameters to compare thermal
# TODO: maybe make a model that does some sort of comparison of various study's params of the same system to continue to quantify how this stuff is dependent on input params.
# ==============================================================================
# TARGET SYSTEM PARAMETERS - KELT-20b (Ultra-Hot Jupiter)
# ==============================================================================




# Orbital parameters

# TODO: add all of the study's params along with their corresponding uncertainties, similar to how is done for PERIOD_DAY

PLANET = "KELT-20b"
EPHEMERIS = "Lund17"

PERIOD_DAY = {
    "KELT-20b" : {
        "Duck24": ufloat(3.47410151, 0.00000012),
        "Singh24": ufloat(3.4741039, 0.0000040),
        ""

        }
    }





PERIOD_DAY = 3.4741095  # Lund et al. 2017
SEMI_MAJOR_AXIS = 0.0542  # AU

# Planet parameters (Lund et al. 2017, updated)
MP_MEAN, MP_STD = 3.382, 0.13  # [M_J]
RP_MEAN, RP_STD = 1.735, 0.07  # [R_J]

# Stellar parameters
RSTAR_MEAN, RSTAR_STD = 1.60, 0.06  # [R_Sun]
MSTAR_MEAN, MSTAR_STD = 1.89, 0.06  # [M_Sun]
TSTAR = 8980  # K (A2V star)





# TODO: remove this code from different codebase after you're done filling in values

def get_planet_parameters(planet_name):

    MJoMS = 1./1047. #MJ in MSun

    if planet_name == 'KELT-20b':
        #for KELT-20 b:, from Alison Duck 2024
        Period = ufloat(3.47410151, 0.00000012) #days
        epoch = ufloat(2459757.811, 0.000019) #BJD_TDB
        dur = 0.14762
        i = ufloat(86.065, 0.073) #degrees

        # from Singh et al 2024
        #Period = ufloat(3.4741039, 0.0000040)
        #epoch = ufloat(2457000+2406.927174,0.000024)
        #dur = 0.1475
        #i=ufloat(86.03, 0.05)

        # from Lund et al 2018, "linear ephemeris from transits"
        #Period = ufloat(3.4741085, 0.0000019)
        #epoch = ufloat(2457503.120049, 0.000190)
        #dur = 0.14898
        #i = ufloat(86.12, 0.28) #degrees

        M_star = ufloat(1.76, 0.19) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        M_p = 3.382 #3-sigma limit
        R_p = 1.741
        RA = '19h38m38.74s'
        Dec = '+31d13m09.12s'

        Ks_expected = 0.0

        #add vsini, lambda, spin-orbit-misalignment,any other horus params


    if planet_name == 'WASP-76b':
        #For WASP-76 b:, from West et al. 2016
        Period = ufloat(1.809886, 0.000001) #days
        epoch = ufloat(2456107.85507, 0.00034) #BJD_TDB

        M_star = ufloat(1.46, 0.07) #MSun
        RV_abs = ufloat(-1.152, 0.0033) #km/s
        i = ufloat(88.0, 1.6) #degrees
        M_p = 0.92
        R_p = 1.83

        RA = '01h46m31.90s'
        Dec = '+02d42m01.40s'

        dur = 3.694/24.

        Ks_expected = 0.0

    if planet_name == 'KELT-9b':
        #For KELT-9 b:, from Gaudi et al. 2017 and Pai Asnodkar et al. 2021
        Period = ufloat(1.4811235, 0.0000011) #days
        epoch = ufloat(2457095.68572, 0.00014) #BJD_TDB

        M_star = ufloat(2.11, 0.78) #MSun
        RV_abs = ufloat(-37.11, 1.0) #km/s
        i = ufloat(86.79, 0.25) #degrees
        M_p = ufloat(2.17, 0.56)
        R_p = 1.891

        RA = '20h31m26.38s'
        Dec = '+39d56m20.10s'

        dur = 3.9158/24.

        Ks_expected = 0.0

    if planet_name == 'WASP-12b':
        #For WASP-12 b:, from Ivishina & Winn 2022, Bonomo+17, Charkabarty & Sengupta 2019
        Period = ufloat(1.091419108, 5.5e-08) #days
        epoch = ufloat(2457010.512173, 7e-05) #BJD_TDB

        M_star = ufloat(1.38, 0.18) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(83.3, 1.1) #degrees
        M_p = ufloat(1.39, 0.12)
        R_p = 1.937

        RA = '06h30m32.79s'
        Dec = '+29d40m20.16s'

        dur = 3.0408/24.

        Ks_expected = 0.0

    if planet_name == 'WASP-33b':
        #For WASP-33 b:, from Ivishina & Winn 2022, Bonomo+17, Charkabarty & Sengupta 2019
        Period = ufloat(1.219870, 0.000001) #days
        epoch = ufloat(2454163.22367, 0.00022) #BJD_TDB

        M_star = ufloat(1.495, 0.031) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(86.63, 0.03) #degrees
        M_p = ufloat(2.093, 0.139)
        R_p = 1.593

        RA = '02h26m51.06s'
        Dec = '+37d33m01.60s '

        dur = 2.854/24.

        Ks_expected = 0.0

    if planet_name == 'WASP-18b':
        #For WASP-18b: from Cortes-Zuleta+20
        Period = ufloat(0.94145223, 0.00000024) #days
        epoch = ufloat(2456740.80560, 0.00019) #BJD_TDB

        M_star = ufloat(1.294, 0.063) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(83.5, 2.0) #degrees
        M_p = ufloat(10.20, 0.35)
        R_p = 1.240

        RA = '01h37m25.07s'
        Dec = '-45d40m40.06s'

        dur = 2.21/24.

        Ks_expected = 0.0

    if planet_name == 'WASP-189b':

        Period = ufloat(2.7240308, 0.0000028) #days
        epoch = ufloat(2458926.5416960, 0.0000650) #BJD_TDB

        M_star = ufloat(2.030, 0.066) #MSun
        RV_abs = ufloat(-22.4, 0.0) #km/s
        i = ufloat(84.03, 0.14) #degrees
        M_p = ufloat(1.99, 0.16)
        R_p = 1.619

        RA = '15h02m44.82s'
        Dec = '-03d01m53.35s'

        dur = 4.3336/24.

        Ks_expected = 0.0

    if planet_name == 'MASCARA-1b':

        Period = ufloat(2.14877381, 0.00000088) #days
        epoch = ufloat(2458833.488151, 0.000092) #BJD_TDB

        M_star = ufloat(1.900, 0.068) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(88.45, 0.17) #degrees
        M_p = ufloat(3.7, 0.9)
        R_p = 1.597

        RA = '21h10m12.37s'
        Dec = '+10d44m20.03s'

        dur = 4.226/24.

        Ks_expected = 0.0

    if planet_name == 'TOI-1431b':

        Period = ufloat(2.650237, 0.000003) #days
        epoch = ufloat(2458739.17737, 0.00007) #BJD_TDB

        M_star = ufloat(1.90, 0.10) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(80.13, 0.13) #degrees
        M_p = ufloat(3.12, 0.18)
        R_p = 1.49

        RA = '21h04m48.89s'
        Dec = '+55d35m16.88s'

        dur = 2.489/24.
        Ks_expected = 294.1 #m/s


    if planet_name == 'TOI-1518b':

        Period = ufloat(1.902603, 0.000011) #days
        epoch = ufloat(2458787.049255, 0.000094) #BJD_TDB

        M_star = ufloat(1.79, 0.26) #MSun
        RV_abs = ufloat(0.0, 0.0) #km/s
        i = ufloat(77.84, 0.26) #degrees
        M_p = ufloat(2.3, 2.3)
        R_p = 1.875

        RA = '23h29m04.20s'
        Dec = '+67d02m05.30s'

        dur = 2.365/24.
        Ks_expected = 0.0




# ==============================================================================
# OBSERVATION PARAMETERS - PEPSI/LBT
# ==============================================================================

# Wavelength range depends on observing mode
# Options: "blue" (383-476 nm), "green" (476-657 nm), "red" (650-907 nm)
OBSERVING_MODE = "red"  # Change based on your data


# TODO: maybe convert to angstroms? although i'd rather not
# TODO: check the CD ranges in the PEPSI docs and the old code and edit these.
# TODO: green?
# TODO: uncertainties? probably not
WAVELENGTH_RANGES = {
    "blue": (383, 476),    # nm
    "green": (476, 657),   # nm
    "red": (650, 907),     # nm
    "full": (383, 907),    # If combining multiple modes
}

WAV_MIN, WAV_MAX = WAVELENGTH_RANGES[OBSERVING_MODE]

RESOLUTION = 120000  # Adjust based on your setup

# ==============================================================================
# ATMOSPHERIC RT PARAMETERS
# ==============================================================================

DIFFMODE = 0
NLAYER = 100  # Number of atmospheric layers

# Pressure range [bar]
PRESSURE_TOP = 1e-8   # Extended for ultra-hot Jupiter
PRESSURE_BTM = 1e2

# Temperature range [K] - Ultra-hot Jupiter
TLOW = 1500.0   # Cooler nightside
THIGH = 4500.0  # Very hot dayside with thermal inversion

# ==============================================================================
# SPECTRAL GRID PARAMETERS
# ==============================================================================

N_SPECTRAL_POINTS = 100000  # Higher resolution for PEPSI
WAV_MIN_OFFSET = 10  # nm
WAV_MAX_OFFSET = 10  # nm

# preMODIT parameters
NDIV = 8  # More divisions for higher resolution

# ==============================================================================
# CLOUD/HAZE PARAMETERS
# ==============================================================================

# Ultra-hot Jupiters may have minimal clouds on dayside
# but could have condensates on nightside
CLOUD_WIDTH = 1.0 / 20.0  # Cloud width in log10(P)
CLOUD_INTEGRATED_TAU = 30.0  # Lower than WASP-39b

# ==============================================================================
# DATABASE PATHS
# ==============================================================================


# TODO: make input the base directory for these, so input dir will be placed in the root level and also it will be in the gitignore
# Molecular databases
DB_HITEMP = os.path.expanduser("~/.db_HITEMP/")
DB_EXOMOL = os.path.expanduser("~/.db_ExoMol/")
DB_KURUCZ = os.path.expanduser("~/.db_kurucz/")  # For atomic lines


# TODO: make input the base directory for these, so input dir will be placed in the root level and also it will be in the gitignore
# CIA paths
CIA_PATHS = {
    "H2H2": os.path.join(DB_HITEMP, "../.db_CIA/H2-H2_2011.cia"),
    "H2He": os.path.join(DB_HITEMP, "../.db_CIA/H2-He_2011.cia"),
}


# TODO: make input the base directory for these, so input dir will be placed in the root level and also it will be in the gitignore
# Molecular line lists for ultra-hot Jupiter
MOLPATH_HITEMP = {
    "H2O": f"{DB_HITEMP}H2O/",
    "CO": f"{DB_HITEMP}CO/",
    "OH": f"{DB_HITEMP}OH/",
}


# TODO: make input the base directory for these, so input dir will be placed in the root level and also it will be in the gitignore
MOLPATH_EXOMOL = {
    "TiO": f"{DB_EXOMOL}TiO/48Ti-16O/Toto/",
    "VO": f"{DB_EXOMOL}VO/51V-16O/VOMYT/",
    "FeH": f"{DB_EXOMOL}FeH/56Fe-1H/MoLLIST/",
    "CaH": f"{DB_EXOMOL}CaH/40Ca-1H/MoLLIST/",
    "CrH": f"{DB_EXOMOL}CrH/52Cr-1H/MoLLIST/",
    "AlO": f"{DB_EXOMOL}AlO/27Al-16O/ATP/",
    # Add more as needed
}


# TODO: make input the base directory for these, so input dir will be placed in the root level and also it will be in the gitignore
# Atomic line lists (Kurucz/VALD)
ATOMIC_SPECIES = {
    "Na": {"element": "Na", "ionization": 0},  # Na I
    "K": {"element": "K", "ionization": 0},    # K I
    "Ca": {"element": "Ca", "ionization": 1},  # Ca II
    "Fe": {"element": "Fe", "ionization": 0},  # Fe I
    "Ti": {"element": "Ti", "ionization": 0},  # Ti I
    "V": {"element": "V", "ionization": 0},    # V I
}

# ==============================================================================
# TELLURIC LINE MODELING
# ==============================================================================

# Enable telluric correction for ground-based observations
ENABLE_TELLURICS = True

# Telluric species (Earth's atmosphere)
TELLURIC_SPECIES = {
    "H2O": f"{DB_HITEMP}H2O/",  # Water vapor (dominant in optical)
    "O2": f"{DB_HITEMP}O2/",    # Oxygen
    # Add more if needed
}

# Typical telluric parameters (can be free parameters in retrieval)
TELLURIC_PWV = 5.0  # Precipitable water vapor [mm]
TELLURIC_AIRMASS = 1.2  # Typical airmass

# ==============================================================================
# DATA PATHS
# ==============================================================================


# TODO: this input directory stucture should be strengthened
DATA_DIR = "input/spectra/kelt20b_pepsi"
os.makedirs(DATA_DIR, exist_ok=True)



# TODO: Adjust these based on your actual data files
TRANSMISSION_DATA = {
    "wavelength": f"{DATA_DIR}/wavelength_transmission.npy",
    "spectrum": f"{DATA_DIR}/spectrum_transmission.npy",
    "uncertainty": f"{DATA_DIR}/uncertainty_transmission.npy",
}

EMISSION_DATA = {
    "wavelength": f"{DATA_DIR}/wavelength_emission.npy",
    "spectrum": f"{DATA_DIR}/spectrum_emission.npy",
    "uncertainty": f"{DATA_DIR}/uncertainty_emission.npy",
}

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================


# TODO: dumb name fix
DIR_SAVE = "output_kelt20b"
os.makedirs(DIR_SAVE, exist_ok=True)

# Opacity loading/saving
OPA_LOAD = True
OPA_SAVE = False

# ==============================================================================
# INFERENCE PARAMETERS
# ==============================================================================

# SVI parameters
SVI_NUM_STEPS = 2000
SVI_LEARNING_RATE = 0.005

# MCMC parameters
MCMC_NUM_WARMUP = 2000
MCMC_NUM_SAMPLES = 2000
MCMC_MAX_TREE_DEPTH = 6

# Parallel chains for better convergence
MCMC_NUM_CHAINS = 4

# ==============================================================================
# RETRIEVAL MODE
# ==============================================================================

# Options: "transmission", "emission"
RETRIEVAL_MODE = "transmission"  # Change based on your analysis


# TODO: probably just completely unnecessary
# For combined retrieval
COMBINE_TRANSMISSION_EMISSION = False
