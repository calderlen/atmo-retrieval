"""
HORUS: Doppler shadow model for Rossiter-McLaughlin effect correction.

Adapation of the HORUS code by Marshall C. Johnson (Johnson 2016) for computing
time-series line profile residuals during planetary transit. The Doppler
shadow is the spectrally-resolved manifestation of the Rossiter-McLaughlin
effect, appearing as a bump that tracks across the stellar line profile
as the planet transits.
"""

import numpy as np
from typing import Optional

# Speed of light in km/s
C_KMS = 2.9979e5


def compute_stellar_disk(
    rstar: int,
    vsini: float,
    gamma1: float,
    gamma2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute stellar disk properties: limb darkening and velocity field.

    Creates a 2D stellar disk grid with quadratic limb darkening and
    solid-body rotation velocity field.

    Args:
        rstar: Stellar disk resolution (number of pixels for radius)
        vsini: Projected stellar rotation velocity (km/s)
        gamma1: Linear limb darkening coefficient
        gamma2: Quadratic limb darkening coefficient

    Returns:
        limbarr: Limb darkening array, shape (2*rstar+1, 2*rstar+1)
        vx_arr: Radial velocity at each pixel (km/s), shape (2*rstar+1,)
        rarr: Distance from center array, shape (2*rstar+1, 2*rstar+1)
    """
    size = 2 * rstar + 1
    limbarr = np.zeros((size, size))
    rarr = np.zeros((size, size))
    vx_arr = np.zeros(size)

    for count, x in enumerate(range(-rstar, rstar + 1)):
        # Rotational velocity at this x-position (solid body)
        vx_arr[count] = vsini * x / rstar

        for ycount, y in enumerate(range(-rstar, rstar + 1)):
            r = np.sqrt(float(x)**2 + float(y)**2)
            rarr[count, ycount] = r

            if r <= rstar:
                # Angle from disk center
                theta = np.arcsin(r / rstar)
                mu = np.cos(theta)
                # Quadratic limb darkening: I(mu) = 1 - gamma1*(1-mu) - gamma2*(1-mu)^2
                limbarr[count, ycount] = 1.0 - gamma1 * (1.0 - mu) - gamma2 * (1.0 - mu)**2

    return limbarr, vx_arr, rarr


def compute_baseline_profile(
    limbarr: np.ndarray,
    vx_arr: np.ndarray,
    rarr: np.ndarray,
    v_grid: np.ndarray,
    width: float,
    rstar: int,
) -> np.ndarray:
    """Compute the unocculted stellar line profile.

    Integrates Gaussian line profiles across the stellar disk, weighted
    by limb darkening, to produce the baseline rotationally-broadened
    stellar line profile.

    Args:
        limbarr: Limb darkening array
        vx_arr: Velocity at each x-position
        rarr: Radial distance array
        v_grid: Velocity grid for output profile (km/s)
        width: Intrinsic line width (km/s)
        rstar: Stellar disk resolution

    Returns:
        baseline: Baseline line profile, shape (len(v_grid),)
    """
    size = 2 * rstar + 1
    baseline = np.zeros(len(v_grid))

    # Pre-compute line profiles at each x position
    linearr = np.zeros((size, len(v_grid)))
    for count, vx in enumerate(vx_arr):
        linearr[count, :] = np.exp(-((vx - v_grid) / width)**2)

    # Integrate across disk
    for count in range(size):
        for ycount in range(size):
            if rarr[count, ycount] <= rstar:
                baseline += linearr[count, :] * limbarr[count, ycount]

    return baseline


def compute_planet_positions(
    phase: np.ndarray,
    a_rs: float,
    b: float,
    lambda_angle: float,
    period: float,
    eccentricity: float = 0.0,
    periarg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute planet center positions on stellar disk for each exposure.

    Args:
        phase: Orbital phase array (0 = mid-transit)
        a_rs: Semi-major axis in stellar radii
        b: Impact parameter
        lambda_angle: Projected spin-orbit angle (radians)
        period: Orbital period (same units as phase denominator)
        eccentricity: Orbital eccentricity
        periarg: Argument of periastron (radians)

    Returns:
        x_planet: X positions on stellar disk (stellar radii), shape (n_exp,)
        y_planet: Y positions on stellar disk (stellar radii), shape (n_exp,)
    """
    # Convert phase to true anomaly position
    # For circular orbit: z = a * sin(2*pi*phase)
    if eccentricity == 0.0:
        # Simplified circular case
        z = a_rs * np.sin(2.0 * np.pi * phase)
    else:
        # Eccentric case
        omega = 2.0 * np.pi / period * (1.0 - eccentricity**2)**(-1.5) * (1.0 + eccentricity * np.sin(periarg))**2
        theta = omega * phase * period + np.pi / 2.0 - periarg
        z = a_rs * (1.0 - eccentricity**2) / (1.0 + eccentricity * np.cos(theta)) * (
            np.sin(periarg) * np.sin(theta) - np.cos(periarg) * np.cos(theta)
        )

    # Project onto stellar disk with spin-orbit angle
    x_planet = z * np.cos(lambda_angle) + b * np.sin(lambda_angle)
    y_planet = b * np.cos(lambda_angle) - z * np.sin(lambda_angle)

    return x_planet, y_planet


def compute_doppler_shadow(
    phase: np.ndarray,
    vsini: float,
    lambda_angle: float,
    b: float,
    rp_rs: float,
    gamma1: float,
    gamma2: float,
    v_grid: np.ndarray,
    a_rs: float,
    period: float = 1.0,
    resolution: int = 300,
    width: float = 5.0,
    eccentricity: float = 0.0,
    instrument_resolution: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Doppler shadow signal for each exposure.

    Models the distortion of the rotationally-broadened stellar line profile
    caused by a transiting planet obscuring part of the stellar disk.

    Args:
        phase: Orbital phase array (0 = mid-transit), shape (n_exposures,)
        vsini: Projected stellar rotation velocity (km/s)
        lambda_angle: Projected spin-orbit angle (degrees)
        b: Impact parameter (0 = center, 1 = limb)
        rp_rs: Planet-to-star radius ratio
        gamma1: Linear limb darkening coefficient
        gamma2: Quadratic limb darkening coefficient
        v_grid: Velocity grid for output profiles (km/s)
        a_rs: Semi-major axis in units of stellar radius
        period: Orbital period (for eccentric orbits)
        resolution: Stellar disk resolution (pixels per stellar radius)
        width: Intrinsic line width before rotation (km/s)
        eccentricity: Orbital eccentricity
        instrument_resolution: Spectral resolving power R for convolution

    Returns:
        shadow_profiles: Shadow signal for each exposure, shape (n_exposures, len(v_grid))
        baseline: Unocculted stellar line profile, shape (len(v_grid),)
    """
    n_exp = len(phase)
    npix = len(v_grid)
    rstar = int(resolution)

    # Convert angle to radians
    lambda_rad = np.radians(lambda_angle)

    # Planet radius in grid units
    rplanet = rp_rs * rstar

    # Compute stellar disk properties
    limbarr, vx_arr, rarr = compute_stellar_disk(rstar, vsini, gamma1, gamma2)

    # Compute baseline (unocculted) profile
    baseline = compute_baseline_profile(limbarr, vx_arr, rarr, v_grid, width, rstar)

    # Pre-compute line profiles at each x position
    size = 2 * rstar + 1
    linearr = np.zeros((size, npix))
    for count, vx in enumerate(vx_arr):
        linearr[count, :] = np.exp(-((vx - v_grid) / width)**2)

    # Get planet positions for each exposure
    x_planet, y_planet = compute_planet_positions(
        phase, a_rs, b, lambda_rad, period, eccentricity
    )

    # Scale to grid units
    x_planet *= rstar
    y_planet *= rstar

    # Compute shadow for each exposure
    shadow_profiles = np.zeros((n_exp, npix))

    for exp in range(n_exp):
        cx, cy = x_planet[exp], y_planet[exp]

        # Check if planet is on stellar disk
        if np.sqrt(cx**2 + cy**2) > rstar + rplanet:
            continue  # Planet not transiting

        # Compute occulted flux
        occulted = np.zeros(npix)

        # Loop over pixels in planet vicinity
        x_min = max(-rstar, int(cx - rplanet) - 1)
        x_max = min(rstar, int(cx + rplanet) + 1)
        y_min = max(-rstar, int(cy - rplanet) - 1)
        y_max = min(rstar, int(cy + rplanet) + 1)

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                idx_x = x + rstar
                idx_y = y + rstar

                # Check if pixel is on stellar disk
                if rarr[idx_x, idx_y] > rstar:
                    continue

                # Check if pixel is covered by planet
                dist_to_planet = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist_to_planet <= rplanet:
                    # Fully covered pixel
                    occulted += linearr[idx_x, :] * limbarr[idx_x, idx_y]
                elif dist_to_planet < rplanet + 1:
                    # Partial coverage (simple linear interpolation)
                    fraction = rplanet + 1 - dist_to_planet
                    occulted += fraction * linearr[idx_x, :] * limbarr[idx_x, idx_y]

        # Shadow is the difference: baseline - occulted profile
        # Normalized to baseline
        shadow_profiles[exp, :] = occulted

    # Normalize baseline
    baseline_max = np.max(baseline)
    if baseline_max > 0:
        baseline /= baseline_max
        shadow_profiles /= baseline_max

    # Apply instrumental convolution if requested
    if instrument_resolution is not None and instrument_resolution > 0:
        sigma = C_KMS / instrument_resolution
        # Create Gaussian kernel
        v_kernel = v_grid[np.abs(v_grid) < 5 * sigma] if len(v_grid) > 10 else v_grid
        if len(v_kernel) == 0:
            v_kernel = np.linspace(-3 * sigma, 3 * sigma, 21)
        kernel = np.exp(-0.5 * (v_kernel / sigma)**2)
        kernel /= np.sum(kernel)

        # Convolve baseline
        baseline = np.convolve(baseline, kernel, mode='same')

        # Convolve each shadow profile
        for exp in range(n_exp):
            shadow_profiles[exp, :] = np.convolve(shadow_profiles[exp, :], kernel, mode='same')

    return shadow_profiles, baseline


def remove_doppler_shadow(
    residual_flux: np.ndarray,
    wave: np.ndarray,
    phase: np.ndarray,
    planet_params: dict,
    stellar_params: dict,
    fit_scaling: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove Doppler shadow from residual spectra.

    Computes the Doppler shadow model and subtracts it from the data.
    Optionally fits a scaling factor to match the shadow amplitude.

    Args:
        residual_flux: Residual flux after median subtraction, shape (n_spectra, npix)
        wave: Wavelength grid (Angstroms)
        phase: Orbital phase for each spectrum
        planet_params: Dict with 'rp_rs', 'b', 'lambda_angle', 'a_rs', 'period'
        stellar_params: Dict with 'vsini', 'gamma1', 'gamma2'
        fit_scaling: If True, fit scaling factor to data

    Returns:
        corrected_flux: Shadow-corrected flux
        shadow_model: The subtracted shadow model
        fit_info: Dict with fitting information (scaling factor, etc.)
    """
    from config.instrument_config import get_resolution

    n_spectra, npix = residual_flux.shape

    # Convert wavelength to velocity (relative to line center)
    # Use middle of wavelength range as reference
    wave_ref = np.median(wave)
    v_grid = (wave - wave_ref) / wave_ref * C_KMS

    # Get instrument resolution
    R = get_resolution()

    # Compute shadow model
    shadow_profiles, baseline = compute_doppler_shadow(
        phase=phase,
        vsini=stellar_params['vsini'],
        lambda_angle=planet_params.get('lambda_angle', 0.0),
        b=planet_params['b'],
        rp_rs=planet_params['rp_rs'],
        gamma1=stellar_params['gamma1'],
        gamma2=stellar_params['gamma2'],
        v_grid=v_grid,
        a_rs=planet_params['a_rs'],
        period=planet_params.get('period', 1.0),
        instrument_resolution=R,
    )

    fit_info = {'baseline': baseline}

    if fit_scaling:
        # Fit scaling factor using least squares
        # Minimize ||residual - scale * shadow||^2
        shadow_flat = shadow_profiles.ravel()
        flux_flat = residual_flux.ravel()

        # Simple least squares: scale = (shadow . flux) / (shadow . shadow)
        shadow_dot_shadow = np.sum(shadow_flat**2)
        if shadow_dot_shadow > 0:
            scale = np.sum(shadow_flat * flux_flat) / shadow_dot_shadow
        else:
            scale = 1.0

        fit_info['scaling'] = scale
        shadow_model = scale * shadow_profiles
    else:
        fit_info['scaling'] = 1.0
        shadow_model = shadow_profiles

    # Subtract shadow
    corrected_flux = residual_flux - shadow_model

    return corrected_flux, shadow_model, fit_info


def fit_doppler_shadow(
    ccf_2d: np.ndarray,
    v_grid: np.ndarray,
    phase: np.ndarray,
    planet_params: dict,
    stellar_params: dict,
    fit_params: list[str] = ['lambda_angle', 'b', 'scaling'],
) -> dict:
    """Fit Doppler shadow model to 2D CCF data.

    Uses nonlinear least squares to fit the shadow model parameters
    to observed CCF residuals.

    Args:
        ccf_2d: 2D CCF array, shape (n_exposures, n_velocities)
        v_grid: Velocity grid (km/s)
        phase: Orbital phase array
        planet_params: Initial planet parameters
        stellar_params: Stellar parameters (fixed)
        fit_params: List of parameters to fit

    Returns:
        Dict with best-fit parameters and uncertainties
    """
    from scipy.optimize import least_squares

    def residual_func(params, ccf_data, v_grid, phase, planet_params, stellar_params, fit_params):
        # Unpack parameters
        pp = planet_params.copy()
        sp = stellar_params.copy()
        scaling = 1.0

        for i, pname in enumerate(fit_params):
            if pname == 'scaling':
                scaling = params[i]
            elif pname in ['lambda_angle', 'b', 'rp_rs', 'a_rs']:
                pp[pname] = params[i]
            elif pname in ['vsini', 'gamma1', 'gamma2']:
                sp[pname] = params[i]

        shadow, _ = compute_doppler_shadow(
            phase=phase,
            vsini=sp['vsini'],
            lambda_angle=pp.get('lambda_angle', 0.0),
            b=pp['b'],
            rp_rs=pp['rp_rs'],
            gamma1=sp['gamma1'],
            gamma2=sp['gamma2'],
            v_grid=v_grid,
            a_rs=pp['a_rs'],
        )

        return (ccf_data - scaling * shadow).ravel()

    # Initial parameter values
    x0 = []
    for pname in fit_params:
        if pname == 'scaling':
            x0.append(1.0)
        elif pname == 'lambda_angle':
            x0.append(planet_params.get('lambda_angle', 0.0))
        elif pname == 'b':
            x0.append(planet_params['b'])
        elif pname == 'vsini':
            x0.append(stellar_params['vsini'])
        else:
            x0.append(planet_params.get(pname, stellar_params.get(pname, 1.0)))

    # Run optimization
    result = least_squares(
        residual_func,
        x0,
        args=(ccf_2d, v_grid, phase, planet_params, stellar_params, fit_params),
        method='lm',
    )

    # Package results
    fit_result = {
        'success': result.success,
        'parameters': {},
    }
    for i, pname in enumerate(fit_params):
        fit_result['parameters'][pname] = result.x[i]

    return fit_result
