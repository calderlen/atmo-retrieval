"""
Species aliasing diagnostics for atmospheric retrieval.

This module computes cross-correlations between species templates to identify
potential aliasing/cross-talk. High correlation at v~0 between templates
indicates that species may be confused in the retrieval.

Key concept: If template A and template B have high cross-correlation at zero
velocity offset, then a detection of species A could be partially caused by
species B (or vice versa). This is particularly problematic when a strong
species (like Fe I with many lines) aliases with a weaker species.

References:
    - Borsato et al. (2023) for aliasing methodology
    - Petz et al. (2023) for PEPSI KELT-20b analysis
"""

from __future__ import annotations

import numpy as np
from typing import Literal


def compute_ccf(
    template_a: np.ndarray,
    template_b: np.ndarray,
    velocities: np.ndarray,
    wave: np.ndarray,
) -> np.ndarray:
    """Compute cross-correlation function between two templates.
    
    Args:
        template_a: First template spectrum (flux or absorption depth)
        template_b: Second template spectrum
        velocities: Array of velocity shifts to compute (km/s)
        wave: Wavelength array (Angstroms)
        
    Returns:
        CCF array, same length as velocities
    """
    c_kms = 299792.458  # Speed of light in km/s
    
    ccf = np.zeros(len(velocities))
    
    # Normalize templates
    a_norm = template_a - np.mean(template_a)
    b_norm = template_b - np.mean(template_b)
    
    # Precompute normalization
    a_std = np.std(template_a)
    b_std = np.std(template_b)
    
    if a_std == 0 or b_std == 0:
        return ccf  # One template is flat
    
    for i, v in enumerate(velocities):
        # Doppler shift template_b
        doppler_factor = np.sqrt((1 + v / c_kms) / (1 - v / c_kms))
        wave_shifted = wave * doppler_factor
        
        # Interpolate shifted template onto original wavelength grid
        b_shifted = np.interp(wave, wave_shifted, template_b, left=0, right=0)
        b_shifted_norm = b_shifted - np.mean(b_shifted)
        
        # Normalized cross-correlation
        ccf[i] = np.sum(a_norm * b_shifted_norm) / (len(wave) * a_std * np.std(b_shifted))
    
    return ccf


def compute_template_cross_correlation(
    templates: dict[str, tuple[np.ndarray, np.ndarray]],
    velocity_range: tuple[float, float] = (-50, 50),
    velocity_step: float = 1.0,
) -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray]:
    """Compute CCF between all pairs of species templates.
    
    For N species, computes N*(N-1)/2 cross-correlations. High correlation
    at v=0 indicates potential aliasing.
    
    Args:
        templates: Dict mapping species name to (wavelength, flux) tuple
        velocity_range: (min, max) velocity range to compute (km/s)
        velocity_step: Velocity grid spacing (km/s)
        
    Returns:
        Tuple of:
            - Dict mapping (species_A, species_B) -> ccf_array
            - Velocity array used for CCFs
    """
    velocities = np.arange(velocity_range[0], velocity_range[1] + velocity_step, velocity_step)
    ccf_dict = {}
    
    species_names = list(templates.keys())
    n_species = len(species_names)
    
    for i in range(n_species):
        for j in range(i + 1, n_species):
            species_a = species_names[i]
            species_b = species_names[j]
            
            wave_a, flux_a = templates[species_a]
            wave_b, flux_b = templates[species_b]
            
            # Interpolate both templates onto a common wavelength grid
            wave_min = max(wave_a.min(), wave_b.min())
            wave_max = min(wave_a.max(), wave_b.max())
            
            # Use finer grid
            n_points = max(len(wave_a), len(wave_b))
            wave_common = np.linspace(wave_min, wave_max, n_points)
            
            flux_a_interp = np.interp(wave_common, wave_a, flux_a)
            flux_b_interp = np.interp(wave_common, wave_b, flux_b)
            
            # Compute CCF
            ccf = compute_ccf(flux_a_interp, flux_b_interp, velocities, wave_common)
            ccf_dict[(species_a, species_b)] = ccf
    
    return ccf_dict, velocities


def identify_aliased_species(
    ccf_dict: dict[tuple[str, str], np.ndarray],
    velocities: np.ndarray,
    threshold: float = 0.3,
    velocity_window: float = 10.0,
) -> list[tuple[str, str, float, float]]:
    """Identify species pairs with significant cross-correlation near v=0.
    
    Args:
        ccf_dict: Dict from compute_template_cross_correlation
        velocities: Velocity array
        threshold: Correlation threshold for flagging (default 0.3)
        velocity_window: Window around v=0 to check (km/s, default 10)
        
    Returns:
        List of (species_A, species_B, peak_correlation, peak_velocity) tuples
        for pairs exceeding threshold.
    """
    aliased_pairs = []
    
    # Find indices within velocity window
    vel_mask = np.abs(velocities) <= velocity_window
    
    for (species_a, species_b), ccf in ccf_dict.items():
        # Find peak in velocity window
        ccf_window = ccf[vel_mask]
        vel_window = velocities[vel_mask]
        
        if len(ccf_window) == 0:
            continue
        
        peak_idx = np.argmax(np.abs(ccf_window))
        peak_corr = ccf_window[peak_idx]
        peak_vel = vel_window[peak_idx]
        
        if np.abs(peak_corr) >= threshold:
            aliased_pairs.append((species_a, species_b, peak_corr, peak_vel))
    
    # Sort by correlation strength (descending)
    aliased_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return aliased_pairs


def build_aliasing_matrix(
    ccf_dict: dict[tuple[str, str], np.ndarray],
    velocities: np.ndarray,
    species_names: list[str],
) -> np.ndarray:
    """Build matrix of peak cross-correlations between all species.
    
    Args:
        ccf_dict: Dict from compute_template_cross_correlation
        velocities: Velocity array
        species_names: Ordered list of species names
        
    Returns:
        Square matrix of peak correlations (symmetric with 1.0 on diagonal)
    """
    n = len(species_names)
    matrix = np.eye(n)  # Diagonal is 1.0 (self-correlation)
    
    # Find index of v=0 (or closest)
    v0_idx = np.argmin(np.abs(velocities))
    
    for i, sp_a in enumerate(species_names):
        for j, sp_b in enumerate(species_names):
            if i >= j:
                continue
            
            # Try both orderings
            key = (sp_a, sp_b) if (sp_a, sp_b) in ccf_dict else (sp_b, sp_a)
            if key in ccf_dict:
                # Use peak correlation near v=0
                ccf = ccf_dict[key]
                # Search in ±10 km/s window
                window = 10
                v_min_idx = max(0, v0_idx - window)
                v_max_idx = min(len(ccf), v0_idx + window)
                peak_corr = np.max(np.abs(ccf[v_min_idx:v_max_idx]))
                
                matrix[i, j] = peak_corr
                matrix[j, i] = peak_corr
    
    return matrix


def generate_aliasing_report(
    templates: dict[str, tuple[np.ndarray, np.ndarray]],
    threshold: float = 0.3,
    velocity_range: tuple[float, float] = (-50, 50),
    output_path: str | None = None,
) -> dict:
    """Generate comprehensive aliasing diagnostic report.
    
    Args:
        templates: Dict mapping species name to (wavelength, flux) tuple
        threshold: Correlation threshold for flagging
        velocity_range: Velocity range for CCF computation
        output_path: If provided, write text report to this path
        
    Returns:
        Dict with 'ccf_dict', 'velocities', 'aliased_pairs', 'matrix', 'species'
    """
    species_names = sorted(templates.keys())
    
    print(f"Computing cross-correlations for {len(species_names)} species...")
    ccf_dict, velocities = compute_template_cross_correlation(
        templates, velocity_range=velocity_range
    )
    
    aliased_pairs = identify_aliased_species(ccf_dict, velocities, threshold=threshold)
    matrix = build_aliasing_matrix(ccf_dict, velocities, species_names)
    
    result = {
        "ccf_dict": ccf_dict,
        "velocities": velocities,
        "aliased_pairs": aliased_pairs,
        "matrix": matrix,
        "species": species_names,
    }
    
    # Generate text report
    if output_path or True:  # Always print to console
        lines = []
        lines.append("=" * 70)
        lines.append("SPECIES ALIASING DIAGNOSTIC REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Number of species analyzed: {len(species_names)}")
        lines.append(f"Correlation threshold: {threshold}")
        lines.append(f"Velocity range: {velocity_range[0]} to {velocity_range[1]} km/s")
        lines.append("")
        
        if aliased_pairs:
            lines.append("WARNING: Potential aliasing detected!")
            lines.append("-" * 70)
            for sp_a, sp_b, corr, vel in aliased_pairs:
                lines.append(f"  {sp_a} <-> {sp_b}: r = {corr:.3f} at v = {vel:.1f} km/s")
            lines.append("")
            lines.append("These species pairs have significant template overlap.")
            lines.append("Detections of the weaker species may be affected by the stronger one.")
        else:
            lines.append("No significant aliasing detected (all correlations < threshold).")
        
        lines.append("")
        lines.append("=" * 70)
        
        report_text = "\n".join(lines)
        print(report_text)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_path}")
    
    return result


def check_aliasing_with_fe(
    templates: dict[str, tuple[np.ndarray, np.ndarray]],
    reference_species: str = "Fe",
    threshold: float = 0.3,
) -> list[tuple[str, float, float]]:
    """Check aliasing of all species against a strong reference (usually Fe I).
    
    Fe I typically has the most lines in optical spectra and is the most
    common source of aliasing for weaker species.
    
    Args:
        templates: Dict mapping species name to (wavelength, flux) tuple
        reference_species: Reference species to check against (default "Fe")
        threshold: Correlation threshold
        
    Returns:
        List of (species, correlation, velocity) tuples for aliased species
    """
    # Find the reference species (try variations)
    ref_key = None
    for key in templates.keys():
        if reference_species in key:
            ref_key = key
            break
    
    if ref_key is None:
        print(f"Warning: Reference species '{reference_species}' not found in templates")
        return []
    
    ref_wave, ref_flux = templates[ref_key]
    velocities = np.arange(-50, 51, 1.0)
    
    aliased = []
    
    for species, (wave, flux) in templates.items():
        if species == ref_key:
            continue
        
        # Interpolate onto common grid
        wave_min = max(wave.min(), ref_wave.min())
        wave_max = min(wave.max(), ref_wave.max())
        wave_common = np.linspace(wave_min, wave_max, max(len(wave), len(ref_wave)))
        
        flux_interp = np.interp(wave_common, wave, flux)
        ref_flux_interp = np.interp(wave_common, ref_wave, ref_flux)
        
        ccf = compute_ccf(ref_flux_interp, flux_interp, velocities, wave_common)
        
        # Check near v=0
        v0_mask = np.abs(velocities) <= 10
        ccf_near_zero = ccf[v0_mask]
        vel_near_zero = velocities[v0_mask]
        
        peak_idx = np.argmax(np.abs(ccf_near_zero))
        peak_corr = ccf_near_zero[peak_idx]
        peak_vel = vel_near_zero[peak_idx]
        
        if np.abs(peak_corr) >= threshold:
            aliased.append((species, peak_corr, peak_vel))
    
    aliased.sort(key=lambda x: abs(x[1]), reverse=True)
    return aliased
