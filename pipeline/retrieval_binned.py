from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import config
from config.planets_config import PHASE_BINS, get_params
from dataio.make_transmission import filter_data_by_phase, summarize_phase_coverage
from pipeline.retrieval import load_timeseries_data, run_retrieval, _normalize_phase


def run_phase_binned_retrieval(
    phase_bins: list[str] = ["T12", "T23", "T34"],
    base_output_dir: str = "output/phase_binned",
    mode: str = "transmission",
    epoch: str | None = None,
    data_dir: str | Path | None = None,
    data_format: str = "auto",
    **retrieval_kwargs,
) -> dict[str, dict]:
    if mode != "transmission":
        raise ValueError("Phase-binned retrieval is only supported for transmission mode.")

    if data_format == "spectrum":
        raise ValueError("Phase-binned retrieval requires time-series data.")

    resolved_data_dir = Path(data_dir) if data_dir is not None else config.get_data_dir(epoch=epoch)

    try:
        wav_obs, data, sigma, phase = load_timeseries_data(resolved_data_dir)
    except FileNotFoundError as exc:
        raise ValueError(
            "Phase-binned retrieval requires time-series files (wavelength.npy, data.npy, sigma.npy, phase.npy). "
            f"Missing in {resolved_data_dir}."
        ) from exc

    phase = _normalize_phase(phase)
    params = retrieval_kwargs.get("params", get_params())
    
    # Validate phase bins
    for bin_name in phase_bins:
        if bin_name not in PHASE_BINS:
            raise ValueError(f"Unknown phase bin: {bin_name}. Available: {list(PHASE_BINS.keys())}")
    
    # Print phase coverage summary
    coverage = summarize_phase_coverage(phase, params)
    print("\n" + "=" * 70)
    print("PHASE-BINNED RETRIEVAL")
    print("=" * 70)
    print(f"\nTotal exposures: {coverage['total_exposures']}")
    print(f"In-transit exposures: {coverage['total_in_transit']}")
    print("\nPhase bin coverage:")
    for bin_name in phase_bins:
        info = coverage["bins"].get(bin_name, {})
        count = info.get("count", 0)
        if count > 0:
            print(f"  {bin_name} ({PHASE_BINS[bin_name]}): {count} exposures")
            print(f"    Phase range: {info['phase_min']:.4f} to {info['phase_max']:.4f}")
        else:
            print(f"  {bin_name}: NO EXPOSURES (will skip)")
    print("=" * 70 + "\n")
    
    results = {}
    
    for bin_name in phase_bins:
        bin_info = coverage["bins"].get(bin_name, {})
        if bin_info.get("count", 0) == 0:
            print(f"\nSkipping {bin_name}: no exposures in this phase bin")
            results[bin_name] = None
            continue
        
        print(f"\n{'='*70}")
        print(f"Running retrieval for phase bin: {bin_name} ({PHASE_BINS[bin_name]})")
        print(f"{'='*70}")
        
        # Filter data to this phase bin
        data_bin, sigma_bin, phase_bin = filter_data_by_phase(
            data, sigma, phase, bin_name, params
        )
        
        print(f"Filtered to {len(phase_bin)} exposures")
        
        # Set up output directory for this bin
        bin_output_dir = os.path.join(base_output_dir, bin_name)
        os.makedirs(bin_output_dir, exist_ok=True)
        
        # Run retrieval
        previous_dir = config.DIR_SAVE
        try:
            config.DIR_SAVE = str(bin_output_dir)
            run_retrieval(
                mode=mode,
                epoch=epoch,
                wav_obs=wav_obs,
                data=data_bin,
                sigma=sigma_bin,
                phase=phase_bin,
                **retrieval_kwargs,
            )
            results[bin_name] = {"output_dir": str(bin_output_dir)}
            
            # Save bin-specific info
            np.savez(
                os.path.join(bin_output_dir, "phase_bin_info.npz"),
                bin_name=bin_name,
                n_exposures=len(phase_bin),
                phase_min=float(np.min(phase_bin)),
                phase_max=float(np.max(phase_bin)),
                phase_values=phase_bin,
            )
            
        except Exception as e:
            print(f"ERROR in {bin_name} retrieval: {e}")
            results[bin_name] = {"error": str(e)}
        finally:
            config.DIR_SAVE = previous_dir
    
    # Generate comparison if we have multiple successful retrievals
    successful_bins = [
        b for b, r in results.items()
        if isinstance(r, dict) and "samples" in r and "error" not in r
    ]
    
    if len(successful_bins) >= 2:
        comparison_dir = os.path.join(base_output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        try:
            comparison = compare_phase_posteriors(
                {b: results[b] for b in successful_bins},
                output_dir=comparison_dir,
            )
            results["comparison"] = comparison
        except Exception as e:
            print(f"ERROR generating comparison: {e}")
            results["comparison"] = {"error": str(e)}
    
    return results


def compare_phase_posteriors(
    posteriors: dict[str, dict],
    params_to_compare: list[str] | None = None,
    output_dir: str | None = None,
) -> dict:
    from scipy import stats
    
    bin_names = list(posteriors.keys())
    
    if len(bin_names) < 2:
        return {"error": "Need at least 2 bins for comparison"}
    
    if params_to_compare is None:
        all_params = set()
        for post in posteriors.values():
            if isinstance(post, dict) and "samples" in post:
                all_params.update(post["samples"].keys())
        
        # Filter to interesting parameters
        params_to_compare = [
            p for p in all_params 
            if p.startswith("dRV") or p.startswith("logVMR") or p in ["Kp", "Vsys"]
        ]
    
    comparison = {
        "bins": bin_names,
        "parameters": params_to_compare,
        "statistics": {},
    }
    
    print("\n" + "=" * 70)
    print("PHASE BIN COMPARISON")
    print("=" * 70)
    
    for param in params_to_compare:
        param_stats = {}
        samples_by_bin = {}
        
        # Collect samples for this parameter
        for bin_name in bin_names:
            post = posteriors[bin_name]
            if isinstance(post, dict) and "samples" in post:
                if param in post["samples"]:
                    samples = post["samples"][param]
                    # Handle array-valued samples (e.g., per-exposure dRV)
                    if samples.ndim > 1:
                        samples = samples.mean(axis=-1)  # Take mean across exposures
                    samples_by_bin[bin_name] = samples
        
        if len(samples_by_bin) < 2:
            continue
        
        # Compute summary statistics for each bin
        for bin_name, samples in samples_by_bin.items():
            param_stats[f"{bin_name}_mean"] = float(np.mean(samples))
            param_stats[f"{bin_name}_std"] = float(np.std(samples))
            param_stats[f"{bin_name}_median"] = float(np.median(samples))
            q16, q84 = np.percentile(samples, [16, 84])
            param_stats[f"{bin_name}_q16"] = float(q16)
            param_stats[f"{bin_name}_q84"] = float(q84)
        
        # Pairwise comparisons
        bin_list = list(samples_by_bin.keys())
        for i, bin_a in enumerate(bin_list):
            for bin_b in bin_list[i+1:]:
                samples_a = samples_by_bin[bin_a]
                samples_b = samples_by_bin[bin_b]
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = stats.ks_2samp(samples_a, samples_b)
                param_stats[f"ks_{bin_a}_vs_{bin_b}"] = {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_pval),
                }
                
                # Difference of means (with uncertainty)
                diff_mean = np.mean(samples_a) - np.mean(samples_b)
                # Bootstrap uncertainty on difference
                n_bootstrap = 1000
                diff_samples = []
                for _ in range(n_bootstrap):
                    idx_a = np.random.randint(0, len(samples_a), len(samples_a))
                    idx_b = np.random.randint(0, len(samples_b), len(samples_b))
                    diff_samples.append(np.mean(samples_a[idx_a]) - np.mean(samples_b[idx_b]))
                diff_std = np.std(diff_samples)
                
                param_stats[f"diff_{bin_a}_minus_{bin_b}"] = {
                    "mean": float(diff_mean),
                    "std": float(diff_std),
                    "significance": float(abs(diff_mean) / diff_std) if diff_std > 0 else 0,
                }
        
        comparison["statistics"][param] = param_stats
        
        # Print summary
        print(f"\n{param}:")
        for bin_name in bin_names:
            if bin_name in samples_by_bin:
                mean = param_stats[f"{bin_name}_mean"]
                std = param_stats[f"{bin_name}_std"]
                print(f"  {bin_name}: {mean:.3f} +/- {std:.3f}")
        
        # Print KS test results
        for i, bin_a in enumerate(bin_list):
            for bin_b in bin_list[i+1:]:
                key = f"ks_{bin_a}_vs_{bin_b}"
                if key in param_stats:
                    pval = param_stats[key]["p_value"]
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                    print(f"  KS test {bin_a} vs {bin_b}: p = {pval:.4f} {sig}")
    
    print("=" * 70 + "\n")
    
    # Save comparison results
    if output_dir:
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        comparison_serializable = convert_to_serializable(comparison)
        
        with open(os.path.join(output_dir, "comparison.json"), "w") as f:
            json.dump(comparison_serializable, f, indent=2)
        
        # Also save as text summary
        with open(os.path.join(output_dir, "comparison_summary.txt"), "w") as f:
            f.write("Phase Bin Comparison Summary\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Bins compared: {', '.join(bin_names)}\n")
            f.write(f"Parameters: {', '.join(params_to_compare)}\n\n")
            
            for param, stats in comparison["statistics"].items():
                f.write(f"\n{param}:\n")
                f.write("-" * 40 + "\n")
                for key, value in stats.items():
                    if isinstance(value, dict):
                        f.write(f"  {key}:\n")
                        for k, v in value.items():
                            f.write(f"    {k}: {v}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
        
        print(f"Comparison saved to {output_dir}")
    
    return comparison


def detect_asymmetry(
    posteriors: dict[str, dict],
    param: str = "dRV",
    significance_threshold: float = 2.0,
) -> dict:
    if "T12" not in posteriors or "T34" not in posteriors:
        return {"error": "Need both T12 and T34 bins for asymmetry test"}
    
    try:
        samples_ingress = posteriors["T12"]["samples"][param]
        samples_egress = posteriors["T34"]["samples"][param]
    except (KeyError, TypeError):
        return {"error": f"Parameter {param} not found in posteriors"}
    
    if samples_ingress.ndim > 1:
        samples_ingress = samples_ingress.mean(axis=-1)
    if samples_egress.ndim > 1:
        samples_egress = samples_egress.mean(axis=-1)
    
    # Compute difference distribution
    # Sample with replacement to match array sizes
    n_samples = min(len(samples_ingress), len(samples_egress))
    idx_in = np.random.randint(0, len(samples_ingress), n_samples)
    idx_eg = np.random.randint(0, len(samples_egress), n_samples)
    
    diff = samples_ingress[idx_in] - samples_egress[idx_eg]
    
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)
    significance = abs(diff_mean) / diff_std if diff_std > 0 else 0
    
    # Is difference consistent with zero?
    q16, q50, q84 = np.percentile(diff, [16, 50, 84])
    consistent_with_zero = (q16 <= 0 <= q84)
    
    result = {
        "parameter": param,
        "ingress_mean": float(np.mean(samples_ingress)),
        "ingress_std": float(np.std(samples_ingress)),
        "egress_mean": float(np.mean(samples_egress)),
        "egress_std": float(np.std(samples_egress)),
        "difference_mean": float(diff_mean),
        "difference_std": float(diff_std),
        "significance_sigma": float(significance),
        "asymmetry_detected": significance >= significance_threshold,
        "consistent_with_zero": consistent_with_zero,
        "percentiles": {
            "q16": float(q16),
            "q50": float(q50),
            "q84": float(q84),
        },
    }
    
    # Interpretation
    if result["asymmetry_detected"]:
        if diff_mean > 0:
            result["interpretation"] = f"{param} is larger at ingress than egress"
        else:
            result["interpretation"] = f"{param} is larger at egress than ingress"
    else:
        result["interpretation"] = f"No significant asymmetry in {param}"
    
    return result
