"""Emission-specific diagnostics for prepared high-resolution time series.

The helpers in :mod:`plotting.transmission_diagnostics` provide the shared,
coverage-aware line sampling and species stacks.  This module adds views that
are specific to dayside observations: secondary-eclipse phase coverage,
pre/post-eclipse comparisons, cross-epoch repeatability, arm alignment, and
available SYSREM provenance.  These are diagnostic visualizations only; none
of the summaries is used by the retrieval likelihood or presented as a formal
detection statistic.
"""

from __future__ import annotations

import math

from plotting.style import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np

from plotting.transmission_diagnostics import (
    DEFAULT_SPECIES,
    TRANSMISSION_LINE_CATALOG,
    species_frame_stack_results,
)


def _phase_mod1(bundle):
    return np.mod(np.asarray(bundle["phase"], dtype=float), 1.0)


def _robust_limit(values, percentile=99.0, floor=2e-5):
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(floor)
    value = float(np.nanpercentile(finite, percentile))
    if not np.isfinite(value) or value <= 0:
        return float(floor)
    return max(float(floor), value)


def phase_coverage_rows(bundles):
    """Return per-bundle dayside phase-coverage records."""

    rows = []
    for bundle in bundles:
        phase = _phase_mod1(bundle)
        finite = phase[np.isfinite(phase)]
        if finite.size == 0:
            continue
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if lo <= 0.5 <= hi:
            eclipse_side = "crosses 0.5"
        elif hi < 0.5:
            eclipse_side = "pre-eclipse"
        else:
            eclipse_side = "post-eclipse"
        visibility = 0.5 * (1.0 - np.cos(2.0 * np.pi * finite))
        rows.append(
            {
                "epoch": bundle.get("epoch"),
                "arm": bundle.get("arm"),
                "n_exp": int(finite.size),
                "phase_min": lo,
                "phase_max": hi,
                "phase_mean": float(np.nanmean(finite)),
                "eclipse_side": eclipse_side,
                "mean_dayside_visibility_proxy": float(np.nanmean(visibility)),
            }
        )
    return rows


def plot_dayside_phase_coverage(bundles):
    """Plot every exposure and secondary eclipse on one orbital circle."""

    records = []
    for bundle in bundles:
        phase = _phase_mod1(bundle)
        phase = phase[np.isfinite(phase)]
        if phase.size:
            records.append((bundle, np.sort(phase)))
    if not records:
        return None

    fig, ax = plt.subplots(
        figsize=(7.2, 7.2),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    colors = {"blue": "tab:blue", "red": "tab:red"}
    labeled_arms = set()
    for bundle, phase in records:
        arm = str(bundle.get("arm"))
        color = colors.get(arm, "0.25")
        label = f"{arm} arm" if arm not in labeled_arms else None
        ax.scatter(
            2.0 * np.pi * phase,
            np.ones(phase.size),
            s=18,
            color=color,
            alpha=0.42,
            linewidths=0,
            label=label,
            zorder=3,
        )
        labeled_arms.add(arm)

    orbit_angle = np.linspace(0.0, 2.0 * np.pi, 720)
    ax.plot(orbit_angle, np.ones_like(orbit_angle), color="0.35", lw=1.0, zorder=1)
    ax.scatter(
        [np.pi],
        [1.0],
        marker="*",
        s=180,
        color="tab:purple",
        edgecolor="white",
        linewidth=0.8,
        label="secondary eclipse center",
        zorder=5,
    )

    phase_ticks = np.arange(0.0, 1.0, 0.125)
    ax.set_xticks(2.0 * np.pi * phase_ticks)
    ax.set_xticklabels([f"{phase:.3g}" for phase in phase_ticks])
    # View the orbit from an observer below the page looking upward: transit
    # (phase 0) is on the near side at the bottom and secondary eclipse
    # (phase 0.5) is on the far side at the top.
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.1)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_color("0.7")
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_title("Exposure coverage on the orbital phase circle", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)
    ax.annotate(
        "observer line of sight",
        xy=(0.5, 0.16),
        xytext=(0.5, -0.08),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8,
        color="0.25",
        arrowprops={"arrowstyle": "-|>", "color": "0.25", "lw": 1.0},
        annotation_clip=False,
    )
    return fig, ax


def arm_phase_alignment_rows(bundles):
    """Summarize blue/red sampling mismatches within each epoch."""

    grouped = {}
    for bundle in bundles:
        grouped.setdefault(str(bundle.get("epoch")), {})[str(bundle.get("arm"))] = _phase_mod1(bundle)

    rows = []
    for epoch in sorted(grouped):
        arms = grouped[epoch]
        if "blue" not in arms or "red" not in arms:
            continue
        blue = np.sort(arms["blue"][np.isfinite(arms["blue"])])
        red = np.sort(arms["red"][np.isfinite(arms["red"])])
        if blue.size == 0 or red.size == 0:
            continue
        intersection = max(0.0, min(float(blue[-1]), float(red[-1])) - max(float(blue[0]), float(red[0])))
        union = max(float(blue[-1]), float(red[-1])) - min(float(blue[0]), float(red[0]))
        nearest_blue = np.min(np.abs(blue[:, None] - red[None, :]), axis=1)
        nearest_red = np.min(np.abs(red[:, None] - blue[None, :]), axis=1)
        rows.append(
            {
                "epoch": epoch,
                "n_blue": int(blue.size),
                "n_red": int(red.size),
                "n_exp_delta": int(blue.size - red.size),
                "blue_phase_min": float(blue[0]),
                "blue_phase_max": float(blue[-1]),
                "red_phase_min": float(red[0]),
                "red_phase_max": float(red[-1]),
                "phase_span_jaccard": float(intersection / union) if union > 0 else 1.0,
                "median_nearest_phase_delta": float(np.nanmedian(np.concatenate([nearest_blue, nearest_red]))),
            }
        )
    return rows


def plot_arm_phase_alignment(bundles):
    """Plot blue/red exposure sampling on aligned epoch lanes."""

    grouped = {}
    for bundle in bundles:
        grouped.setdefault(str(bundle.get("epoch")), {})[str(bundle.get("arm"))] = np.sort(_phase_mod1(bundle))
    epochs = [epoch for epoch in sorted(grouped) if "blue" in grouped[epoch] and "red" in grouped[epoch]]
    if not epochs:
        return None

    all_phase = np.concatenate([grouped[epoch][arm] for epoch in epochs for arm in ("blue", "red")])
    lo = max(0.0, min(float(np.nanmin(all_phase)) - 0.02, 0.48))
    hi = min(1.0, max(float(np.nanmax(all_phase)) + 0.02, 0.52))
    fig, ax = plt.subplots(figsize=(12.5, max(3.5, 0.72 * len(epochs) + 1.7)), constrained_layout=True)
    offsets = {"blue": -0.14, "red": 0.14}
    colors = {"blue": "tab:blue", "red": "tab:red"}
    for index, epoch in enumerate(epochs):
        for arm in ("blue", "red"):
            phase = grouped[epoch][arm]
            finite = phase[np.isfinite(phase)]
            if finite.size == 0:
                continue
            y = index + offsets[arm]
            ax.plot([finite[0], finite[-1]], [y, y], color=colors[arm], lw=1.25, alpha=0.8)
            ax.scatter(
                finite,
                np.full(finite.size, y),
                s=14,
                color=colors[arm],
                marker="o" if arm == "blue" else "s",
                alpha=0.72,
                label=arm if index == 0 else None,
            )
    ax.axvline(0.5, color="tab:purple", lw=1.0, ls="--", label="secondary eclipse center")
    ax.set_xlim(lo, hi)
    ax.set_yticks(np.arange(len(epochs)))
    ax.set_yticklabels(epochs)
    ax.invert_yaxis()
    ax.set_xlabel("Orbital phase (mod 1)")
    ax.set_ylabel("Epoch")
    ax.set_title("Blue/red arm phase alignment and exposure-count coverage")
    ax.legend(loc="best")
    return fig, ax


def _display_standardize_profile(velocity, profile, core_kms=35.0):
    velocity = np.asarray(velocity, dtype=float)
    profile = np.asarray(profile, dtype=float)
    reference = np.isfinite(profile) & (np.abs(velocity) >= float(core_kms))
    if np.count_nonzero(reference) < 8:
        reference = np.isfinite(profile)
    if np.count_nonzero(reference) == 0:
        return np.full(profile.shape, np.nan)
    center = float(np.nanmedian(profile[reference]))
    mad = float(1.4826 * np.nanmedian(np.abs(profile[reference] - center)))
    if not np.isfinite(mad) or mad <= 0:
        mad = float(np.nanstd(profile[reference]))
    if not np.isfinite(mad) or mad <= 0:
        return np.full(profile.shape, np.nan)
    return (profile - center) / mad


def plot_epoch_species_coherence(
    bundles,
    *,
    line_catalog=TRANSMISSION_LINE_CATALOG,
    species=DEFAULT_SPECIES,
    window_kms=150.0,
    bin_kms=2.0,
    sigma_threshold=0.5,
    kp_kms=None,
    eccentricity=0.0,
    omega_deg=None,
    ncols=3,
):
    """Plot display-standardized species profiles as bundle-by-velocity maps."""

    by_species = {name: [] for name in species}
    for bundle in bundles:
        results = species_frame_stack_results(
            bundle,
            line_catalog=line_catalog,
            species=species,
            window_kms=window_kms,
            bin_kms=bin_kms,
            sigma_threshold=sigma_threshold,
            kp_kms=kp_kms,
            vsys_kms=0.0,
            eccentricity=eccentricity,
            omega_deg=omega_deg,
        )
        phase = _phase_mod1(bundle)
        phase_mean = float(np.nanmean(phase)) if np.any(np.isfinite(phase)) else np.nan
        for result in results:
            by_species[result["species"]].append(
                {
                    "bundle": bundle,
                    "result": result,
                    "phase_mean": phase_mean,
                    "standardized": _display_standardize_profile(
                        result["velocity"], result["mean"]
                    ),
                }
            )
    available = [(name, by_species[name]) for name in species if by_species.get(name)]
    if not available:
        return None

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(len(available) / ncols))
    max_bundle_rows = max(len(records) for _, records in available)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, max(3.0 * nrows, (1.9 + 0.30 * max_bundle_rows) * nrows)),
        constrained_layout=True,
        squeeze=False,
    )
    images = []
    for ax, (species_name, records) in zip(axes.ravel(), available):
        records = sorted(records, key=lambda item: (item["phase_mean"], item["bundle"].get("epoch"), item["bundle"].get("arm")))
        matrix = np.stack([item["standardized"] for item in records], axis=0)
        velocity = records[0]["result"]["velocity"]
        finite = np.abs(matrix[np.isfinite(matrix)])
        vmax = 3.0 if finite.size == 0 else float(np.clip(np.nanpercentile(finite, 98.5), 3.0, 8.0))
        image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[float(velocity[0]), float(velocity[-1]), len(records) - 0.5, -0.5],
        )
        images.append(image)
        ax.axvline(0.0, color="0.2", lw=0.75, ls=":")
        ax.set_yticks(np.arange(len(records)))
        ax.set_yticklabels(
            [
                f"{item['bundle'].get('epoch')} {item['bundle'].get('arm')}  phi={item['phase_mean']:.3f}"
                f"  cov0={item['result']['center_coverage_fraction']:.0%}"
                for item in records
            ],
            fontsize=6.7,
        )
        ax.set_title(f"{species_name}: cross-epoch profile coherence", fontsize=10)
        ax.set_xlabel("Velocity relative to vacuum rest wavelength [km/s]")
    for ax in axes.ravel()[len(available) :]:
        ax.set_axis_off()
    if images:
        fig.colorbar(
            images[-1],
            ax=[ax for ax in axes.ravel() if ax.axison],
            pad=0.012,
            fraction=0.018,
            label="Profile / off-line MAD scale (display only)",
        )
    frame = "planet frame" if kp_kms is not None and np.isfinite(float(kp_kms)) else "line rest frame"
    fig.suptitle(
        f"Species repeatability across prepared bundles ({frame}); rows are independently standardized",
        fontsize=13,
    )
    return fig, axes, by_species


def _subset_bundle(bundle, keep):
    keep = np.asarray(keep, dtype=bool)
    subset = dict(bundle)
    for key in ("data", "sigma", "phase"):
        subset[key] = np.asarray(bundle[key])[keep]
    return subset


def _combine_profiles(records):
    if not records:
        return None
    velocity = np.asarray(records[0]["velocity"], dtype=float)
    means = np.stack([record["mean"] for record in records], axis=0)
    errors = np.stack([record["error"] for record in records], axis=0)
    valid = np.isfinite(means) & np.isfinite(errors) & (errors > 0)
    weights = np.where(valid, 1.0 / np.square(errors), 0.0)
    weight_sum = np.sum(weights, axis=0)
    mean = np.divide(
        np.sum(np.where(valid, means * weights, 0.0), axis=0),
        weight_sum,
        out=np.full(velocity.shape, np.nan),
        where=weight_sum > 0,
    )
    error = np.divide(
        1.0,
        np.sqrt(weight_sum),
        out=np.full(velocity.shape, np.nan),
        where=weight_sum > 0,
    )
    return velocity, mean, error


def plot_eclipse_side_species_comparison(
    bundles,
    *,
    line_catalog=TRANSMISSION_LINE_CATALOG,
    species=DEFAULT_SPECIES,
    window_kms=150.0,
    bin_kms=2.0,
    sigma_threshold=0.5,
    kp_kms=None,
    eccentricity=0.0,
    omega_deg=None,
    minimum_exposures=4,
    ncols=3,
):
    """Compare target-wide pre- and post-eclipse species stacks."""

    side_results = {
        "pre": {name: [] for name in species},
        "post": {name: [] for name in species},
    }
    side_exposures = {"pre": {name: 0 for name in species}, "post": {name: 0 for name in species}}
    for bundle in bundles:
        phase = _phase_mod1(bundle)
        for side, keep in (("pre", phase < 0.5), ("post", phase > 0.5)):
            if np.count_nonzero(keep) < int(minimum_exposures):
                continue
            subset = _subset_bundle(bundle, keep)
            results = species_frame_stack_results(
                subset,
                line_catalog=line_catalog,
                species=species,
                window_kms=window_kms,
                bin_kms=bin_kms,
                sigma_threshold=sigma_threshold,
                kp_kms=kp_kms,
                vsys_kms=0.0,
                eccentricity=eccentricity,
                omega_deg=omega_deg,
            )
            for result in results:
                side_results[side][result["species"]].append(result)
                side_exposures[side][result["species"]] += int(np.count_nonzero(keep))

    comparable = []
    for species_name in species:
        pre = _combine_profiles(side_results["pre"].get(species_name, []))
        post = _combine_profiles(side_results["post"].get(species_name, []))
        if pre is not None and post is not None:
            comparable.append((species_name, pre, post))
    if not comparable:
        print("No species has sufficient coverage on both sides of secondary eclipse.")
        return None

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(len(comparable) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 2.9 * nrows),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (species_name, pre, post) in zip(axes.ravel(), comparable):
        velocity, pre_mean, pre_error = pre
        _, post_mean, post_error = post
        for mean, error, color, label in (
            (
                pre_mean,
                pre_error,
                "tab:blue",
                f"pre (n={side_exposures['pre'][species_name]})",
            ),
            (
                post_mean,
                post_error,
                "tab:red",
                f"post (n={side_exposures['post'][species_name]})",
            ),
        ):
            ax.plot(velocity, mean, color=color, lw=1.0, label=label)
            ax.fill_between(velocity, mean - error, mean + error, color=color, alpha=0.14, lw=0)
        ax.axhline(0.0, color="0.4", lw=0.7, ls="--")
        ax.axvline(0.0, color="0.25", lw=0.7, ls=":")
        limit = _robust_limit(np.concatenate([pre_mean, post_mean]), percentile=98.8)
        ax.set_ylim(-limit, limit)
        ax.set_title(species_name)
        ax.set_ylabel("Residual")
        ax.legend(loc="best")
    for ax in axes.ravel()[len(comparable) :]:
        ax.set_axis_off()
    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel("Velocity relative to vacuum rest wavelength [km/s]")
    frame = "planet frame" if kp_kms is not None and np.isfinite(float(kp_kms)) else "line rest frame"
    fig.suptitle(
        f"Pre/post-secondary-eclipse species comparison ({frame}; diagnostic, not a detection test)",
        fontsize=13,
    )
    return fig, axes, side_results


def plot_sysrem_component_diagnostics(bundles):
    """Visualize accepted basis counts and scatter reduction from U_sysrem files."""

    records = []
    chunk_names = []
    for bundle in bundles:
        z = bundle.get("sysrem")
        if z is None or "basis_counts" not in z.files:
            continue
        names = (
            [str(value) for value in np.asarray(z["chunk_names"]).tolist()]
            if "chunk_names" in z.files
            else [f"chunk_{index}" for index in range(np.asarray(z["basis_counts"]).size)]
        )
        for name in names:
            if name not in chunk_names:
                chunk_names.append(name)
        records.append((bundle, z, names))
    if not records:
        return None

    labels = [f"{bundle.get('epoch')} {bundle.get('arm')}" for bundle, _, _ in records]
    counts = np.full((len(records), len(chunk_names)), np.nan)
    for row, (_, z, names) in enumerate(records):
        basis_counts = np.asarray(z["basis_counts"], dtype=float).ravel()
        for index, name in enumerate(names):
            if index < basis_counts.size:
                counts[row, chunk_names.index(name)] = basis_counts[index]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, max(5.0, 0.38 * len(records) + 2.1)), constrained_layout=True)
    masked_counts = np.ma.masked_invalid(counts)
    image = axes[0].imshow(masked_counts, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[0].set_xticks(np.arange(len(chunk_names)))
    axes[0].set_xticklabels(chunk_names, rotation=25, ha="right")
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7.5)
    axes[0].set_title("Accepted SYSREM basis counts")
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            if np.isfinite(counts[row, column]):
                axes[0].text(column, row, f"{counts[row, column]:.0f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=axes[0], pad=0.012, fraction=0.045, label="Accepted components")

    epoch_colors = {}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["0.2"])
    for bundle, z, names in records:
        epoch = str(bundle.get("epoch"))
        if epoch not in epoch_colors:
            epoch_colors[epoch] = color_cycle[len(epoch_colors) % len(color_cycle)]
        before = np.asarray(z["sysrem_stddev_before"], dtype=float) if "sysrem_stddev_before" in z.files else None
        after = np.asarray(z["sysrem_stddev_after"], dtype=float) if "sysrem_stddev_after" in z.files else None
        accepted = np.asarray(z["sysrem_component_accepted"], dtype=bool) if "sysrem_component_accepted" in z.files else None
        if before is None or after is None:
            continue
        before = np.atleast_2d(before)
        after = np.atleast_2d(after)
        if accepted is None:
            accepted = np.isfinite(after)
        accepted = np.atleast_2d(accepted)
        target_ax = axes[1] if bundle.get("arm") == "blue" else axes[2]
        for chunk_index, name in enumerate(names):
            if chunk_index >= before.shape[1] or chunk_index >= after.shape[1]:
                continue
            keep = accepted[:, chunk_index] & np.isfinite(after[:, chunk_index])
            component = np.flatnonzero(keep) + 1
            if component.size == 0:
                continue
            baseline = before[component[0] - 1, chunk_index]
            if not np.isfinite(baseline) or baseline <= 0:
                continue
            remaining = after[keep, chunk_index] / baseline
            target_ax.plot(
                component,
                remaining,
                marker="o" if name == "non_telluric" else "s",
                ms=3,
                lw=0.9,
                color=epoch_colors[epoch],
                ls="-" if name == "non_telluric" else "--",
                label=f"{epoch} {name}",
            )
    for ax, arm in zip(axes[1:], ("blue", "red")):
        ax.axhline(1.0, color="0.35", lw=0.8, ls=":")
        ax.set_xlabel("Accepted component number")
        ax.set_ylabel("Residual scatter / initial scatter")
        ax.set_title(f"{arm.title()}-arm scatter reduction")
        if ax.lines:
            ax.legend(loc="best", fontsize=6.5)
    fig.suptitle("Persisted SYSREM component diagnostics", fontsize=13)
    return fig, axes


def plot_time_domain_coherence(bundle, *, sigma_threshold=0.5, max_wavelength_points=2400):
    """Plot exposure-to-exposure spectral correlation for one bundle."""

    data = np.asarray(bundle["data"], dtype=float)
    sigma = np.asarray(bundle["sigma"], dtype=float)
    phase = _phase_mod1(bundle)
    order = np.argsort(phase)
    data = data[order]
    sigma = sigma[order]
    phase = phase[order]
    if data.shape[0] < 3:
        return None
    if data.shape[1] > int(max_wavelength_points):
        index = np.unique(np.linspace(0, data.shape[1] - 1, int(max_wavelength_points)).astype(int))
        data = data[:, index]
        sigma = sigma[:, index]
    valid = np.isfinite(data) & np.isfinite(sigma) & (sigma > 0) & (sigma < float(sigma_threshold))
    row_rms = np.sqrt(
        np.divide(
            np.sum(np.where(valid, np.square(data), 0.0), axis=1),
            np.sum(valid, axis=1),
            out=np.full(data.shape[0], np.nan),
            where=np.sum(valid, axis=1) > 0,
        )
    )
    correlation = np.full((data.shape[0], data.shape[0]), np.nan)
    for left in range(data.shape[0]):
        for right in range(left, data.shape[0]):
            shared = valid[left] & valid[right]
            if np.count_nonzero(shared) < 20:
                continue
            x = data[left, shared]
            y = data[right, shared]
            x = x - np.nanmean(x)
            y = y - np.nanmean(y)
            denom = float(np.sqrt(np.sum(np.square(x)) * np.sum(np.square(y))))
            value = float(np.sum(x * y) / denom) if denom > 0 else np.nan
            correlation[left, right] = value
            correlation[right, left] = value
    lag_one = np.diag(correlation, k=1)
    midpoint = 0.5 * (phase[:-1] + phase[1:])
    phase_step = np.diff(phase)
    positive_step = phase_step[np.isfinite(phase_step) & (phase_step > 0)]
    typical_step = float(np.nanmedian(positive_step)) if positive_step.size else np.nan
    gap_limit = max(0.01, 5.0 * typical_step) if np.isfinite(typical_step) else 0.01
    gap_after = np.flatnonzero(phase_step > gap_limit)
    groups = np.split(np.arange(phase.size), gap_after + 1)
    lag_one = np.where(phase_step <= gap_limit, lag_one, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)
    for group_index, group in enumerate(groups):
        axes[0].plot(
            phase[group],
            row_rms[group],
            "o-",
            ms=3,
            lw=0.85,
            color="tab:blue",
            label="contiguous phase segment" if group_index == 0 and len(groups) > 1 else None,
        )
    axes[0].set_xlabel("Orbital phase (mod 1)")
    axes[0].set_ylabel("Residual RMS")
    axes[0].set_title("Exposure-wise residual scale")
    if len(groups) > 1:
        axes[0].legend(loc="best")

    image = axes[1].imshow(correlation, vmin=-1.0, vmax=1.0, cmap="RdBu_r", interpolation="nearest")
    ticks = np.unique(np.linspace(0, phase.size - 1, min(8, phase.size)).astype(int))
    axes[1].set_xticks(ticks)
    axes[1].set_yticks(ticks)
    axes[1].set_xticklabels([f"{phase[index]:.3f}" for index in ticks], rotation=45, ha="right")
    axes[1].set_yticklabels([f"{phase[index]:.3f}" for index in ticks])
    axes[1].set_xlabel("Orbital phase (mod 1)")
    axes[1].set_ylabel("Orbital phase (mod 1)")
    axes[1].set_title("Exposure residual correlation")
    fig.colorbar(image, ax=axes[1], pad=0.012, fraction=0.045, label="Pearson correlation")

    axes[2].plot(midpoint, lag_one, "o-", ms=3, lw=0.85, color="tab:purple")
    axes[2].axhline(0.0, color="0.35", lw=0.8, ls="--")
    axes[2].set_ylim(-1.03, 1.03)
    axes[2].set_xlabel("Midpoint orbital phase (mod 1)")
    axes[2].set_ylabel("Adjacent-exposure correlation")
    axes[2].set_title("Lag-one spectral coherence (phase gaps excluded)")
    fig.suptitle(f"Time-domain residual coherence: {bundle.get('epoch')} {bundle.get('arm')}", fontsize=13)
    return fig, axes


__all__ = [
    "arm_phase_alignment_rows",
    "phase_coverage_rows",
    "plot_arm_phase_alignment",
    "plot_dayside_phase_coverage",
    "plot_eclipse_side_species_comparison",
    "plot_epoch_species_coherence",
    "plot_sysrem_component_diagnostics",
    "plot_time_domain_coherence",
]
