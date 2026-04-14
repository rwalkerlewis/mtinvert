"""
Publication-quality visualization for moment tensor solutions.

Produces:
1. Waveform fit plots (observed vs. synthetic, all stations/components)
2. Focal mechanism beachball
3. Depth-misfit curve
4. Station map
5. Decomposition summary
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyArrowPatch

from .decomposition import MTDecomposition
from .inversion import DepthSearchResult, InversionResult
from .preprocess import StationData

logger = logging.getLogger(__name__)

# Style defaults
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 150,
})


def _draw_beachball(ax, mt_vector: np.ndarray, color_pos="k", color_neg="w"):
    """
    Draw a focal mechanism beachball on the given axes using
    the equal-area projection of the radiation pattern.
    """
    from .decomposition import vector_to_tensor

    M = vector_to_tensor(mt_vector)

    n_theta = 180
    n_phi = 360
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    # Direction cosines (lower hemisphere)
    gamma = np.array([
        np.sin(THETA) * np.cos(PHI),
        np.sin(THETA) * np.sin(PHI),
        np.cos(THETA),
    ])

    # P-wave radiation pattern
    amplitude = np.zeros_like(THETA)
    for i in range(3):
        for j in range(3):
            amplitude += gamma[i] * M[i, j] * gamma[j]

    # Equal-area projection (lower hemisphere only)
    mask = THETA <= np.pi / 2
    r = np.sqrt(2) * np.sin(THETA / 2)
    x = r * np.cos(PHI)  # NOTE: phi measured from East
    y = r * np.sin(PHI)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw filled contour
    ax.contourf(
        x, y, np.where(mask, amplitude, np.nan),
        levels=[-1e30, 0, 1e30],
        colors=[color_neg, color_pos],
    )

    # Outline circle
    circ_theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circ_theta), np.sin(circ_theta), "k-", linewidth=1.5)


def plot_waveform_fits(
    result: InversionResult,
    stations: list[StationData],
    output_path: str | Path,
    event_id: str = "",
) -> None:
    """
    Plot observed vs. synthetic waveforms for all stations and components.
    """
    accepted = [s for s in stations if not s.rejected]
    n_sta = len(accepted)
    if n_sta == 0:
        return

    fig, axes = plt.subplots(n_sta, 3, figsize=(14, 2.5 * n_sta + 1.5))
    if n_sta == 1:
        axes = axes[np.newaxis, :]

    comps = ["Z", "R", "T"]
    dec = result.decomposition

    fig.suptitle(
        f"{event_id}  Depth={result.depth_km:.1f} km  "
        f"Mw={dec.moment_magnitude:.2f}  "
        f"VR={result.variance_reduction:.1%}  "
        f"DC={dec.dc_percent:.0f}%",
        fontsize=12, fontweight="bold", y=0.98,
    )

    for i, sta in enumerate(accepted):
        key = f"{sta.network}.{sta.station}"
        for j, comp in enumerate(comps):
            ax = axes[i, j]

            obs = result.observed.get(key, {}).get(comp)
            syn = result.synthetic.get(key, {}).get(comp)

            if obs is not None:
                t = np.arange(len(obs)) * sta.dt
                ax.plot(t, obs, "k-", linewidth=0.8, label="Obs")
                if syn is not None:
                    ax.plot(t, syn, "r-", linewidth=0.8, label="Syn")

            if i == 0:
                ax.set_title(comp, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(
                    f"{sta.network}.{sta.station}\n"
                    f"{sta.distance_km:.0f} km, {sta.azimuth:.0f}\u00b0",
                    fontsize=8,
                )
            if i == n_sta - 1:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticklabels([])

            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.tick_params(labelsize=7)

    axes[0, 2].legend(fontsize=8, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Waveform fit plot saved to {output_path}")


def plot_focal_mechanism(
    decomposition: MTDecomposition,
    output_path: str | Path,
    event_id: str = "",
) -> None:
    """Plot focal mechanism beachball with decomposition info."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Beachball
    _draw_beachball(axes[0], decomposition.mt_vector)
    axes[0].set_title("Focal Mechanism", fontsize=12, fontweight="bold")

    # Info panel
    ax = axes[1]
    ax.axis("off")
    info_lines = [
        f"Event: {event_id}" if event_id else "",
        f"Mw = {decomposition.moment_magnitude:.2f}",
        f"M0 = {decomposition.scalar_moment:.2e} N\u00b7m",
        "",
        "Decomposition:",
        f"  DC  = {decomposition.dc_percent:5.1f}%",
        f"  CLVD = {decomposition.clvd_percent:5.1f}%",
        f"  ISO  = {decomposition.iso_percent:5.1f}%",
        "",
        "Nodal Plane 1:",
        f"  Strike={decomposition.strike1:.0f}\u00b0  "
        f"Dip={decomposition.dip1:.0f}\u00b0  "
        f"Rake={decomposition.rake1:.0f}\u00b0",
        "Nodal Plane 2:",
        f"  Strike={decomposition.strike2:.0f}\u00b0  "
        f"Dip={decomposition.dip2:.0f}\u00b0  "
        f"Rake={decomposition.rake2:.0f}\u00b0",
        "",
        f"P-axis: trend={decomposition.p_axis[0]:.0f}\u00b0, "
        f"plunge={decomposition.p_axis[1]:.0f}\u00b0",
        f"T-axis: trend={decomposition.t_axis[0]:.0f}, "
        f"plunge={decomposition.t_axis[1]:.0f}\u00b0",
        "",
        "MT (Mxx Myy Mzz Mxy Mxz Myz):",
        f"  {decomposition.mt_vector}",
    ]
    text = "\n".join(line for line in info_lines if line is not None)
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes, verticalalignment="top",
        fontsize=9, fontfamily="monospace",
    )

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Focal mechanism plot saved to {output_path}")


def plot_depth_search(
    search_result: DepthSearchResult,
    output_path: str | Path,
    event_id: str = "",
) -> None:
    """Plot depth vs. variance reduction and depth vs. misfit."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    depths = search_result.depths
    vrs = search_result.variance_reductions
    misfits = search_result.misfits

    valid = vrs > -np.inf

    # VR curve
    ax1.plot(depths[valid], vrs[valid] * 100, "ko-", markersize=4)
    best_d = search_result.best_result.depth_km
    best_vr = search_result.best_result.variance_reduction * 100
    ax1.axvline(best_d, color="r", linestyle="--", alpha=0.7)
    ax1.plot(best_d, best_vr, "r*", markersize=12)
    ax1.set_xlabel("Depth (km)")
    ax1.set_ylabel("Variance Reduction (%)")
    ax1.set_title("Depth vs. Variance Reduction")
    ax1.grid(True, alpha=0.3)
    ax1.annotate(
        f"Best: {best_d:.1f} km\nVR={best_vr:.1f}%",
        xy=(best_d, best_vr), xytext=(10, -20),
        textcoords="offset points", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="r"),
    )

    # Misfit curve
    valid_m = misfits < np.inf
    ax2.plot(depths[valid_m], misfits[valid_m], "ko-", markersize=4)
    ax2.axvline(best_d, color="r", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Depth (km)")
    ax2.set_ylabel("Residual Norm")
    ax2.set_title("Depth vs. Misfit")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"{event_id} Depth Grid Search" if event_id else "Depth Grid Search",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Depth search plot saved to {output_path}")


def plot_station_map(
    stations: list[StationData],
    event_latitude: float,
    event_longitude: float,
    output_path: str | Path,
    event_id: str = "",
) -> None:
    """Plot station map with event location."""
    fig, ax = plt.subplots(figsize=(8, 8))

    accepted = [s for s in stations if not s.rejected]
    rejected = [s for s in stations if s.rejected and s.latitude != 0]

    if rejected:
        ax.scatter(
            [s.longitude for s in rejected],
            [s.latitude for s in rejected],
            marker="^", c="gray", s=60, alpha=0.5, label="Rejected",
            edgecolors="k", linewidths=0.5,
        )

    if accepted:
        ax.scatter(
            [s.longitude for s in accepted],
            [s.latitude for s in accepted],
            marker="^", c="blue", s=80, label="Used",
            edgecolors="k", linewidths=0.5,
        )
        for s in accepted:
            ax.annotate(
                f"{s.network}.{s.station}",
                (s.longitude, s.latitude),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7,
            )

    ax.scatter(
        event_longitude, event_latitude,
        marker="*", c="red", s=200, zorder=10, label="Event",
        edgecolors="k", linewidths=0.5,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Station Map{' - ' + event_id if event_id else ''}")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Station map saved to {output_path}")


def plot_summary(
    search_result: DepthSearchResult,
    stations: list[StationData],
    event_latitude: float,
    event_longitude: float,
    output_path: str | Path,
    event_id: str = "",
) -> None:
    """Combined summary figure with beachball, depth curve, and waveform fits."""
    best = search_result.best_result
    dec = best.decomposition
    accepted = [s for s in stations if not s.rejected]
    n_sta = min(len(accepted), 6)  # limit for readability

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Beachball (top-left)
    ax_bb = fig.add_subplot(gs[0, 0])
    _draw_beachball(ax_bb, dec.mt_vector)
    ax_bb.set_title(
        f"Mw {dec.moment_magnitude:.2f}\n"
        f"DC {dec.dc_percent:.0f}%",
        fontsize=10,
    )

    # Info text (top, second column)
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")
    info = (
        f"Depth: {best.depth_km:.1f} km\n"
        f"VR: {best.variance_reduction:.1%}\n"
        f"Cond #: {best.condition_number:.1e}\n"
        f"Stations: {best.n_stations}\n\n"
        f"NP1: {dec.strike1:.0f}/{dec.dip1:.0f}/{dec.rake1:.0f}\n"
        f"NP2: {dec.strike2:.0f}/{dec.dip2:.0f}/{dec.rake2:.0f}"
    )
    ax_info.text(0.1, 0.9, info, transform=ax_info.transAxes,
                 verticalalignment="top", fontsize=9, fontfamily="monospace")

    # Depth curve (top-right half)
    ax_depth = fig.add_subplot(gs[0, 2:])
    valid = search_result.variance_reductions > -np.inf
    ax_depth.plot(
        search_result.depths[valid],
        search_result.variance_reductions[valid] * 100,
        "ko-", markersize=3,
    )
    ax_depth.axvline(best.depth_km, color="r", ls="--", alpha=0.7)
    ax_depth.plot(best.depth_km, best.variance_reduction * 100, "r*", ms=10)
    ax_depth.set_xlabel("Depth (km)")
    ax_depth.set_ylabel("VR (%)")
    ax_depth.set_title("Depth Search")
    ax_depth.grid(True, alpha=0.3)

    # Waveform fits (bottom rows)
    for i in range(min(n_sta, 6)):
        sta = accepted[i]
        key = f"{sta.network}.{sta.station}"
        for j, comp in enumerate(["Z", "R", "T"]):
            row = 1 + i // 3
            col = (i % 3) if row == 1 else (i % 3)
            if i < 3:
                ax = fig.add_subplot(gs[1, j + (i > 0 and j == 0)])
            # simplified: just do first 3 stations across bottom
            if i >= 3:
                break
            ax = fig.add_subplot(gs[1 + i // 3, j])
            obs = best.observed.get(key, {}).get(comp)
            syn = best.synthetic.get(key, {}).get(comp)
            if obs is not None:
                t = np.arange(len(obs)) * sta.dt
                ax.plot(t, obs, "k-", lw=0.7)
                if syn is not None:
                    ax.plot(t, syn, "r-", lw=0.7)
            ax.set_title(f"{key} {comp}", fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle(
        f"{event_id} Moment Tensor Solution" if event_id else "Moment Tensor Solution",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Summary figure saved to {output_path}")
