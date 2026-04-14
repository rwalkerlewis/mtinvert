"""
Linear least-squares moment tensor inversion with depth grid search.

Solves for the six independent moment tensor elements from
three-component broadband displacement waveforms using
pre-computed Green's functions. Supports both full and
deviatoric (trace-free) inversion.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq

from .config import Config
from .decomposition import MTDecomposition, decompose
from .greens import MT_ELEMENTS, load_fundamental_gf, assemble_mt_gf, get_gf_metadata
from .preprocess import StationData

logger = logging.getLogger(__name__)


@dataclass
class InversionResult:
    """Result from a single-depth inversion."""
    depth_km: float
    mt_vector: np.ndarray  # (6,) Mxx, Myy, Mzz, Mxy, Mxz, Myz
    variance_reduction: float
    condition_number: float
    residual_norm: float
    n_stations: int
    n_components: int
    decomposition: MTDecomposition | None = None
    observed: dict = field(default_factory=dict)
    synthetic: dict = field(default_factory=dict)


@dataclass
class DepthSearchResult:
    """Result from depth grid search."""
    best_result: InversionResult
    all_results: list[InversionResult]
    depths: np.ndarray
    misfits: np.ndarray
    variance_reductions: np.ndarray


def _align_and_trim(
    observed: np.ndarray,
    green: np.ndarray,
    max_shift_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure observed and Green's function arrays have same length."""
    n = min(observed.shape[-1], green.shape[-1])
    return observed[..., :n], green[..., :n]


def _build_kernel_matrix(
    stations: list[StationData],
    gf_library_path: str | Path,
    depth_km: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str, str]]]:
    """
    Build the G matrix and d vector for the linear system Gm = d.

    G has shape (n_data, 6) where n_data = n_stations * 3 * npts
    and 6 is the number of MT elements.
    d is the data vector of shape (n_data,).

    Returns G, d, and list of (net, sta, comp) labels.
    """
    G_blocks = []
    d_blocks = []
    labels = []

    for sta_data in stations:
        if sta_data.rejected:
            continue

        fund_gf = load_fundamental_gf(gf_library_path, depth_km, sta_data.distance_km)
        if fund_gf is None:
            logger.warning(
                f"No GF for {sta_data.network}.{sta_data.station} "
                f"at depth={depth_km}, dist={sta_data.distance_km:.1f}"
            )
            continue

        # Assemble per-MT-element GFs using station azimuth
        gf = assemble_mt_gf(fund_gf, sta_data.azimuth)

        for ic, comp in enumerate(["Z", "R", "T"]):
            obs = sta_data.data_zrt[ic]

            # Build row: each MT element contributes its GF for this component
            g_row_blocks = []
            for ie, elem in enumerate(MT_ELEMENTS):
                gf_trace = gf[elem][ic]  # component ic for this MT element
                obs_trimmed, gf_trimmed = _align_and_trim(obs, gf_trace)
                g_row_blocks.append(gf_trimmed)

            npts = len(obs_trimmed)
            g_block = np.column_stack(g_row_blocks)  # (npts, 6)
            G_blocks.append(g_block)
            d_blocks.append(obs_trimmed)
            labels.append((sta_data.network, sta_data.station, comp))

    if not G_blocks:
        return np.array([]), np.array([]), []

    G = np.vstack(G_blocks)
    d = np.concatenate(d_blocks)
    return G, d, labels


def invert_mt(
    stations: list[StationData],
    config: Config,
    depth_km: float,
) -> InversionResult | None:
    """
    Perform moment tensor inversion at a single depth.

    Parameters
    ----------
    stations : list of StationData
        Preprocessed station data.
    config : Config
        Configuration.
    depth_km : float
        Source depth in km.

    Returns
    -------
    InversionResult or None
    """
    accepted = [s for s in stations if not s.rejected]
    if len(accepted) < 2:
        logger.error("Need at least 2 stations for inversion")
        return None

    G, d, labels = _build_kernel_matrix(accepted, config.gf_library_path, depth_km)
    if G.size == 0:
        logger.error("Empty kernel matrix")
        return None

    n_data, n_params = G.shape

    if config.inversion_type == "deviatoric":
        # Enforce trace-free constraint: Mxx + Myy + Mzz = 0
        # Add constraint row with large weight
        constraint_row = np.zeros((1, n_params))
        constraint_row[0, 0] = 1.0  # Mxx
        constraint_row[0, 1] = 1.0  # Myy
        constraint_row[0, 2] = 1.0  # Mzz
        weight = np.max(np.abs(G)) * 1000.0
        G_aug = np.vstack([G, constraint_row * weight])
        d_aug = np.append(d, 0.0)
    else:
        G_aug = G
        d_aug = d

    # Solve via least squares
    m, residuals, rank, sv = lstsq(G_aug, d_aug)

    # Condition number
    if sv[-1] > 0:
        cond = sv[0] / sv[-1]
    else:
        cond = np.inf

    # Compute synthetics and variance reduction
    d_pred = G @ m
    residual = d - d_pred
    residual_norm = float(np.sqrt(np.sum(residual**2)))

    var_data = np.sum(d**2)
    var_residual = np.sum(residual**2)
    vr = 1.0 - var_residual / (var_data + 1e-30)

    # Organize observed/synthetic per station
    obs_dict = {}
    syn_dict = {}
    idx = 0
    meta = get_gf_metadata(config.gf_library_path)
    for net, sta, comp in labels:
        key = f"{net}.{sta}"
        npts_gf = meta["npts"]
        # Find actual length used
        sta_obj = next(
            (s for s in accepted if s.network == net and s.station == sta), None
        )
        if sta_obj is not None:
            npts_used = min(len(sta_obj.data_zrt[0]), npts_gf)
        else:
            npts_used = npts_gf

        if key not in obs_dict:
            obs_dict[key] = {}
            syn_dict[key] = {}
        obs_dict[key][comp] = d[idx : idx + npts_used]
        syn_dict[key][comp] = d_pred[idx : idx + npts_used]
        idx += npts_used

    # Decompose
    dec = decompose(m)

    result = InversionResult(
        depth_km=depth_km,
        mt_vector=m,
        variance_reduction=float(vr),
        condition_number=float(cond),
        residual_norm=residual_norm,
        n_stations=len(accepted),
        n_components=len(labels),
        decomposition=dec,
        observed=obs_dict,
        synthetic=syn_dict,
    )

    logger.info(
        f"Depth {depth_km:.1f} km: VR={vr:.3f}, Mw={dec.moment_magnitude:.2f}, "
        f"DC={dec.dc_percent:.1f}%, cond={cond:.1e}"
    )
    return result


def depth_grid_search(
    stations: list[StationData],
    config: Config,
) -> DepthSearchResult | None:
    """
    Perform moment tensor inversion at each depth in the grid
    and return the best-fitting solution.

    Parameters
    ----------
    stations : list of StationData
        Preprocessed station data.
    config : Config
        Configuration with depth search range.

    Returns
    -------
    DepthSearchResult or None
    """
    depths = np.arange(
        config.depth_min_km,
        config.depth_max_km + config.depth_step_km / 2,
        config.depth_step_km,
    )

    all_results = []
    misfits = []
    vrs = []

    logger.info(f"Depth grid search: {len(depths)} trial depths")

    for depth in depths:
        result = invert_mt(stations, config, depth)
        if result is not None:
            all_results.append(result)
            misfits.append(result.residual_norm)
            vrs.append(result.variance_reduction)
        else:
            misfits.append(np.inf)
            vrs.append(-np.inf)

    if not all_results:
        logger.error("All depth inversions failed")
        return None

    misfits = np.array(misfits)
    vrs = np.array(vrs)

    best_idx = np.argmax(vrs[vrs > -np.inf])
    best = all_results[best_idx]

    logger.info(
        f"Best depth: {best.depth_km:.1f} km, VR={best.variance_reduction:.3f}, "
        f"Mw={best.decomposition.moment_magnitude:.2f}"
    )

    return DepthSearchResult(
        best_result=best,
        all_results=all_results,
        depths=depths,
        misfits=misfits,
        variance_reductions=vrs,
    )


def write_solution_csv(
    result: DepthSearchResult,
    output_path: str | Path,
) -> None:
    """Write depth search results to CSV."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "depth_km", "variance_reduction", "condition_number",
            "residual_norm", "Mw", "DC_percent", "CLVD_percent",
            "ISO_percent", "strike1", "dip1", "rake1",
            "strike2", "dip2", "rake2",
        ])
        for r in result.all_results:
            d = r.decomposition
            writer.writerow([
                f"{r.depth_km:.1f}",
                f"{r.variance_reduction:.4f}",
                f"{r.condition_number:.2e}",
                f"{r.residual_norm:.4e}",
                f"{d.moment_magnitude:.2f}",
                f"{d.dc_percent:.1f}",
                f"{d.clvd_percent:.1f}",
                f"{d.iso_percent:.1f}",
                f"{d.strike1:.1f}", f"{d.dip1:.1f}", f"{d.rake1:.1f}",
                f"{d.strike2:.1f}", f"{d.dip2:.1f}", f"{d.rake2:.1f}",
            ])
