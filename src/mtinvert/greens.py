"""
Green's function library builder and accessor.

Computes synthetic Green's functions for a 1D layered velocity model
using frequency-wavenumber (FK) integration via Computer Programs in
Seismology (CPS), and stores them in an HDF5 database indexed by
(depth, distance).

The library stores responses for four fundamental source types
(DD, DS, SS, EP) on ZRT components, following the CPS/Herrmann
convention. Azimuth-dependent radiation pattern coefficients are
applied during kernel matrix assembly via assemble_mt_gf().

Fundamental source types:
  DD -- Mzz source (vertical force dipole): Z, R (T=0 by symmetry)
  DS -- dip-slip on vertical fault: Z, R, T
  SS -- strike-slip on vertical fault: Z, R, T
  EP -- horizontal explosion (Mxx+Myy)/2: Z, R (T=0 by symmetry)

Total: 10 nonzero traces per (depth, distance) node.
"""

import logging
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .config import Config, VelocityLayer

logger = logging.getLogger(__name__)

# The six independent moment tensor elements (Jost & Herrmann 1989)
MT_ELEMENTS = ["Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"]

# Receiver components
COMPONENTS = ["Z", "R", "T"]

# Fundamental source types stored in the GF library
FUNDAMENTAL_SOURCES = ["DD", "DS", "SS", "EP"]

# Which components are nonzero for each fundamental source
FUND_COMPONENTS = {
    "DD": ["Z", "R"],
    "DS": ["Z", "R", "T"],
    "SS": ["Z", "R", "T"],
    "EP": ["Z", "R"],
}


def get_azimuth_coefficients(
    azimuth_deg: float,
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """
    Return azimuth-dependent coefficients mapping each MT element to
    weighted combinations of fundamental GFs on each component.

    Following Herrmann (2002) / CPS convention:

    Z and R components:
      Mxx -> 0.5*EP + 0.5*cos(2*phi)*SS
      Myy -> 0.5*EP - 0.5*cos(2*phi)*SS
      Mzz -> DD
      Mxy -> sin(2*phi)*SS
      Mxz -> cos(phi)*DS
      Myz -> sin(phi)*DS

    T component:
      Mxx -> 0.5*sin(2*phi)*SS
      Myy -> -0.5*sin(2*phi)*SS
      Mzz -> 0
      Mxy -> -cos(2*phi)*SS
      Mxz -> sin(phi)*DS
      Myz -> -cos(phi)*DS

    Returns
    -------
    dict : MT_element -> component -> list of (fund_source, coefficient)
    """
    phi = np.radians(azimuth_deg)
    c2p = np.cos(2 * phi)
    s2p = np.sin(2 * phi)
    cp = np.cos(phi)
    sp = np.sin(phi)

    return {
        "Mxx": {
            "Z": [("EP", 0.5), ("SS", 0.5 * c2p)],
            "R": [("EP", 0.5), ("SS", 0.5 * c2p)],
            "T": [("SS", 0.5 * s2p)],
        },
        "Myy": {
            "Z": [("EP", 0.5), ("SS", -0.5 * c2p)],
            "R": [("EP", 0.5), ("SS", -0.5 * c2p)],
            "T": [("SS", -0.5 * s2p)],
        },
        "Mzz": {
            "Z": [("DD", 1.0)],
            "R": [("DD", 1.0)],
            "T": [],
        },
        "Mxy": {
            "Z": [("SS", s2p)],
            "R": [("SS", s2p)],
            "T": [("SS", -c2p)],
        },
        "Mxz": {
            "Z": [("DS", cp)],
            "R": [("DS", cp)],
            "T": [("DS", sp)],
        },
        "Myz": {
            "Z": [("DS", sp)],
            "R": [("DS", sp)],
            "T": [("DS", -cp)],
        },
    }


def _write_velocity_model(layers: list[VelocityLayer], path: Path) -> None:
    """Write CPS-format velocity model file."""
    with open(path, "w") as f:
        f.write("MODEL\n")
        f.write("Model for mtinvert GF computation\n")
        f.write("ISOTROPIC\n")
        f.write("KGS\n")
        f.write("FLAT EARTH\n")
        f.write("1-D\n")
        f.write("CONSTANT VELOCITY\n")
        f.write("LINE08\n")
        f.write("LINE09\n")
        f.write("LINE10\n")
        f.write("LINE11\n")
        f.write(
            "H(KM)  VP(KM/S)  VS(KM/S)  RHO(GM/CC)  QP  QS  ETAP  ETAS  FREFP  FREFS\n"
        )
        for lyr in layers:
            f.write(
                f"  {lyr.thickness_km:.4f}  {lyr.vp_km_s:.4f}  "
                f"{lyr.vs_km_s:.4f}  {lyr.rho_g_cc:.4f}  "
                f"{lyr.qp:.1f}  {lyr.qs:.1f}  0.0  0.0  1.0  1.0\n"
            )


def _compute_gf_single(
    model_path: str,
    depth_km: float,
    distance_km: float,
    dt: float = 0.1,
    npts: int = 2048,
) -> Optional[dict[str, np.ndarray]]:
    """
    Compute fundamental Green's functions for a single (depth, distance)
    pair using CPS hprep96 / hspec96 / hpulse96.

    Returns dict mapping "SOURCE_COMP" keys to 1D arrays, or None.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            cmd_prep = [
                "hprep96", "-M", str(model_path),
                "-d", str(dt), "-HS", str(depth_km),
                "-HR", "0.0", "-EQEX", "-R", str(distance_km),
            ]
            subprocess.run(cmd_prep, cwd=tmp, capture_output=True, check=True, timeout=120)
            subprocess.run(["hspec96"], cwd=tmp, capture_output=True, check=True, timeout=300)
            subprocess.run(
                ["hpulse96", "-D", "-i", "-l", str(npts)],
                cwd=tmp, capture_output=True, check=True, timeout=120,
            )
            subprocess.run(["f96tosac", "-G"], cwd=tmp, capture_output=True, check=True, timeout=60)

            fund_map = {0: "DD", 1: "DS", 2: "SS", 3: "EP"}
            comp_map = {0: "Z", 1: "R", 2: "T"}
            result = {}
            for fi, fname in fund_map.items():
                for ci, cname in comp_map.items():
                    key = f"{fname}_{cname}"
                    if cname == "T" and fname in ("DD", "EP"):
                        result[key] = np.zeros(npts)
                        continue
                    sac_file = tmp / f"B{fi+1:03d}{ci+1:02d}ZSS.sac"
                    if sac_file.exists():
                        from obspy import read
                        st = read(str(sac_file))
                        data = np.zeros(npts)
                        data[: min(npts, len(st[0].data))] = st[0].data[:npts]
                        result[key] = data
                    else:
                        result[key] = np.zeros(npts)
            return result

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"GF computation failed for depth={depth_km}, distance={distance_km}: {e}")
        return None


def _compute_gf_synthetic(
    layers: list[VelocityLayer],
    depth_km: float,
    distance_km: float,
    dt: float = 0.1,
    npts: int = 2048,
) -> dict[str, np.ndarray]:
    """
    Compute approximate fundamental Green's functions analytically
    for testing when CPS is not available.

    Uses far-field P and S radiation patterns in a homogeneous
    whole-space with distinct takeoff-angle dependence for each
    fundamental source type:

      DD (Mzz):  P_rad ~ cos^2(ih),  SV ~ -sin(2ih)/2
      EP (horiz. explosion):  P_rad ~ sin^2(ih),  SV ~ sin(2ih)/2
      DS (dip-slip):  P_rad ~ sin(2ih)/2,  SV ~ cos(2ih),  SH ~ sin(ih)
      SS (strike-slip):  P_rad ~ sin^2(ih),  SV ~ sin(2ih)/2,  SH ~ sin(ih)

    ih = takeoff angle from vertical = arctan(distance/depth).
    """
    if not layers:
        vp, vs, rho = 6.0, 3.5, 2.7
    else:
        vp, vs, rho = layers[0].vp_km_s, layers[0].vs_km_s, layers[0].rho_g_cc

    r_km = np.sqrt(distance_km**2 + depth_km**2)
    r_m = r_km * 1000.0
    ih = np.arctan2(distance_km, depth_km)

    tp = r_km / vp
    ts = r_km / vs
    t = np.arange(npts) * dt

    pw = 3.0 * dt
    p_base = np.exp(-((t - tp) ** 2) / (2 * pw**2))
    s_base = np.exp(-((t - ts) ** 2) / (2 * pw**2))

    p_norm = 1.0 / (4.0 * np.pi * rho * vp**3 * r_m + 1e-30)
    s_norm = 1.0 / (4.0 * np.pi * rho * vs**3 * r_m + 1e-30)

    si = np.sin(ih)
    ci = np.cos(ih)

    result = {}

    # DD: Mzz source
    dd_p = ci**2
    dd_sv = -0.5 * np.sin(2 * ih)
    result["DD_Z"] = p_base * p_norm * dd_p * ci + s_base * s_norm * dd_sv * (-si)
    result["DD_R"] = p_base * p_norm * dd_p * si + s_base * s_norm * dd_sv * ci
    result["DD_T"] = np.zeros(npts)

    # EP: (Mxx+Myy)/2 horizontal explosion
    ep_p = si**2
    ep_sv = 0.5 * np.sin(2 * ih)
    result["EP_Z"] = p_base * p_norm * ep_p * ci + s_base * s_norm * ep_sv * (-si)
    result["EP_R"] = p_base * p_norm * ep_p * si + s_base * s_norm * ep_sv * ci
    result["EP_T"] = np.zeros(npts)

    # DS: dip-slip
    ds_p = 0.5 * np.sin(2 * ih)
    ds_sv = np.cos(2 * ih)
    ds_sh = si
    result["DS_Z"] = p_base * p_norm * ds_p * ci + s_base * s_norm * ds_sv * (-si)
    result["DS_R"] = p_base * p_norm * ds_p * si + s_base * s_norm * ds_sv * ci
    result["DS_T"] = s_base * s_norm * ds_sh

    # SS: strike-slip
    ss_p = si**2
    ss_sv = 0.5 * np.sin(2 * ih)
    ss_sh = si
    result["SS_Z"] = p_base * p_norm * ss_p * ci + s_base * s_norm * ss_sv * (-si)
    result["SS_R"] = p_base * p_norm * ss_p * si + s_base * s_norm * ss_sv * ci
    result["SS_T"] = s_base * s_norm * ss_sh

    return result


def build_gf_library(
    config: Config,
    use_cps: bool = True,
    dt: float = 0.1,
    npts: int = 2048,
) -> None:
    """
    Build the full Green's function HDF5 library.

    Stores fundamental source type GFs (DD, DS, SS, EP) on Z, R, T
    components for each (depth, distance) node.
    """
    depths = np.arange(
        config.depth_min_km, config.depth_max_km + config.depth_step_km / 2,
        config.depth_step_km,
    )
    distances = np.arange(
        config.distance_min_km,
        config.distance_max_km + config.distance_step_km / 2,
        config.distance_step_km,
    )

    out_path = Path(config.gf_library_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = None
    if use_cps:
        model_path = out_path.parent / "velocity_model.mod"
        _write_velocity_model(config.velocity_model, model_path)

    n_total = len(depths) * len(distances)
    logger.info(f"Building GF library: {len(depths)} x {len(distances)} = {n_total} entries")

    gf_keys = []
    for src in FUNDAMENTAL_SOURCES:
        for comp in COMPONENTS:
            gf_keys.append(f"{src}_{comp}")

    with h5py.File(str(out_path), "w") as hf:
        hf.attrs["dt"] = dt
        hf.attrs["npts"] = npts
        hf.attrs["fundamental_sources"] = FUNDAMENTAL_SOURCES
        hf.attrs["components"] = COMPONENTS
        hf.attrs["gf_keys"] = gf_keys
        hf.attrs["depths_km"] = depths
        hf.attrs["distances_km"] = distances

        vm_grp = hf.create_group("velocity_model")
        for i, lyr in enumerate(config.velocity_model):
            lg = vm_grp.create_group(f"layer_{i:03d}")
            lg.attrs["thickness_km"] = lyr.thickness_km
            lg.attrs["vp_km_s"] = lyr.vp_km_s
            lg.attrs["vs_km_s"] = lyr.vs_km_s
            lg.attrs["rho_g_cc"] = lyr.rho_g_cc
            lg.attrs["qp"] = lyr.qp
            lg.attrs["qs"] = lyr.qs

        tasks = [(d, r) for d in depths for r in distances]

        def _compute(depth_km, distance_km):
            if use_cps and model_path is not None:
                gf = _compute_gf_single(str(model_path), depth_km, distance_km, dt, npts)
                if gf is not None:
                    return depth_km, distance_km, gf
            return depth_km, distance_km, _compute_gf_synthetic(
                config.velocity_model, depth_km, distance_km, dt, npts
            )

        completed = 0
        for depth_km, distance_km in tasks:
            _, _, gf_data = _compute(depth_km, distance_km)
            key = f"d{depth_km:.1f}_r{distance_km:.1f}"
            grp = hf.create_group(key)
            grp.attrs["depth_km"] = depth_km
            grp.attrs["distance_km"] = distance_km
            for gk in gf_keys:
                grp.create_dataset(gk, data=gf_data.get(gk, np.zeros(npts)), compression="gzip")
            completed += 1
            if completed % 100 == 0:
                logger.info(f"  {completed}/{n_total} entries computed")

    logger.info(f"GF library written to {out_path} ({n_total} entries)")


def load_fundamental_gf(
    library_path: str | Path,
    depth_km: float,
    distance_km: float,
) -> Optional[dict[str, np.ndarray]]:
    """
    Load fundamental GFs for the nearest (depth, distance) node.

    Returns dict mapping "SOURCE_COMP" keys to 1D arrays, or None.
    """
    with h5py.File(str(library_path), "r") as hf:
        depths = hf.attrs["depths_km"]
        distances = hf.attrs["distances_km"]
        d_idx = np.argmin(np.abs(depths - depth_km))
        r_idx = np.argmin(np.abs(distances - distance_km))
        snap_d = depths[d_idx]
        snap_r = distances[r_idx]

        key = f"d{snap_d:.1f}_r{snap_r:.1f}"
        if key not in hf:
            logger.warning(f"GF entry {key} not found in library")
            return None

        grp = hf[key]
        gf_keys = list(hf.attrs["gf_keys"])
        return {gk: grp[gk][:] for gk in gf_keys}


def assemble_mt_gf(
    fund_gf: dict[str, np.ndarray],
    azimuth_deg: float,
) -> dict[str, np.ndarray]:
    """
    Assemble per-MT-element Green's functions from fundamental GFs
    and station azimuth.

    Applies azimuth-dependent radiation pattern coefficients to
    combine fundamental GFs into the response for each of the six
    MT elements on each component.

    Parameters
    ----------
    fund_gf : dict
        Fundamental GFs from load_fundamental_gf().
    azimuth_deg : float
        Station azimuth from source (degrees from north).

    Returns
    -------
    dict mapping MT element name to (3, npts) array [Z, R, T].
    """
    coeffs = get_azimuth_coefficients(azimuth_deg)

    npts = None
    for v in fund_gf.values():
        npts = len(v)
        break
    if npts is None:
        return {}

    result = {}
    for elem in MT_ELEMENTS:
        traces = np.zeros((3, npts))
        for ic, comp in enumerate(COMPONENTS):
            for fund_src, coeff in coeffs[elem][comp]:
                gf_key = f"{fund_src}_{comp}"
                if gf_key in fund_gf:
                    traces[ic] += coeff * fund_gf[gf_key]
        result[elem] = traces

    return result


def get_gf_metadata(library_path: str | Path) -> dict:
    """Return metadata from GF library."""
    with h5py.File(str(library_path), "r") as hf:
        return {
            "dt": float(hf.attrs["dt"]),
            "npts": int(hf.attrs["npts"]),
            "depths_km": hf.attrs["depths_km"][:],
            "distances_km": hf.attrs["distances_km"][:],
        }
