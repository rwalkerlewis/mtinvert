"""
Waveform preprocessing for moment tensor inversion.

Handles instrument response removal, rotation to ZRT, bandpass
filtering, windowing, tapering, and SNR-based quality control.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from obspy import Stream, read, read_inventory
from obspy.geodetics import gps2dist_azimuth

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class StationData:
    """Preprocessed waveform data for a single station."""
    network: str
    station: str
    latitude: float
    longitude: float
    distance_km: float
    azimuth: float
    back_azimuth: float
    data_zrt: np.ndarray  # (3, npts) -- Z, R, T
    dt: float
    starttime: float
    snr: dict[str, float]  # per-component SNR
    rejected: bool = False
    reject_reason: str = ""


def _compute_snr(
    trace_data: np.ndarray,
    dt: float,
    pre_event_sec: float,
) -> float:
    """Compute SNR as ratio of RMS signal to RMS noise."""
    n_noise = max(1, int(pre_event_sec / dt))
    if len(trace_data) <= n_noise:
        return 0.0
    noise = trace_data[:n_noise]
    signal = trace_data[n_noise:]
    rms_noise = np.sqrt(np.mean(noise**2)) + 1e-30
    rms_signal = np.sqrt(np.mean(signal**2))
    return float(rms_signal / rms_noise)


def preprocess_event(
    config: Config,
    event_latitude: float,
    event_longitude: float,
    event_origin_time,
    mseed_files: list[str | Path] | None = None,
) -> list[StationData]:
    """
    Read waveforms, remove instrument response, rotate to ZRT,
    filter, window, and perform QC.

    Parameters
    ----------
    config : Config
        Processing configuration.
    event_latitude, event_longitude : float
        Event coordinates.
    event_origin_time : UTCDateTime
        Event origin time.
    mseed_files : list of paths, optional
        Specific miniSEED files. If None, reads all *.mseed / *.miniseed
        from config.data_dir.

    Returns
    -------
    list of StationData
        Preprocessed station data, including rejected stations.
    """
    from obspy import UTCDateTime

    if not isinstance(event_origin_time, UTCDateTime):
        event_origin_time = UTCDateTime(event_origin_time)

    # Read data
    if mseed_files is None:
        data_dir = Path(config.data_dir)
        mseed_files = list(data_dir.glob("*.mseed")) + list(data_dir.glob("*.miniseed"))

    if not mseed_files:
        logger.error(f"No miniSEED files found in {config.data_dir}")
        return []

    st = Stream()
    for f in mseed_files:
        try:
            st += read(str(f))
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")

    # Read inventory
    inv = read_inventory(config.stationxml_path)

    # Merge and sort
    st.merge(fill_value=0)
    st.sort()

    # Get unique station codes
    stations = set()
    for tr in st:
        stations.add((tr.stats.network, tr.stats.station))

    results = []

    for net, sta in sorted(stations):
        st_sta = st.select(network=net, station=sta)
        if len(st_sta) < 3:
            logger.info(f"Skipping {net}.{sta}: fewer than 3 components")
            sd = StationData(
                network=net, station=sta,
                latitude=0, longitude=0,
                distance_km=0, azimuth=0, back_azimuth=0,
                data_zrt=np.array([]), dt=0, starttime=0,
                snr={}, rejected=True,
                reject_reason="fewer than 3 components",
            )
            results.append(sd)
            continue

        # Get coordinates from inventory
        try:
            coords = inv.get_coordinates(st_sta[0].get_id())
            sta_lat = coords["latitude"]
            sta_lon = coords["longitude"]
        except Exception:
            logger.warning(f"No coordinates for {net}.{sta}, skipping")
            continue

        dist_m, az, baz = gps2dist_azimuth(
            event_latitude, event_longitude, sta_lat, sta_lon
        )
        dist_km = dist_m / 1000.0

        # Distance filter
        if dist_km < config.distance_min_km or dist_km > config.distance_max_km:
            sd = StationData(
                network=net, station=sta,
                latitude=sta_lat, longitude=sta_lon,
                distance_km=dist_km, azimuth=az, back_azimuth=baz,
                data_zrt=np.array([]), dt=0, starttime=0,
                snr={}, rejected=True,
                reject_reason=f"distance {dist_km:.1f} km outside range",
            )
            results.append(sd)
            continue

        # Process
        try:
            st_proc = st_sta.copy()

            # Trim to time window
            t1 = event_origin_time - config.pre_event_sec
            t2 = event_origin_time + config.post_event_sec
            st_proc.trim(t1, t2, pad=True, fill_value=0)

            # Detrend and taper
            st_proc.detrend("demean")
            st_proc.detrend("linear")
            st_proc.taper(max_percentage=config.taper_fraction)

            # Remove instrument response to displacement
            st_proc.remove_response(
                inventory=inv,
                output="DISP",
                water_level=config.water_level_db,
                pre_filt=[
                    config.freqmin_hz * 0.5,
                    config.freqmin_hz,
                    config.freqmax_hz,
                    config.freqmax_hz * 1.5,
                ],
            )

            # Bandpass filter
            st_proc.filter(
                "bandpass",
                freqmin=config.freqmin_hz,
                freqmax=config.freqmax_hz,
                corners=4,
                zerophase=True,
            )

            # Rotate NE -> RT
            st_proc.rotate("NE->RT", back_azimuth=baz)

            dt = st_proc[0].stats.delta

            # Extract ZRT arrays
            z_tr = st_proc.select(component="Z")
            r_tr = st_proc.select(component="R")
            t_tr = st_proc.select(component="T")

            if not (z_tr and r_tr and t_tr):
                raise ValueError("Missing component after rotation")

            npts = min(len(z_tr[0].data), len(r_tr[0].data), len(t_tr[0].data))
            data_zrt = np.zeros((3, npts))
            data_zrt[0] = z_tr[0].data[:npts]
            data_zrt[1] = r_tr[0].data[:npts]
            data_zrt[2] = t_tr[0].data[:npts]

            # Compute SNR
            snr = {
                "Z": _compute_snr(data_zrt[0], dt, config.pre_event_sec),
                "R": _compute_snr(data_zrt[1], dt, config.pre_event_sec),
                "T": _compute_snr(data_zrt[2], dt, config.pre_event_sec),
            }

            rejected = False
            reason = ""
            # Reject if any component below threshold
            for comp, val in snr.items():
                if val < config.snr_threshold:
                    rejected = True
                    reason = f"SNR({comp})={val:.1f} < {config.snr_threshold}"
                    break

            sd = StationData(
                network=net, station=sta,
                latitude=sta_lat, longitude=sta_lon,
                distance_km=dist_km, azimuth=az, back_azimuth=baz,
                data_zrt=data_zrt, dt=dt,
                starttime=float(st_proc[0].stats.starttime),
                snr=snr, rejected=rejected, reject_reason=reason,
            )
            results.append(sd)

        except Exception as e:
            logger.warning(f"Processing failed for {net}.{sta}: {e}")
            sd = StationData(
                network=net, station=sta,
                latitude=sta_lat, longitude=sta_lon,
                distance_km=dist_km, azimuth=az, back_azimuth=baz,
                data_zrt=np.array([]), dt=0, starttime=0,
                snr={}, rejected=True, reject_reason=str(e),
            )
            results.append(sd)

    n_good = sum(1 for s in results if not s.rejected)
    n_bad = sum(1 for s in results if s.rejected)
    logger.info(f"Preprocessing complete: {n_good} accepted, {n_bad} rejected")
    return results
