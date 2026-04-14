#!/usr/bin/env python3
"""
Download example earthquake data from IRIS FDSN web services.

Uses the 2019-07-06 M7.1 Ridgecrest, CA aftershock sequence event,
well-recorded on the CI (Southern California Seismic Network) broadband
stations. We pick a moderate M4.9 foreshock for faster processing.

Event: 2019-07-04 17:33:49 UTC, M6.4 Ridgecrest foreshock
       Lat 35.705, Lon -117.506, Depth 10.5 km
       (USGS event ci38443183)

We use a smaller set of nearby CI broadband stations for a quick demo.
"""

import sys
from pathlib import Path

from obspy import UTCDateTime
from obspy.clients.fdsn import Client


def main():
    # Event parameters -- 2019-07-04 M6.4 Ridgecrest foreshock
    event_lat = 35.705
    event_lon = -117.506
    event_depth_km = 10.5
    origin_time = UTCDateTime("2019-07-04T17:33:49")

    # Output directories
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    stations_dir = base_dir / "stations"
    data_dir.mkdir(exist_ok=True)
    stations_dir.mkdir(exist_ok=True)

    print("Connecting to IRIS FDSN web service...")
    client = Client("IRIS")

    # Select CI broadband stations within 50-200 km
    # Use BH? channels (broadband high-gain)
    network = "CI"
    channel = "BH?"
    min_radius_deg = 0.5   # ~55 km
    max_radius_deg = 1.8   # ~200 km

    print(f"Fetching station inventory for {network} near event...")
    inv = client.get_stations(
        network=network,
        channel=channel,
        latitude=event_lat,
        longitude=event_lon,
        minradius=min_radius_deg,
        maxradius=max_radius_deg,
        starttime=origin_time - 60,
        endtime=origin_time + 300,
        level="response",
    )

    # Limit to first 8 stations for a quick demo
    station_codes = set()
    for net in inv:
        for sta in net:
            station_codes.add(sta.code)
            if len(station_codes) >= 8:
                break
        if len(station_codes) >= 8:
            break

    station_list = sorted(station_codes)
    print(f"Selected {len(station_list)} stations: {', '.join(station_list)}")

    # Filter inventory to selected stations (select one at a time and combine)
    from obspy import Inventory
    inv_selected = Inventory()
    for sta_code in station_list:
        inv_selected += inv.select(station=sta_code)

    # Save StationXML
    stationxml_path = stations_dir / "CI.xml"
    inv_selected.write(str(stationxml_path), format="STATIONXML")
    print(f"StationXML saved to {stationxml_path} ({len(inv_selected.get_contents()['stations'])} stations)")

    # Download waveforms
    t1 = origin_time - 60    # 60 s before
    t2 = origin_time + 300   # 300 s after

    print(f"Downloading waveforms from {t1} to {t2}...")
    for sta_code in station_list:
        print(f"  Fetching {network}.{sta_code}.*.{channel}...")
        try:
            st = client.get_waveforms(
                network=network,
                station=sta_code,
                location="*",
                channel=channel,
                starttime=t1,
                endtime=t2,
            )
            outfile = data_dir / f"{network}.{sta_code}.mseed"
            st.write(str(outfile), format="MSEED")
            print(f"    -> {outfile} ({len(st)} traces)")
        except Exception as e:
            print(f"    -> FAILED: {e}")

    # Write a config file tuned for this event
    config_path = base_dir / "ridgecrest_config.yaml"
    config_content = f"""\
# mtinvert configuration for Ridgecrest M6.4 foreshock example
# Event: 2019-07-04 17:33:49 UTC, M6.4, Ridgecrest CA

# --- Paths ---
data_dir: "{data_dir}"
stationxml_path: "{stationxml_path}"
output_dir: "{base_dir / 'output'}"
gf_library_path: "{base_dir / 'gf_library' / 'ridgecrest.h5'}"

# --- 1D Velocity Model (SoCal) ---
velocity_model:
  - thickness_km: 5.5
    vp_km_s: 5.5
    vs_km_s: 3.18
    rho_g_cc: 2.4
    qp: 600
    qs: 300
  - thickness_km: 10.5
    vp_km_s: 6.3
    vs_km_s: 3.64
    rho_g_cc: 2.67
    qp: 600
    qs: 300
  - thickness_km: 16.0
    vp_km_s: 6.7
    vs_km_s: 3.87
    rho_g_cc: 2.8
    qp: 600
    qs: 300
  - thickness_km: 0.0
    vp_km_s: 7.8
    vs_km_s: 4.5
    rho_g_cc: 3.0
    qp: 1000
    qs: 500

# --- Green's Function Grid ---
distance_min_km: 50.0
distance_max_km: 200.0
distance_step_km: 5.0
depth_min_km: 2.0
depth_max_km: 20.0
depth_step_km: 2.0

# --- Waveform Processing ---
freqmin_hz: 0.02
freqmax_hz: 0.05
pre_event_sec: 30.0
post_event_sec: 150.0
taper_fraction: 0.05
snr_threshold: 2.0
water_level_db: 60.0

# --- Inversion ---
inversion_type: deviatoric

# --- Parallel Processing ---
n_workers: 1
"""
    config_path.write_text(config_content)
    print(f"Config saved to {config_path}")

    print(f"""
{'='*60}
Data download complete!

To run the example:

  1. Build Green's functions (uses analytic approximation):
     mtinvert build-gf {config_path} --no-cps

  2. Run the inversion:
     mtinvert invert {config_path} \\
       --event-lat {event_lat} --event-lon {event_lon} \\
       --origin-time {origin_time} \\
       --event-id ridgecrest_m64

{'='*60}
""")


if __name__ == "__main__":
    main()
