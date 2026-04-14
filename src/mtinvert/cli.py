"""
Command-line interface for mtinvert.

Usage:
    mtinvert build-gf config.yaml
    mtinvert invert config.yaml --event-lat 35.0 --event-lon -118.0 --origin-time 2024-01-15T10:30:00
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_build_gf(args):
    """Build Green's function library."""
    from .config import Config
    from .greens import build_gf_library

    cfg = Config.from_yaml(args.config)
    errors = cfg.validate()
    if errors:
        for e in errors:
            print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    build_gf_library(
        cfg,
        use_cps=not args.no_cps,
        dt=args.dt,
        npts=args.npts,
    )


def cmd_invert(args):
    """Run moment tensor inversion with depth grid search."""
    from obspy import UTCDateTime

    from .config import Config
    from .inversion import depth_grid_search, write_solution_csv
    from .preprocess import preprocess_event
    from .visualization import (
        plot_depth_search,
        plot_focal_mechanism,
        plot_station_map,
        plot_summary,
        plot_waveform_fits,
    )

    cfg = Config.from_yaml(args.config)
    errors = cfg.validate()
    if errors:
        for e in errors:
            print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    origin = UTCDateTime(args.origin_time)
    event_id = args.event_id or origin.strftime("%Y%m%d_%H%M%S")

    # Preprocess
    stations = preprocess_event(
        cfg,
        event_latitude=args.event_lat,
        event_longitude=args.event_lon,
        event_origin_time=origin,
    )

    accepted = [s for s in stations if not s.rejected]
    if len(accepted) < 2:
        print("Fewer than 2 stations passed QC. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Inversion
    result = depth_grid_search(stations, cfg)
    if result is None:
        print("Inversion failed at all depths.", file=sys.stderr)
        sys.exit(1)

    # Output
    out_dir = Path(cfg.output_dir) / event_id
    out_dir.mkdir(parents=True, exist_ok=True)

    best = result.best_result

    # CSV
    write_solution_csv(result, out_dir / "depth_search.csv")

    # Plots
    plot_waveform_fits(
        best, stations, out_dir / "waveform_fits.pdf", event_id=event_id
    )
    plot_focal_mechanism(
        best.decomposition, out_dir / "focal_mechanism.pdf", event_id=event_id
    )
    plot_depth_search(
        result, out_dir / "depth_search.pdf", event_id=event_id
    )
    plot_station_map(
        stations, args.event_lat, args.event_lon,
        out_dir / "station_map.pdf", event_id=event_id,
    )
    plot_summary(
        result, stations, args.event_lat, args.event_lon,
        out_dir / "summary.pdf", event_id=event_id,
    )

    # Print solution
    dec = best.decomposition
    print(f"\n{'='*60}")
    print(f"Event: {event_id}")
    print(f"Best depth: {best.depth_km:.1f} km")
    print(f"Mw: {dec.moment_magnitude:.2f}")
    print(f"M0: {dec.scalar_moment:.2e} N*m")
    print(f"VR: {best.variance_reduction:.1%}")
    print(f"DC: {dec.dc_percent:.1f}%  CLVD: {dec.clvd_percent:.1f}%  ISO: {dec.iso_percent:.1f}%")
    print(f"NP1: strike={dec.strike1:.0f} dip={dec.dip1:.0f} rake={dec.rake1:.0f}")
    print(f"NP2: strike={dec.strike2:.0f} dip={dec.dip2:.0f} rake={dec.rake2:.0f}")
    print(f"Condition number: {best.condition_number:.2e}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        prog="mtinvert",
        description="Automated regional moment tensor inversion",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-gf
    p_gf = subparsers.add_parser("build-gf", help="Build Green's function library")
    p_gf.add_argument("config", help="YAML configuration file")
    p_gf.add_argument("--no-cps", action="store_true",
                       help="Use analytic approximation instead of CPS")
    p_gf.add_argument("--dt", type=float, default=0.1, help="Sample interval (s)")
    p_gf.add_argument("--npts", type=int, default=2048, help="Samples per trace")

    # invert
    p_inv = subparsers.add_parser("invert", help="Run MT inversion")
    p_inv.add_argument("config", help="YAML configuration file")
    p_inv.add_argument("--event-lat", type=float, required=True)
    p_inv.add_argument("--event-lon", type=float, required=True)
    p_inv.add_argument("--origin-time", required=True, help="ISO format origin time")
    p_inv.add_argument("--event-id", default="", help="Event identifier string")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "build-gf":
        cmd_build_gf(args)
    elif args.command == "invert":
        cmd_invert(args)


if __name__ == "__main__":
    main()
