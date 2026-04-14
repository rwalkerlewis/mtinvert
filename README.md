# mtinvert

Automated regional moment tensor inversion from broadband seismic waveforms.

`mtinvert` performs deviatoric (or full) moment tensor inversion using pre-computed Green's function libraries and linear least-squares with depth grid search. It reads standard miniSEED waveforms and StationXML metadata, and produces publication-quality figures of waveform fits, focal mechanisms, and solution diagnostics.

## Features

- **Green's function database**: Builds HDF5 libraries from 1D velocity models using CPS (Computer Programs in Seismology) FK integration, with an analytic fallback for testing
- **Automated preprocessing**: Instrument response removal, NE-to-RT rotation, bandpass filtering, SNR-based station/component rejection
- **Linear MT inversion**: Solves for 6 independent moment tensor elements with optional deviatoric (trace-free) constraint
- **Depth grid search**: Full inversion at each trial depth with variance reduction and condition number diagnostics
- **Decomposition**: ISO/CLVD/DC percentages, scalar moment, Mw, fault plane solutions following Jost & Herrmann (1989)
- **Visualization**: Waveform fits, beachball focal mechanisms, depth-misfit curves, station maps, combined summary figures (all PDF-exportable)
- **CLI and YAML configuration**: Single command to run the full workflow

## Installation

```bash
git clone https://github.com/username/mtinvert.git
cd mtinvert
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

### Dependencies

- Python >= 3.9
- ObsPy >= 1.4.0
- NumPy, SciPy, Matplotlib, h5py, PyYAML

Optional (for FK Green's functions):
- [Computer Programs in Seismology](http://www.eas.slu.edu/eqc/eqccps.html) -- `hprep96`, `hspec96`, `hpulse96`, `f96tosac` must be on PATH

## Quickstart

### 1. Create a configuration file

Copy and edit the example:

```bash
cp examples/example_config.yaml my_config.yaml
```

Edit `my_config.yaml` to set your velocity model, data paths, frequency band, and depth search range.

### 2. Build the Green's function library

```bash
mtinvert build-gf my_config.yaml
```

If CPS is not installed, use the analytic approximation (suitable for testing only):

```bash
mtinvert build-gf my_config.yaml --no-cps
```

This creates an HDF5 file at the path specified by `gf_library_path` in your config.

### 3. Run the inversion

```bash
mtinvert invert my_config.yaml \
    --event-lat 34.05 \
    --event-lon -118.25 \
    --origin-time 2024-07-15T08:30:00 \
    --event-id ridgecrest_test
```

Output goes to `output_dir/event_id/` and includes:
- `depth_search.csv` -- MT solution at each trial depth
- `waveform_fits.pdf` -- observed vs. synthetic for all stations
- `focal_mechanism.pdf` -- beachball with decomposition info
- `depth_search.pdf` -- VR and misfit vs. depth
- `station_map.pdf` -- station and event locations
- `summary.pdf` -- combined figure

## Configuration Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_dir` | string | `"."` | Directory containing miniSEED files |
| `stationxml_path` | string | `"stations.xml"` | Path to StationXML response file |
| `output_dir` | string | `"output"` | Output directory |
| `gf_library_path` | string | `"greens.h5"` | Path to GF HDF5 library |
| `velocity_model` | list | required | 1D velocity model layers |
| `distance_min_km` | float | 10.0 | Minimum epicentral distance |
| `distance_max_km` | float | 300.0 | Maximum epicentral distance |
| `distance_step_km` | float | 5.0 | Distance grid spacing |
| `depth_min_km` | float | 1.0 | Minimum trial depth |
| `depth_max_km` | float | 30.0 | Maximum trial depth |
| `depth_step_km` | float | 1.0 | Depth grid spacing |
| `freqmin_hz` | float | 0.02 | Lower corner frequency |
| `freqmax_hz` | float | 0.1 | Upper corner frequency |
| `pre_event_sec` | float | 30.0 | Pre-origin window for noise |
| `post_event_sec` | float | 120.0 | Post-origin window |
| `taper_fraction` | float | 0.05 | Cosine taper fraction |
| `snr_threshold` | float | 3.0 | Minimum SNR for acceptance |
| `water_level_db` | float | 60.0 | Water level for response removal |
| `inversion_type` | string | `"deviatoric"` | `"deviatoric"` or `"full"` |
| `n_workers` | int | 1 | Parallel workers for GF computation |

## API Usage

```python
from mtinvert.config import Config
from mtinvert.greens import build_gf_library
from mtinvert.preprocess import preprocess_event
from mtinvert.inversion import depth_grid_search, write_solution_csv
from mtinvert.visualization import plot_summary

cfg = Config.from_yaml("my_config.yaml")

# Build GFs (one-time)
build_gf_library(cfg, use_cps=True)

# Process and invert
stations = preprocess_event(cfg, event_lat, event_lon, origin_time)
result = depth_grid_search(stations, cfg)

# Output
best = result.best_result
print(f"Mw {best.decomposition.moment_magnitude:.2f} at {best.depth_km:.1f} km")
write_solution_csv(result, "solution.csv")
plot_summary(result, stations, event_lat, event_lon, "summary.pdf")
```

## Testing

```bash
pytest tests/ -v
```

The test suite uses synthetic data with known moment tensors to verify:
- Vector/tensor conversion roundtrip
- Pure DC decomposition (>95% DC)
- Pure isotropic decomposition (>95% ISO)
- Moment magnitude computation
- Percentage sum constraint (ISO + CLVD + DC = 100%)
- Full inversion recovery of known MT from synthetic waveforms

## References

- Jost, M. L., & Herrmann, R. B. (1989). A student's guide to and review of moment tensors. Seismological Research Letters, 60(2), 37-57.
- Herrmann, R. B. (2013). Computer Programs in Seismology: An evolving tool for instruction and research. Seismological Research Letters, 84(6), 1081-1088.

## License

MIT
# mtinvert
