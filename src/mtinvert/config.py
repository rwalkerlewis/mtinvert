"""Configuration loader and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class VelocityLayer:
    thickness_km: float
    vp_km_s: float
    vs_km_s: float
    rho_g_cc: float
    qp: float = 600.0
    qs: float = 300.0


@dataclass
class Config:
    # Paths
    data_dir: str = "."
    stationxml_path: str = "stations.xml"
    output_dir: str = "output"
    gf_library_path: str = "greens.h5"

    # Velocity model
    velocity_model: list[VelocityLayer] = field(default_factory=list)

    # Green's function grid
    distance_min_km: float = 10.0
    distance_max_km: float = 300.0
    distance_step_km: float = 5.0
    depth_min_km: float = 1.0
    depth_max_km: float = 30.0
    depth_step_km: float = 1.0

    # Processing
    freqmin_hz: float = 0.02
    freqmax_hz: float = 0.1
    pre_event_sec: float = 30.0
    post_event_sec: float = 120.0
    taper_fraction: float = 0.05
    snr_threshold: float = 3.0
    water_level_db: float = 60.0

    # Inversion
    n_tapers: int = 4
    nfft: Optional[int] = None
    inversion_type: str = "deviatoric"  # "deviatoric" or "full"

    # Parallel
    n_workers: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        layers = []
        for lyr in raw.pop("velocity_model", []):
            layers.append(VelocityLayer(**lyr))

        cfg = cls(**raw)
        cfg.velocity_model = layers
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        d = self.__dict__.copy()
        d["velocity_model"] = [v.__dict__ for v in d["velocity_model"]]
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> list[str]:
        errors = []
        if not self.velocity_model:
            errors.append("velocity_model must have at least one layer")
        if self.freqmin_hz >= self.freqmax_hz:
            errors.append("freqmin_hz must be less than freqmax_hz")
        if self.depth_min_km >= self.depth_max_km:
            errors.append("depth_min_km must be less than depth_max_km")
        if self.distance_min_km >= self.distance_max_km:
            errors.append("distance_min_km must be less than distance_max_km")
        if self.snr_threshold <= 0:
            errors.append("snr_threshold must be positive")
        if self.inversion_type not in ("deviatoric", "full"):
            errors.append("inversion_type must be 'deviatoric' or 'full'")
        return errors
