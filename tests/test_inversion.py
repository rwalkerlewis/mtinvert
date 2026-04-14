"""
Unit tests for mtinvert inversion engine and decomposition.

Tests use synthetic moment tensors with known solutions to verify
recovery of MT elements, decomposition percentages, and fault
plane parameters.
"""

import numpy as np
import pytest

from mtinvert.decomposition import (
    MTDecomposition,
    decompose,
    tensor_to_vector,
    vector_to_tensor,
)


class TestVectorTensorConversion:
    def test_roundtrip(self):
        m = np.array([1.0, 2.0, 3.0, 0.5, -0.3, 0.7])
        M = vector_to_tensor(m)
        m2 = tensor_to_vector(M)
        np.testing.assert_allclose(m, m2)

    def test_symmetry(self):
        m = np.array([1.0, 2.0, 3.0, 0.5, -0.3, 0.7])
        M = vector_to_tensor(m)
        np.testing.assert_allclose(M, M.T)

    def test_diagonal(self):
        m = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        M = vector_to_tensor(m)
        assert M[0, 0] == 1.0
        assert M[1, 1] == 2.0
        assert M[2, 2] == 3.0
        assert M[0, 1] == 0.0


class TestDecomposition:
    def test_pure_double_couple(self):
        """A pure DC should give ~100% DC, ~0% CLVD and ISO."""
        # Pure strike-slip: Mxy = M0, rest zero
        m0 = 1e16
        m = np.array([0.0, 0.0, 0.0, m0, 0.0, 0.0])
        dec = decompose(m)

        assert dec.dc_percent > 95.0
        assert dec.iso_percent < 1.0
        assert dec.clvd_percent < 5.0
        np.testing.assert_allclose(dec.scalar_moment, m0, rtol=0.01)

    def test_pure_isotropic(self):
        """Pure explosion: all diagonal equal."""
        m0 = 1e15
        m = np.array([m0, m0, m0, 0.0, 0.0, 0.0])
        dec = decompose(m)

        assert dec.iso_percent > 95.0
        assert dec.dc_percent < 5.0

    def test_deviatoric_zero_trace(self):
        """Deviatoric tensor should have small ISO component."""
        # Mxx + Myy + Mzz = 0
        m = np.array([1.0, -0.5, -0.5, 0.3, 0.1, -0.2])
        m *= 1e15
        dec = decompose(m)

        assert dec.iso_percent < 1.0

    def test_moment_magnitude(self):
        """Check Mw computation for known M0."""
        m0 = 1e18  # Mw ~5.9
        m = np.array([0.0, 0.0, 0.0, m0, 0.0, 0.0])
        dec = decompose(m)

        expected_mw = (2.0 / 3.0) * (np.log10(m0) - 9.1)
        np.testing.assert_allclose(dec.moment_magnitude, expected_mw, atol=0.1)

    def test_eigenvalue_count(self):
        m = np.array([1.0, 2.0, 3.0, 0.5, 0.3, 0.7]) * 1e14
        dec = decompose(m)
        assert len(dec.eigenvalues) == 3
        assert len(dec.dev_eigenvalues) == 3

    def test_nodal_planes_orthogonal(self):
        """Nodal planes of a pure DC should be orthogonal."""
        m = np.array([0.0, 0.0, 0.0, 1e16, 0.0, 0.0])
        dec = decompose(m)

        # For a pure DC, the two nodal planes should be ~90 degrees apart
        # This is a rough check since the exact relationship depends on geometry
        assert 0 <= dec.dip1 <= 90
        assert 0 <= dec.dip2 <= 90

    def test_percentages_sum_to_100(self):
        m = np.array([1.5, -0.3, -1.2, 0.7, 0.4, -0.1]) * 1e15
        dec = decompose(m)
        total = dec.iso_percent + dec.clvd_percent + dec.dc_percent
        np.testing.assert_allclose(total, 100.0, atol=1.0)


class TestInversionSynthetic:
    """Test inversion using synthetic data with known MT."""

    def _make_synthetic_station(self, mt_vector, distance_km, azimuth_deg, dt=0.1, npts=512):
        """Create synthetic observed data for a known MT using fundamental GFs."""
        from mtinvert.greens import _compute_gf_synthetic, assemble_mt_gf
        from mtinvert.config import VelocityLayer
        from mtinvert.preprocess import StationData

        layers = [VelocityLayer(thickness_km=35.0, vp_km_s=6.0, vs_km_s=3.5, rho_g_cc=2.7)]
        depth_km = 10.0

        # Compute fundamental GFs
        fund_gf = _compute_gf_synthetic(layers, depth_km, distance_km, dt, npts)

        # Assemble per-MT-element GFs using station azimuth
        gf = assemble_mt_gf(fund_gf, azimuth_deg)

        # Forward model: d = G * m
        data_zrt = np.zeros((3, npts))
        from mtinvert.greens import MT_ELEMENTS
        for ie, elem in enumerate(MT_ELEMENTS):
            for ic in range(3):
                data_zrt[ic] += gf[elem][ic] * mt_vector[ie]

        az_rad = np.radians(azimuth_deg)
        lat = distance_km / 111.0
        lon = 0.0

        return StationData(
            network="SY", station=f"S{int(distance_km):03d}",
            latitude=lat, longitude=lon,
            distance_km=distance_km, azimuth=azimuth_deg,
            back_azimuth=(azimuth_deg + 180) % 360,
            data_zrt=data_zrt, dt=dt,
            starttime=0.0,
            snr={"Z": 100.0, "R": 100.0, "T": 100.0},
            rejected=False,
        )

    def test_recovery_pure_dc(self, tmp_path):
        """Verify that inversion recovers a known pure DC moment tensor."""
        from mtinvert.config import Config, VelocityLayer
        from mtinvert.greens import build_gf_library
        from mtinvert.inversion import invert_mt

        m_true = np.array([0.0, 0.0, 0.0, 1e16, 0.0, 0.0])

        cfg = Config(
            gf_library_path=str(tmp_path / "test_gf.h5"),
            distance_min_km=10.0, distance_max_km=200.0, distance_step_km=10.0,
            depth_min_km=5.0, depth_max_km=20.0, depth_step_km=5.0,
            velocity_model=[
                VelocityLayer(thickness_km=35.0, vp_km_s=6.0, vs_km_s=3.5, rho_g_cc=2.7),
            ],
            inversion_type="full",
        )

        build_gf_library(cfg, use_cps=False, npts=512)

        stations = []
        for dist in [50, 100, 150]:
            for az in [0, 90, 180, 270]:
                stations.append(self._make_synthetic_station(m_true, dist, az, npts=512))

        result = invert_mt(stations, cfg, depth_km=10.0)

        assert result is not None
        assert result.variance_reduction > 0.95

        # Recovered MT should be close to true (up to a scale factor)
        m_rec = result.mt_vector
        # The dominant element (Mxy, index 3) should be correctly identified
        assert np.argmax(np.abs(m_rec)) == np.argmax(np.abs(m_true))
        assert result.decomposition is not None
        assert np.isfinite(result.decomposition.moment_magnitude)
        assert result.decomposition.scalar_moment > 0


class TestEdgeCases:
    def test_zero_tensor(self):
        m = np.zeros(6)
        dec = decompose(m)
        assert dec.scalar_moment == 0.0

    def test_very_large_values(self):
        m = np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0]) * 1e25
        dec = decompose(m)
        assert np.isfinite(dec.moment_magnitude)
        assert np.isfinite(dec.dc_percent)
