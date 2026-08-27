import hashlib
from datetime import datetime, timezone

import numpy as np

from eopy.data import DEFAULT_GRAVITY_MODEL_FILENAME, DEFAULT_GRAVITY_MODEL_URL, download_file, download_gravity_model, read_icgem_gfc, resolve_data_path
from eopy.gravity import GravityField, central_gravity, gravity_acceleration, j2_acceleration
from eopy import native


def test_resolve_data_path_uses_env_override(tmp_path, monkeypatch):
    root = tmp_path / "eopy-data"
    monkeypatch.setenv("EOPY_DATA_PATH", str(root))

    assert resolve_data_path() == root
    assert root.exists()


def test_download_file_uses_atomic_cache_and_checksum(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("eopy data\n", encoding="utf-8")
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    out = download_file(source.as_uri(), "cached.txt", target_dir=tmp_path / "cache", sha256=expected_hash)

    assert out.name == "cached.txt"
    assert out.read_text(encoding="utf-8") == "eopy data\n"
    assert download_file(source.as_uri(), "cached.txt", target_dir=tmp_path / "cache", sha256=expected_hash) == out


def test_default_gravity_download_url_is_eigen_6s4_v2(monkeypatch, tmp_path):
    captured = {}

    def fake_download_file(url, filename=None, **kwargs):
        captured["url"] = url
        captured["filename"] = filename
        captured["kwargs"] = kwargs
        return tmp_path / filename

    monkeypatch.setattr("eopy.data.download_file", fake_download_file)

    out = download_gravity_model(target_dir=tmp_path)

    assert captured["url"] == DEFAULT_GRAVITY_MODEL_URL
    assert captured["filename"] == DEFAULT_GRAVITY_MODEL_FILENAME
    assert out == tmp_path / DEFAULT_GRAVITY_MODEL_FILENAME


def test_read_icgem_gfc_parses_dense_coefficients(tmp_path):
    gfc = tmp_path / "tiny.gfc"
    gfc.write_text(
        "\n".join(
            [
                "modelname tiny",
                "earth_gravity_constant 398600441800000.0",
                "radius 6378136.3",
                "max_degree 2",
                "norm unnormalized",
                "end_of_head",
                "gfc 0 0 1.0 0.0",
                "gfc 2 0 -0.001 0.0",
                "gfc 2 1 0.0 2.0e-6",
            ]
        ),
        encoding="utf-8",
    )

    model = read_icgem_gfc(gfc)

    assert model.name == "tiny"
    assert model.max_degree == 2
    assert model.max_order == 1
    assert model.normalized is False
    np.testing.assert_allclose(model.c[0, 0], 1.0)
    np.testing.assert_allclose(model.c[2, 0], -0.001)
    np.testing.assert_allclose(model.s[2, 1], 2.0e-6)
    assert model.c.flags.c_contiguous


def test_gravity_field_from_icgem_model_uses_coefficients(tmp_path):
    gfc = tmp_path / "central.gfc"
    gfc.write_text(
        "\n".join(
            [
                "modelname central",
                "earth_gravity_constant 398600441800000.0",
                "radius 6378136.3",
                "norm unnormalized",
                "end_of_head",
                "gfc 0 0 1.0 0.0",
            ]
        ),
        encoding="utf-8",
    )
    model = read_icgem_gfc(gfc)
    position = np.array([[7_000_000.0, 0.0, 0.0], [0.0, 7_100_000.0, 0.0]])

    acc = gravity_acceleration(position, GravityField.from_model(model, backend="python"))

    np.testing.assert_allclose(acc, central_gravity(position, model.mu), rtol=0.0, atol=2e-6)


def test_read_icgem_gfc_evaluates_temporal_coefficients(tmp_path):
    gfc = tmp_path / "temporal.gfc"
    gfc.write_text(
        "\n".join(
            [
                "modelname temporal",
                "earth_gravity_constant 398600441800000.0",
                "radius 6378136.3",
                "norm fully_normalized",
                "end_of_head",
                "gfc 0 0 1.0 0.0",
                "gfct 2 0 -4.0e-4 0.0 20200101.0000 20300101.0000",
                "trnd 2 0 1.0e-6 0.0 20200101.0000 20300101.0000",
                "acos 2 0 2.0e-6 0.0 20200101.0000 20300101.0000 1.0",
            ]
        ),
        encoding="utf-8",
    )

    with np.testing.assert_raises_regex(ValueError, "provide epoch"):
        read_icgem_gfc(gfc)
    model = read_icgem_gfc(
        gfc,
        epoch=datetime(2022, 1, 1, tzinfo=timezone.utc),
    )

    years = 731.0 / 365.25
    expected = -4.0e-4 + 1.0e-6 * years + 2.0e-6 * np.cos(2.0 * np.pi * years)
    np.testing.assert_allclose(model.c[2, 0], expected, atol=1e-15)


def test_native_normalized_c20_matches_analytic_j2_with_truncated_order(tmp_path):
    if not native.available():
        import pytest

        pytest.skip("native gravity library unavailable")
    c = np.zeros((5, 5))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -1.0826266835531513e-3 / np.sqrt(5.0)
    from eopy.data import GravityModel

    model = GravityModel("c20", 3.986004418e14, 6378136.3, 4, 4, c, s, True)
    positions = np.array([[7.0e6, 1.0e5, 1.0e6], [2.0e5, -6.8e6, 5.0e5]])
    actual = gravity_acceleration(
        positions,
        GravityField.from_model(model, degree=2, order=0, backend="native"),
    )
    expected = central_gravity(positions, model.mu) + j2_acceleration(
        positions, model.mu, model.reference_radius, 1.0826266835531513e-3
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-9)
