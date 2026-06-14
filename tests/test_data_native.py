import hashlib

import numpy as np

from eopy.data import DEFAULT_GRAVITY_MODEL_FILENAME, DEFAULT_GRAVITY_MODEL_URL, download_file, download_gravity_model, read_icgem_gfc, resolve_data_path
from eopy.gravity import GravityField, central_gravity, gravity_acceleration
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


def test_native_module_reports_availability_without_import_failure():
    is_available = native.available()
    assert isinstance(is_available, bool)
    if is_available:
        assert native.version()


def test_native_diagnostics_include_build_guidance():
    diagnostics = native.library_diagnostics()

    assert "checked paths:" in diagnostics
    assert "cmake -S . -B build/native" in diagnostics
    assert "EOPY_NATIVE_LIBRARY" in diagnostics
