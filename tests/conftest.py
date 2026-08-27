import os
from pathlib import Path

import pytest

from eopy.data import DEFAULT_GRAVITY_MODEL_FILENAME, resolve_data_path


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", help="Show plots produced by validation tests.")
    parser.addoption(
        "--precision-orbit-file",
        type=Path,
        help="Path to a cached Sentinel-2B AUX_POEORB EOF file.",
    )
    parser.addoption(
        "--precision-orbit-cache",
        type=Path,
        help="Directory in which to cache precision-orbit downloads.",
    )
    parser.addoption("--cdse-user", help="Copernicus Data Space username.")
    parser.addoption("--cdse-password", help="Copernicus Data Space password.")
    parser.addoption("--gravity-model-file", type=Path, help="Path to an ICGEM gravity model.")
    parser.addoption("--space-weather-file", type=Path, help="Path to CelesTrak SW-All.csv.")


@pytest.fixture(scope="session")
def plot(request):
    return request.config.getoption("--plot")


@pytest.fixture(scope="session")
def precision_orbit_paths(request):
    cache_option = request.config.getoption("--precision-orbit-cache")
    cache_dir = (
        cache_option
        if cache_option is not None
        else resolve_data_path() / "precision_orbit"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    file_option = request.config.getoption("--precision-orbit-file")
    env_file = os.environ.get("EOPY_PRECISION_ORBIT_FILE")
    orbit_file = file_option or (Path(env_file) if env_file else cache_dir / "sentinel_2b_20240704.eof")
    return cache_dir, orbit_file


@pytest.fixture(scope="session")
def cdse_credentials(request):
    username = request.config.getoption("--cdse-user") or os.environ.get("COPERNICUS_USER")
    password = request.config.getoption("--cdse-password") or os.environ.get("COPERNICUS_PASSWORD")
    return username, password


def _first_existing(candidates):
    return next((path for path in candidates if path and path.is_file()), None)


@pytest.fixture(scope="session")
def precision_environment_paths(request):
    data_root = resolve_data_path()
    development_root = Path.home() / "data" / "eopy"
    gravity_option = request.config.getoption("--gravity-model-file")
    gravity_env = os.environ.get("EOPY_GRAVITY_MODEL_FILE")
    gravity = _first_existing(
        [
            gravity_option,
            Path(gravity_env) if gravity_env else None,
            data_root / DEFAULT_GRAVITY_MODEL_FILENAME,
            development_root / DEFAULT_GRAVITY_MODEL_FILENAME,
            development_root / "EIGEN-6S4%20(v2).gfc",
        ]
    )
    weather_option = request.config.getoption("--space-weather-file")
    weather_env = os.environ.get("EOPY_SPACE_WEATHER_FILE")
    weather = _first_existing(
        [
            weather_option,
            Path(weather_env) if weather_env else None,
            data_root / "SW-All.csv",
            development_root / "SW-All.csv",
        ]
    )
    return gravity, weather
