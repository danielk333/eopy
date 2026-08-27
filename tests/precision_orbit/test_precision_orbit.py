"""Validate eopy against a Sentinel-2B precise-orbit product.

The test uses the first precise state as the initial condition.  Every later
state is then an independent reference point for the same propagated orbit.
An EOF file can be supplied with ``--precision-orbit-file`` (or
``EOPY_PRECISION_ORBIT_FILE``); otherwise it is cached below ``EOPY_DATA_PATH``
when Copernicus Data Space credentials are available.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from astropy.time import Time
from spacecoords import celestial

from eopy.atmosphere import AtmosphereConfig, CelestrakSpaceWeather
from eopy.data import read_icgem_gfc
from eopy.frames import interpolated_earth_orientation
from eopy.gravity import GravityField
from eopy.integrators import Dop853Options
from eopy.propagator import SpacecraftProperties, propagate
from eopy.profiling import PropagationProfile
from eopy.radiation import RadiationPressure
from eopy.third_body import ThirdBody, interpolated_astropy_body_providers


UTC = dt.timezone.utc
START = dt.datetime(2024, 7, 4, 10, 15, tzinfo=UTC)
END = START + dt.timedelta(hours=24)
SUN_MU = 1.32712440018e20
MOON_MU = 4.902800118e12

# Approximate launch-configuration-independent properties for Sentinel-2B.
# At its roughly 786 km altitude, drag is small, but it is retained so that the
# precision-orbit comparison exercises the complete available LEO force model.
SENTINEL_2B = SpacecraftProperties(
    mass_kg=1_140.0,
    drag_area_m2=10.0,
    drag_coefficient=2.2,
)
SENTINEL_2B_RADIATION = RadiationPressure(
    mass_kg=1_140.0,
    area_m2=10.0,
    reflection_coefficient=1.3,
    shadow_model="conical",
)


def _request_json(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None):
    request = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(request, timeout=60.0) as response:
        return json.load(response)


def _download_precision_orbit(path: Path, username: str, password: str) -> Path:
    """Download the Sentinel-2B product covering ``START`` atomically."""

    query_start = (START - dt.timedelta(days=1)).isoformat().replace("+00:00", "Z")
    query_end = END.isoformat().replace("+00:00", "Z")
    predicate = (
        "Collection/Name eq 'SENTINEL-2' and "
        f"ContentDate/Start gt {query_start} and ContentDate/Start lt {query_end} and "
        "Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and a/Value eq 'AUX_POEORB')"
    )
    catalogue_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?" + urllib.parse.urlencode(
        {"$filter": predicate, "$orderby": "ContentDate/Start", "$top": "20"}
    )
    products = _request_json(catalogue_url)["value"]
    products = [product for product in products if "S2B" in product.get("Name", "")]
    if not products:
        raise RuntimeError("CDSE returned no Sentinel-2B AUX_POEORB product for the test interval")

    def covers_start(product):
        content = product.get("ContentDate", {})
        product_start = dt.datetime.fromisoformat(content["Start"].replace("Z", "+00:00"))
        product_end = dt.datetime.fromisoformat(content["End"].replace("Z", "+00:00"))
        return product_start <= START <= product_end

    covering_products = [product for product in products if covers_start(product)]
    if not covering_products:
        raise RuntimeError("CDSE returned no Sentinel-2B AUX_POEORB product covering the initial epoch")
    product = covering_products[-1]

    token_data = urllib.parse.urlencode(
        {
            "grant_type": "password",
            "username": username,
            "password": password,
            "client_id": "cdse-public",
        }
    ).encode()
    token = _request_json(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data=token_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )["access_token"]

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(
        f"https://download.dataspace.copernicus.eu/odata/v1/Products({product['Id']})/$value",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120.0) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _parse_earth_fixed_states(path: Path, start: dt.datetime, end: dt.datetime):
    root = ET.parse(path).getroot()
    epochs = []
    states = []
    for osv in root.findall(".//Data_Block//OSV"):
        epoch = dt.datetime.fromisoformat(osv.findtext("UTC").removeprefix("UTC=").replace("Z", "+00:00"))
        # Sentinel EOF timestamps are UTC, but products commonly omit the
        # trailing ``Z`` and therefore parse as offset-naive datetimes.
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=UTC)
        else:
            epoch = epoch.astimezone(UTC)
        if start <= epoch <= end:
            epochs.append(epoch)
            states.append([float(osv.findtext(name)) for name in ("X", "Y", "Z", "VX", "VY", "VZ")])
    if len(states) < 2:
        raise ValueError(f"fewer than two precise states in {path} between {start} and {end}")
    return Time(epochs, scale="utc"), np.asarray(states, dtype=np.float64)


def test_parse_earth_fixed_states_accepts_eof_utc_timestamps(tmp_path):
    orbit_file = tmp_path / "orbit.eof"
    orbit_file.write_text(
        """<Earth_Explorer_File><Data_Block><List_of_OSVs>
        <OSV><UTC>UTC=2024-07-04T10:15:00.000000</UTC><X>1</X><Y>2</Y><Z>3</Z><VX>4</VX><VY>5</VY><VZ>6</VZ></OSV>
        <OSV><UTC>UTC=2024-07-04T10:15:10.000000Z</UTC><X>7</X><Y>8</Y><Z>9</Z><VX>10</VX><VY>11</VY><VZ>12</VZ></OSV>
        </List_of_OSVs></Data_Block></Earth_Explorer_File>""",
        encoding="utf-8",
    )

    epochs, states = _parse_earth_fixed_states(
        orbit_file,
        dt.datetime(2024, 7, 4, 10, 15, tzinfo=UTC),
        dt.datetime(2024, 7, 4, 10, 16, tzinfo=UTC),
    )

    assert len(epochs) == 2
    np.testing.assert_array_equal(states[:, 0], [1.0, 7.0])


def _propagation_errors(
    epochs: Time,
    earth_fixed_states: np.ndarray,
    gravity_model_path: Path,
    space_weather_path: Path,
):
    reference = celestial.convert(epochs, earth_fixed_states.T, "ITRS", "GCRS").T
    elapsed_s = np.asarray((epochs - epochs[0]).to_value("sec"), dtype=np.float64)
    propagated = np.empty_like(reference)
    propagated[0] = reference[0]
    initial_epoch = epochs[0].to_datetime(timezone=UTC)
    profile = PropagationProfile()
    orientation = interpolated_earth_orientation(
        initial_epoch,
        elapsed_s[0],
        elapsed_s[-1],
        spacing_s=60.0,
    )
    gravity_model = read_icgem_gfc(
        gravity_model_path,
        max_degree=120,
        max_order=120,
        epoch=initial_epoch,
    )
    body_positions = interpolated_astropy_body_providers(
        ["sun", "moon"],
        initial_epoch,
        elapsed_s[0],
        elapsed_s[-1],
        spacing_s=300.0,
        profile=profile,
    )
    sun_position = body_positions["sun"]
    sun = ThirdBody(
        mu=SUN_MU,
        position_m=sun_position,
        name="Sun",
    )
    moon = ThirdBody(
        mu=MOON_MU,
        position_m=body_positions["moon"],
        name="Moon",
    )
    atmosphere = AtmosphereConfig(model="msis", epoch=initial_epoch)
    activity = CelestrakSpaceWeather.from_csv(space_weather_path)
    options = Dop853Options(rtol=1e-12, atol=1e-5, max_step=30.0)
    _, propagated, _ = propagate(
        reference[0],
        elapsed_s[0],
        elapsed_s[-1],
        gravity=GravityField.from_model(
            gravity_model,
            degree=120,
            order=120,
            backend="native",
        ),
        spacecraft=SENTINEL_2B,
        atmosphere=atmosphere,
        activity=activity,
        radiation=SENTINEL_2B_RADIATION,
        sun_position=sun_position,
        third_bodies=(sun, moon),
        earth_orientation=orientation,
        relativity=True,
        options=options,
        sample_times=elapsed_s,
        profile=profile,
    )

    position_delta = propagated[:, :3] - reference[:, :3]
    radial = reference[:, :3] / np.linalg.norm(reference[:, :3], axis=1)[:, None]
    cross_track = np.cross(reference[:, :3], reference[:, 3:])
    cross_track /= np.linalg.norm(cross_track, axis=1)[:, None]
    along_track = np.cross(cross_track, radial)
    ric_error = np.column_stack(
        (
            np.sum(position_delta * radial, axis=1),
            np.sum(position_delta * along_track, axis=1),
            np.sum(position_delta * cross_track, axis=1),
        )
    )

    return {
        "time_s": elapsed_s,
        "reference_gcrs": reference,
        "propagated_gcrs": propagated,
        "position_error_m": np.linalg.norm(propagated[:, :3] - reference[:, :3], axis=1),
        "velocity_error_mps": np.linalg.norm(propagated[:, 3:] - reference[:, 3:], axis=1),
        "ric_position_error_m": ric_error,
        "profile": profile,
    }


def plot_propagation_errors(result):
    """Plot position and velocity error histories and return the figure."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(result["time_s"] / 60.0, result["position_error_m"])
    axes[0].set_ylabel("Position error [m]")
    axes[0].grid(True)
    axes[1].plot(result["time_s"] / 60.0, result["velocity_error_mps"])
    axes[1].set(xlabel="Time since initial precise state [min]", ylabel="Velocity error [m/s]")
    axes[1].grid(True)
    figure.suptitle("eopy propagation versus Sentinel-2B precise orbit")
    figure.tight_layout()
    return figure


def test_propagation_against_precision_orbit(
    precision_orbit_paths,
    precision_environment_paths,
    cdse_credentials,
    plot,
):
    _cache_dir, orbit_file = precision_orbit_paths
    if not orbit_file.is_file():
        username, password = cdse_credentials
        if not username or not password:
            pytest.skip(
                "precision orbit is not cached; provide --precision-orbit-file or CDSE credentials"
            )
        _download_precision_orbit(orbit_file, username, password)

    epochs, earth_fixed_states = _parse_earth_fixed_states(orbit_file, START, END)
    gravity_model, space_weather = precision_environment_paths
    if gravity_model is None or space_weather is None:
        pytest.skip("precision propagation requires cached gravity and space-weather files")
    from eopy import native

    if not native.available():
        pytest.skip("precision propagation requires the native gravity library")
    result = _propagation_errors(epochs, earth_fixed_states, gravity_model, space_weather)
    print("\n" + result["profile"].report())
    print(
        "precision errors: "
        f"max={np.max(result['position_error_m']):.3f} m, "
        f"rms={np.sqrt(np.mean(result['position_error_m'] ** 2)):.3f} m, "
        f"final={result['position_error_m'][-1]:.3f} m, "
        f"max velocity={np.max(result['velocity_error_mps']):.6f} m/s"
    )

    if plot:
        import matplotlib.pyplot as plt

        plot_propagation_errors(result)
        plt.show()

    assert result["position_error_m"][0] == 0.0
    assert result["velocity_error_mps"][0] == 0.0
    assert np.max(result["position_error_m"]) < 100.0
    assert np.max(result["velocity_error_mps"]) < 0.15
