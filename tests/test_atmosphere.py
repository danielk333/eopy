from datetime import datetime, timezone

import numpy as np
import pymsis
from spacecoords import frames as spacecoords_frames

import eopy.atmosphere as atmosphere_module
from eopy.atmosphere import AtmosphereConfig, CelestrakSpaceWeather, SolarActivity, density_dtm2000, density_msis
from eopy.dtm2000 import density_from_inputs
from eopy.frames import ecef_to_geodetic_wgs84, interpolated_earth_orientation


def test_fast_geodetic_conversion_matches_spacecoords():
    positions = np.array(
        [
            [6_778_000.0, 0.0, 0.0],
            [2_000_000.0, -6_500_000.0, 1_500_000.0],
            [10.0, -5.0, 7_000_000.0],
            [0.0, 0.0, -6_878_137.0],
            [0.0, 0.0, 0.0],
        ]
    )
    expected = spacecoords_frames.ecef_to_geodetic_wgs84(*positions.T, degrees=True)
    actual = ecef_to_geodetic_wgs84(positions, degrees=True)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(actual[2], expected[2], rtol=0.0, atol=1e-3)

    scalar = ecef_to_geodetic_wgs84(positions[1], degrees=True)
    np.testing.assert_allclose(scalar, expected[:, 1], rtol=0.0, atol=1e-3)


def test_msis_density_matches_public_pymsis_for_batch():
    positions = np.array(
        [
            [6_778_000.0, 0.0, 0.0],
            [0.0, 6_878_000.0, 250_000.0],
            [-6_978_000.0, 100_000.0, -300_000.0],
        ]
    )
    activity = SolarActivity(f10=160.0, f10_mean=155.0, ap=8.0)
    config = AtmosphereConfig(
        model="msis",
        epoch=datetime(2003, 3, 5, 0, 24, tzinfo=timezone.utc),
    )
    actual = density_msis(positions, activity, config)

    lats, lons, alts_m = spacecoords_frames.ecef_to_geodetic_wgs84(*positions.T, degrees=True)
    count = len(positions)
    expected = pymsis.calculate(
        [config.epoch.replace(tzinfo=None)] * count,
        lons,
        lats,
        np.clip(alts_m, config.min_altitude_m, config.max_altitude_m) / 1000.0,
        f107s=np.full(count, activity.f10),
        f107as=np.full(count, activity.f10_mean),
        aps=np.full((count, 7), activity.ap),
        version=2.1,
    )[:, pymsis.Variable.MASS_DENSITY]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0.0)


def test_msis_public_fallback_is_batch_safe(monkeypatch):
    positions = np.array([[6_778_000.0, 0.0, 0.0], [0.0, 6_778_000.0, 0.0]])
    expected = density_msis(positions)
    monkeypatch.setattr(atmosphere_module, "_low_level_msis21_available", lambda: False)
    actual = density_msis(positions)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.0)


def test_msis_density_rejects_nonfinite_input():
    with np.testing.assert_raises(ValueError):
        density_msis(np.array([np.nan, 0.0, 0.0]))


def test_celestrak_space_weather_builds_msis_ap_history(tmp_path):
    header = "DATE,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,F10.7_OBS,F10.7_OBS_CENTER81"
    rows = [header]
    for day in range(1, 5):
        aps = ",".join(str(day * 10 + index) for index in range(1, 9))
        rows.append(f"2024-07-0{day},{aps},{day},150,145")
    path = tmp_path / "SW-All.csv"
    path.write_text("\n".join(rows), encoding="utf-8")

    activity = CelestrakSpaceWeather.from_csv(path)(
        datetime(2024, 7, 4, 7, 0, tzinfo=timezone.utc)
    )

    assert activity.f10 == 150.0
    assert activity.f10_mean == 145.0
    np.testing.assert_array_equal(activity.ap[:5], [4.0, 43.0, 42.0, 41.0, 38.0])


def test_celestrak_space_weather_builds_dtm_kp_history(tmp_path):
    header = "DATE,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,F10.7_OBS,F10.7_OBS_CENTER81"
    rows = [header]
    for day in range(1, 5):
        kp = ",".join(str(day * 10 + index) for index in range(1, 9))
        ap = ",".join(str(day * 10 + index) for index in range(1, 9))
        rows.append(f"2024-07-0{day},{kp},{ap},{day},150,145")
    path = tmp_path / "SW-All.csv"
    path.write_text("\n".join(rows), encoding="utf-8")

    activity = CelestrakSpaceWeather.from_csv(path)(
        datetime(2024, 7, 4, 7, 0, tzinfo=timezone.utc)
    )

    assert activity.kp_3h == 4.3
    np.testing.assert_allclose(activity.kp_24h, np.mean([4.3, 4.2, 4.1, 3.8, 3.7, 3.6, 3.5, 3.4]))


def test_dtm2000_matches_original_reference_cases():
    cases = [
        (185, 800_000.0, 40.0, 150.0, 1.8710001353820e-14),
        (275, 800_000.0, 40.0, 150.0, 2.8524195214905e-14),
        (15, 500_000.0, -70.0, 70.0, 1.5699108952425600e-13),
        (15, 800_000.0, -70.0, 70.0, 2.4123751406975562e-15),
    ]
    for day, altitude, latitude, flux, expected in cases:
        actual = density_from_inputs(
            day, altitude, 0.0, np.radians(latitude), 16.0 * np.pi / 12.0,
            flux, flux, 0.0, 0.0,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0.0)


def test_dtm2000_vector_api_uses_epoch_and_sun_geometry():
    config = AtmosphereConfig(
        model="dtm2000",
        epoch=datetime(2024, 7, 4, tzinfo=timezone.utc),
    )
    positions = np.array([[6_778_137.0, 0.0, 0.0], [0.0, 6_778_137.0, 0.0]])
    sun = np.array([1.0e11, 0.0, 0.0])
    activity = SolarActivity(kp_3h=2.0, kp_24h=1.5)

    density = density_dtm2000(
        positions, activity, config, time_s=0.0, sun_position_m=sun
    )

    assert density.shape == (2,)
    assert np.all(np.isfinite(density))
    assert np.all(density > 0.0)
    assert density[0] != density[1]


def test_dtm2000_rejects_altitude_below_120_km():
    with np.testing.assert_raises_regex(ValueError, "120 km"):
        density_from_inputs(1, 119_999.0, 0.0, 0.0, 0.0, 150.0, 150.0, 1.0, 1.0)


def test_interpolated_earth_orientation_matches_direct_transform():
    from astropy.time import Time, TimeDelta
    from spacecoords import celestial

    epoch = datetime(2024, 7, 4, tzinfo=timezone.utc)
    orientation = interpolated_earth_orientation(epoch, 0.0, 120.0, spacing_s=60.0)
    position = np.array([7.0e6, -1.0e6, 2.0e6])
    velocity = np.array([1200.0, 7300.0, -400.0])
    actual = np.concatenate(orientation.state_to_fixed(37.0, position[None], velocity[None]))
    expected = celestial.convert(
        Time(epoch) + TimeDelta(37.0, format="sec"),
        np.concatenate((position, velocity)),
        "GCRS",
        "ITRS",
    )
    np.testing.assert_allclose(actual.ravel()[:3], expected[:3], rtol=0.0, atol=5e-2)
    np.testing.assert_allclose(actual.ravel()[3:], expected[3:], rtol=0.0, atol=1e-3)


def test_interpolated_earth_orientation_preserves_rotation_geometry():
    epoch = datetime(2024, 7, 4, tzinfo=timezone.utc)
    orientation = interpolated_earth_orientation(epoch, 0.0, 120.0, spacing_s=30.0)
    rotation = orientation.rotation(47.0)
    vector = np.array([2.0, -3.0, 5.0])

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), rtol=0.0, atol=2e-10)
    fixed = orientation.position_to_fixed(47.0, vector)
    recovered = orientation.vector_to_inertial(47.0, fixed)
    np.testing.assert_allclose(recovered, vector, rtol=0.0, atol=2e-9)
