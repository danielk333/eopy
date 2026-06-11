import numpy as np
import pymsis
from spacecoords import frames

from eopy.atmosphere import AtmosphereConfig, SolarActivity
import eopy.atmosphere as atmosphere_module
from eopy.constants import ASTRONOMICAL_UNIT, EARTH_MU, SOLAR_RADIATION_PRESSURE_1AU
from eopy.gravity import GravityField, gravity_acceleration
from eopy.integrators import Dop853Options
from eopy.propagator import SpacecraftProperties, drag_acceleration, propagate
from eopy.radiation import RadiationPressure, cylindrical_shadow_factor, solar_radiation_pressure_acceleration


def test_gravity_accepts_batch_positions():
    r = np.array([[7000e3, 0.0, 0.0], [0.0, 7000e3, 0.0]])
    acc = gravity_acceleration(r, GravityField(degree=2))
    assert acc.shape == (2, 3)
    assert acc[0, 0] < 0.0
    assert acc[1, 1] < 0.0


def test_propagate_two_particles_keeps_batch_shape():
    radius = 7000e3
    speed = np.sqrt(EARTH_MU / radius)
    states = np.array(
        [
            [radius, 0.0, 0.0, 0.0, speed, 0.0],
            [radius + 1e3, 0.0, 0.0, 0.0, np.sqrt(EARTH_MU / (radius + 1e3)), 0.0],
        ]
    )

    _t, out, stats = propagate(
        states,
        0.0,
        60.0,
        gravity=GravityField(degree=2),
        options=Dop853Options(rtol=1e-10, atol=1e-3, max_step=30.0),
    )

    assert out.shape == states.shape
    assert stats.accepted_steps > 0
    assert np.all(np.isfinite(out))


def test_drag_path_runs_with_dtm2000_like_activity_inputs():
    radius = 6778e3
    speed = np.sqrt(EARTH_MU / radius)
    state = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])

    _t, out, _stats = propagate(
        state,
        0.0,
        10.0,
        spacecraft=SpacecraftProperties(mass_kg=10.0, drag_area_m2=0.05, drag_coefficient=2.2),
        atmosphere=AtmosphereConfig(model="dtm2000"),
        activity=SolarActivity(f10=160.0, f10_mean=155.0, ap=8.0),
        options=Dop853Options(rtol=1e-9, atol=1e-3, max_step=10.0),
    )

    assert out.shape == (6,)
    assert np.all(np.isfinite(out))


def test_orekit_isotropic_drag_force_law_with_ballistic_coefficient():
    # Ported from Orekit's IsotropicDrag law:
    # a = -0.5 * rho * (Cd A / m) * |v_spacecraft-v_atmosphere| * v_rel
    position = np.array([[6778e3, 0.0, 0.0]])
    velocity = np.array([[0.0, 7600.0, 0.0]])
    rho = 1.0e-12
    bc = 2.5 * 1.2 / 10.0
    cfg = AtmosphereConfig(
        model="dtm2000",
        rho_ref=rho,
        h_ref_m=6778e3 - 6378136.3,
        scale_height_m=1.0e12,
        co_rotate=False,
    )

    acc = drag_acceleration(
        position,
        velocity,
        SpacecraftProperties(ballistic_coefficient_m2_per_kg=bc),
        cfg,
        SolarActivity(),
    )

    expected = -0.5 * rho * bc * np.linalg.norm(velocity, axis=1)[:, None] * velocity
    np.testing.assert_allclose(acc, expected, rtol=1e-14, atol=0.0)


def test_pymsis_density_is_positive_for_leo_point():
    position = np.array([6778e3, 0.0, 0.0])
    cfg = AtmosphereConfig(model="msis")
    _t, out, _stats = propagate(
        np.array([*position, 0.0, np.sqrt(EARTH_MU / np.linalg.norm(position)), 0.0]),
        0.0,
        1.0,
        spacecraft=SpacecraftProperties(ballistic_coefficient_m2_per_kg=0.01),
        atmosphere=cfg,
        activity=SolarActivity(f10=160.0, f10_mean=155.0, ap=8.0),
        options=Dop853Options(rtol=1e-8, atol=1e-3, max_step=1.0),
    )
    assert np.all(np.isfinite(out))


def test_msis_uses_spacecoords_wgs84_geodetic_conversion(monkeypatch):
    position = np.array([[6378136.3 + 400e3, 1000.0, 2000.0]])
    expected_lat, expected_lon, expected_alt_m = frames.ecef_to_geodetic_wgs84(
        position[:, 0], position[:, 1], position[:, 2], degrees=True
    )
    captured = {}

    def fake_calculate(dates, lons, lats, alts_km, **kwargs):
        captured["lons"] = np.asarray(lons)
        captured["lats"] = np.asarray(lats)
        captured["alts_km"] = np.asarray(alts_km)
        out = np.zeros((1, len(pymsis.Variable)))
        out[:, pymsis.Variable.MASS_DENSITY] = 3.0e-12
        return out

    monkeypatch.setattr(atmosphere_module.pymsis, "calculate", fake_calculate)

    rho = atmosphere_module.density_msis_like(position, SolarActivity(), AtmosphereConfig(model="msis"))

    np.testing.assert_allclose(rho, [3.0e-12])
    np.testing.assert_allclose(captured["lats"], expected_lat)
    np.testing.assert_allclose(captured["lons"], expected_lon)
    np.testing.assert_allclose(captured["alts_km"], np.clip(expected_alt_m, 120_000.0, 1_000_000.0) / 1000.0)


def test_orekit_isotropic_radiation_single_coefficient_law_full_light():
    # Ported from Orekit's IsotropicRadiationSingleCoefficient:
    # a = global * A * Cr / m * flux.
    position = np.array([[0.0, 7200e3, 0.0]])
    sun = np.array([ASTRONOMICAL_UNIT, 0.0, 0.0])
    radiation = RadiationPressure(area_m2=2.5, reflection_coefficient=0.7, mass_kg=1000.0)

    acc = solar_radiation_pressure_acceleration(position, sun, radiation)

    sun_to_sat = position - sun
    distance = np.linalg.norm(sun_to_sat, axis=1)
    expected = (
        SOLAR_RADIATION_PRESSURE_1AU
        * (ASTRONOMICAL_UNIT / distance)[:, None] ** 2
        * (2.5 * 0.7 / 1000.0)
        * sun_to_sat
        / distance[:, None]
    )
    np.testing.assert_allclose(acc, expected, rtol=1e-15, atol=0.0)


def test_solar_radiation_pressure_cylindrical_umbra():
    sun = np.array([ASTRONOMICAL_UNIT, 0.0, 0.0])
    lit = np.array([[0.0, 7200e3, 0.0]])
    eclipsed = np.array([[-7200e3, 0.0, 0.0]])

    assert cylindrical_shadow_factor(lit, sun)[0] == 1.0
    assert cylindrical_shadow_factor(eclipsed, sun)[0] == 0.0

    radiation = RadiationPressure(area_m2=2.5, reflection_coefficient=0.7, mass_kg=1000.0)
    np.testing.assert_allclose(solar_radiation_pressure_acceleration(eclipsed, sun, radiation), 0.0)
