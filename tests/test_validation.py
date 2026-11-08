"""validation tests"""

import numpy as np
import pytest
from scipy.special import eval_legendre

from eopy import native
from eopy.constants import EARTH_MU, EARTH_RADIUS
from eopy.gravity import GravityField, gravity_acceleration
from eopy.integrators import Dop853Options, integrate_dop853
from eopy.radiation import conical_shadow_factor
from eopy.relativity import SPEED_OF_LIGHT_MPS, schwarzschild_acceleration


def test_degree_zero_suppresses_default_j2_in_all_backends():
    r = np.array([7e6, 1e6, 2e6])
    expected = -EARTH_MU * r / np.linalg.norm(r) ** 3
    for backend in ("python", "auto"):
        np.testing.assert_allclose(
            gravity_acceleration(r, GravityField(degree=0, backend=backend)), expected, rtol=2e-15
        )


def test_tesseral_c22_matches_cartesian_potential_derivative():
    # Real 4pi normalization: P22 = sqrt(15)/2 cos(latitude)^2.
    # Wieczorek & Meschede (2018), doi:10.1029/2018GC007529.
    c = np.zeros((3, 3))
    c[0, 0] = 1
    c[2, 2] = 2e-6
    r = np.array([7e6, 2e6, 3e6])
    radius = np.linalg.norm(r)
    k = EARTH_MU * EARTH_RADIUS**2 * np.sqrt(15) / 2 * c[2, 2]
    expected = -EARTH_MU * r / radius**3 + k * (
        np.array([2 * r[0], -2 * r[1], 0]) / radius**5 - 5 * (r[0] ** 2 - r[1] ** 2) * r / radius**7
    )
    for backend, tolerance in [("python", 1e-8), ("auto", 1e-8)]:
        field = GravityField(
            c=c, s=np.zeros_like(c), normalized=True, degree=2, order=2, backend=backend
        )
        np.testing.assert_allclose(gravity_acceleration(r, field), expected, rtol=tolerance)


def test_normalized_degree_360_zonal_at_pole():
    if not native.available():
        pytest.skip("native library required")
    n = 360
    c = np.zeros((n + 1, n + 1))
    c[0, 0] = 1
    c[n, 0] = 1e-4
    r = EARTH_RADIUS + 1000.0
    # P_n(1)=1, so the normalized zonal potential is explicit on the axis.
    expected = (
        -EARTH_MU
        / r**2
        * (
            1
            + (n + 1) * (EARTH_RADIUS / r) ** n * np.sqrt(2 * n + 1) * c[n, 0] * eval_legendre(n, 1)
        )
    )
    out = native.spherical_harmonic_acceleration(
        np.array([0.0, 0.0, r]), n, n, c, None, EARTH_MU, EARTH_RADIUS, True
    )
    np.testing.assert_allclose(out, [0, 0, expected], atol=1e-8)
    # Populated sectorals used to overflow even when their physical contribution is tiny.
    c[n, n] = 1e-8
    assert np.all(
        np.isfinite(
            native.spherical_harmonic_acceleration(
                np.array([r, 1e6, 2e6]), n, n, c, None, EARTH_MU, EARTH_RADIUS, True
            )
        )
    )


def test_native_rejects_buffers_that_cannot_be_read_safely():
    with pytest.raises(ValueError):
        native.central_j2_acceleration(np.zeros((2, 2, 3)), EARTH_MU, EARTH_RADIUS, 0)
    with pytest.raises(ValueError):
        native.spherical_harmonic_acceleration(
            np.ones(3), -1, 0, np.ones((1, 1)), None, EARTH_MU, EARTH_RADIUS, True
        )


def test_backward_samples_and_zero_duration():
    times, states, stats = integrate_dop853(
        lambda t, y: y,
        1.0,
        np.array([np.e]),
        0.0,
        sample_times=np.array([1.0, 0.5, 0.0]),
        options=Dop853Options(atol=1e-12),
    )
    np.testing.assert_allclose(states[:, 0], np.exp(times), rtol=1e-8)
    assert stats.accepted_steps is None and stats.rejected_steps is None
    times, states, _ = integrate_dop853(
        lambda t, y: y, 2.0, np.array([3.0]), 2.0, sample_times=np.array([2.0])
    )
    np.testing.assert_array_equal(states, [[3.0]])


def test_terminal_event_before_first_sample_returns_empty_history():
    def event(t, y):
        return t - 0.25

    event.terminal = True
    times, states, stats = integrate_dop853(
        lambda t, y: np.ones_like(y),
        0.0,
        np.zeros(1),
        1.0,
        events=event,
        sample_times=np.array([0.5, 1.0]),
    )
    assert times.shape == (0,) and states.shape == (0, 1)
    np.testing.assert_allclose(stats.event_times[0], [0.25])


def test_circular_schwarzschild_limit():
    # First post-Newtonian circular orbit: outward correction 3 mu^2/(c^2 r^3).
    # IERS Conventions (2010), chapter 10, relativistic satellite equations.
    r = 7e6
    out = schwarzschild_acceleration(np.array([r, 0, 0]), np.array([0, np.sqrt(EARTH_MU / r), 0]))
    np.testing.assert_allclose(
        out, [3 * EARTH_MU**2 / (SPEED_OF_LIGHT_MPS**2 * r**3), 0, 0], rtol=2e-15
    )


def test_annular_shadow_remains_finite():
    # Far behind Earth its apparent disc can be smaller than the Sun.
    factor = conical_shadow_factor(np.array([-2e9, 0, 0]), np.array([1.5e11, 0, 0]))
    assert 0 < factor < 1


def test_j2_secular_node_rate():
    # First-order orbit-averaged J2 node drift (Vallado, Fundamentals of
    # Astrodynamics and Applications, secular zonal perturbations).
    from eopy import EARTH_J2, propagate

    radius, inclination = 7e6, np.deg2rad(63.4)
    speed = np.sqrt(EARTH_MU / radius)
    period = 2 * np.pi * np.sqrt(radius**3 / EARTH_MU)
    state = np.array([radius, 0, 0, 0, speed * np.cos(inclination), speed * np.sin(inclination)])
    times = np.linspace(0, 10 * period, 1001)
    _, history, _ = propagate(
        state,
        0,
        times[-1],
        gravity=GravityField(degree=2),
        sample_times=times,
        options=Dop853Options(rtol=1e-11, atol=1e-6, max_step=60),
    )
    h = np.cross(history[:, :3], history[:, 3:])
    nodes = np.unwrap(np.arctan2(h[:, 0], -h[:, 1]))
    fitted_rate = np.polyfit(times, nodes, 1)[0]
    mean_motion = np.sqrt(EARTH_MU / radius**3)
    expected = -1.5 * mean_motion * EARTH_J2 * (EARTH_RADIUS / radius) ** 2 * np.cos(inclination)
    # First-order theory omits J2^2 and short-period osculating corrections.
    np.testing.assert_allclose(fitted_rate, expected, rtol=0.01)


def test_srp_illuminated_inverse_square_limit():
    from eopy import RadiationPressure, solar_radiation_pressure_acceleration
    from eopy.constants import ASTRONOMICAL_UNIT, SOLAR_RADIATION_PRESSURE_1AU

    r = np.array([7e6, 0, 0])
    cfg = RadiationPressure(area_m2=2, mass_kg=4, reflection_coefficient=1.5, shadow_model="none")
    accelerations = []
    for distance in [ASTRONOMICAL_UNIT, 2 * ASTRONOMICAL_UNIT]:
        accelerations.append(
            solar_radiation_pressure_acceleration(r, r + np.array([distance, 0, 0]), cfg)
        )
    np.testing.assert_allclose(
        accelerations[0], [-SOLAR_RADIATION_PRESSURE_1AU * 0.75, 0, 0], rtol=1e-14
    )
    np.testing.assert_allclose(accelerations[1], accelerations[0] / 4, rtol=1e-14)


def test_single_propagation_event_states_preserve_public_shape():
    from eopy import propagate

    def event(t, state):
        return t - 0.25

    event.terminal = True
    _, _, stats = propagate(
        np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        0.0,
        1.0,
        events=event,
        sample_times=np.array([0.5, 1.0]),
    )
    assert stats.event_states[0].shape == (1, 6)
