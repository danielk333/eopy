import numpy as np

from eopy.atmosphere import AtmosphereConfig, SolarActivity
from eopy.constants import ASTRONOMICAL_UNIT, EARTH_MU, EARTH_RADIUS, SUN_MU
from eopy.events import altitude_crossing_event, cylindrical_eclipse_event
from eopy.gravity import GravityField, j2_acceleration
from eopy.integrators import Dop853Options
from eopy.propagator import SpacecraftProperties, propagate
from eopy.relativity import schwarzschild_acceleration
from eopy.third_body import ThirdBody, third_body_acceleration


def _energy(state):
    r = np.linalg.norm(state[..., :3], axis=-1)
    v2 = np.sum(state[..., 3:] * state[..., 3:], axis=-1)
    return 0.5 * v2 - EARTH_MU / r


def _angular_momentum_norm(state):
    return np.linalg.norm(np.cross(state[..., :3], state[..., 3:]), axis=-1)


def test_two_body_circular_orbit_closes_after_one_period():
    radius = 7_000_000.0
    speed = np.sqrt(EARTH_MU / radius)
    period = 2.0 * np.pi * np.sqrt(radius**3 / EARTH_MU)
    state0 = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])

    _t, state, _stats = propagate(
        state0,
        0.0,
        period,
        gravity=GravityField(degree=0, j2=0.0),
        options=Dop853Options(rtol=1e-12, atol=1e-7, max_step=45.0),
    )

    np.testing.assert_allclose(state[:3], state0[:3], rtol=0.0, atol=2e-3)
    np.testing.assert_allclose(state[3:], state0[3:], rtol=0.0, atol=3e-6)


def test_two_body_energy_and_angular_momentum_are_conserved():
    radius = 7_200_000.0
    speed = np.sqrt(EARTH_MU / radius)
    period = 2.0 * np.pi * np.sqrt(radius**3 / EARTH_MU)
    state0 = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])

    _t, states, _stats = propagate(
        state0,
        0.0,
        4.0 * period,
        gravity=GravityField(degree=0, j2=0.0),
        options=Dop853Options(rtol=1e-11, atol=1e-6, max_step=60.0),
        return_history=True,
    )

    np.testing.assert_allclose(_energy(states), _energy(state0), rtol=0.0, atol=2e-5)
    np.testing.assert_allclose(
        _angular_momentum_norm(states),
        _angular_momentum_norm(state0),
        rtol=0.0,
        atol=0.2,
    )


def test_batch_propagation_matches_individual_propagation():
    radii = np.array([7_000_000.0, 7_010_000.0])
    states = np.column_stack(
        (
            radii,
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.sqrt(EARTH_MU / radii),
            np.zeros(2),
        )
    )

    _t, batch, _stats = propagate(
        states,
        0.0,
        900.0,
        gravity=GravityField(degree=0, j2=0.0),
        options=Dop853Options(rtol=1e-11, atol=1e-6, max_step=60.0),
    )
    singles = []
    for state in states:
        _t, out, _stats = propagate(
            state,
            0.0,
            900.0,
            gravity=GravityField(degree=0, j2=0.0),
            options=Dop853Options(rtol=1e-11, atol=1e-6, max_step=60.0),
        )
        singles.append(out)
    np.testing.assert_allclose(batch, np.asarray(singles), rtol=0.0, atol=2e-5)


def test_sample_times_return_one_continuous_solution():
    radius = 7_000_000.0
    state0 = np.array([radius, 0.0, 0.0, 0.0, np.sqrt(EARTH_MU / radius), 0.0])
    sample_times = np.array([0.0, 300.0, 600.0, 900.0])

    times, states, _stats = propagate(
        state0,
        0.0,
        900.0,
        gravity=GravityField(degree=0, j2=0.0),
        sample_times=sample_times,
        options=Dop853Options(rtol=1e-12, atol=1e-7, max_step=30.0),
    )
    _time, terminal, _stats = propagate(
        state0,
        0.0,
        900.0,
        gravity=GravityField(degree=0, j2=0.0),
        options=Dop853Options(rtol=1e-12, atol=1e-7, max_step=30.0),
    )

    np.testing.assert_array_equal(times, sample_times)
    np.testing.assert_allclose(states[0], state0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(states[-1], terminal, rtol=0.0, atol=1e-8)


def test_schwarzschild_acceleration_is_finite_for_circular_leo():
    radius = 7_000_000.0
    position = np.array([[radius, 0.0, 0.0]])
    velocity = np.array([[0.0, np.sqrt(EARTH_MU / radius), 0.0]])

    acceleration = schwarzschild_acceleration(position, velocity)

    assert acceleration.shape == position.shape
    assert acceleration[0, 0] > 0.0
    np.testing.assert_array_equal(acceleration[0, 1:], 0.0)


def test_j2_acceleration_matches_closed_form_reference_points():
    positions = np.array(
        [
            [7_000_000.0, 0.0, 0.0],
            [0.0, 0.0, 7_000_000.0],
            [5_000_000.0, 2_000_000.0, 4_000_000.0],
        ]
    )
    x, y, z = positions.T
    r2 = np.sum(positions * positions, axis=1)
    z2_r2 = z * z / r2
    factor = (
        1.5 * GravityField().j2 * EARTH_MU * EARTH_RADIUS**2 / (r2 * r2 * np.sqrt(r2))
    )
    expected = np.column_stack(
        (
            factor * x * (5.0 * z2_r2 - 1.0),
            factor * y * (5.0 * z2_r2 - 1.0),
            factor * z * (5.0 * z2_r2 - 3.0),
        )
    )

    np.testing.assert_allclose(
        j2_acceleration(positions), expected, rtol=1e-15, atol=0.0
    )


def test_drag_propagation_reduces_specific_orbital_energy():
    radius = EARTH_RADIUS + 260_000.0
    speed = np.sqrt(EARTH_MU / radius)
    state0 = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])

    _t, state, _stats = propagate(
        state0,
        0.0,
        900.0,
        spacecraft=SpacecraftProperties(ballistic_coefficient_m2_per_kg=0.06),
        atmosphere=AtmosphereConfig(
            model="exponential",
            rho_ref=5e-10,
            h_ref_m=260_000.0,
            scale_height_m=70_000.0,
            co_rotate=False,
        ),
        activity=SolarActivity(f10=180.0, f10_mean=170.0, ap=10.0),
        options=Dop853Options(rtol=1e-10, atol=1e-4, max_step=30.0),
    )

    assert _energy(state) < _energy(state0)


def test_third_body_acceleration_matches_differential_formula():
    position = np.array([[7_000_000.0, 1_000.0, 0.0]])
    body_position = np.array([ASTRONOMICAL_UNIT, 0.0, 0.0])
    sun = ThirdBody(mu=SUN_MU, position_m=body_position)

    acc = third_body_acceleration(position, 0.0, sun)

    sat_to_body = body_position - position
    expected = SUN_MU * (
        sat_to_body / np.linalg.norm(sat_to_body, axis=1)[:, None] ** 3
        - body_position / np.linalg.norm(body_position) ** 3
    )
    np.testing.assert_allclose(acc, expected, rtol=1e-15, atol=0.0)


def test_altitude_event_terminates_propagation():
    radius = EARTH_RADIUS + 200_000.0
    state0 = np.array([radius, 0.0, 0.0, -100.0, 0.0, 0.0])

    t, state, stats = propagate(
        state0,
        0.0,
        4000.0,
        gravity=GravityField(mu=0.0, degree=0, j2=0.0),
        options=Dop853Options(rtol=1e-12, atol=1e-9, max_step=10.0),
        events=altitude_crossing_event(150_000.0, direction=-1.0),
    )

    assert np.isclose(t, 500.0, atol=1e-8)
    assert np.isclose(np.linalg.norm(state[:3]) - EARTH_RADIUS, 150_000.0, atol=1e-6)
    assert stats.event_times is not None
    np.testing.assert_allclose(stats.event_times[0], [500.0], rtol=0.0, atol=1e-8)


def test_cylindrical_eclipse_event_detects_umbra_boundary():
    radius = 7_200_000.0
    speed = np.sqrt(EARTH_MU / radius)
    state0 = np.array([0.0, radius, 0.0, -speed, 0.0, 0.0])
    sun = np.array([ASTRONOMICAL_UNIT, 0.0, 0.0])

    _t, _state, stats = propagate(
        state0,
        0.0,
        2500.0,
        gravity=GravityField(degree=0, j2=0.0),
        options=Dop853Options(rtol=1e-10, atol=1e-4, max_step=20.0),
        events=cylindrical_eclipse_event(sun, direction=-1.0),
    )

    assert stats.event_times is not None
    assert stats.event_times[0].size == 1
