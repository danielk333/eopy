"""J2 perturbation example for an inclined low Earth orbit."""

import numpy as np
import pyorb

from eopy.constants import EARTH_MU
from eopy.gravity import GravityField
from eopy.integrators import Dop853Options
from eopy.propagator import propagate


def _angular_momentum_unit(state):
    h = np.cross(state[:3], state[3:])
    return h / np.linalg.norm(h)


def _orbital_elements_from_pyorb(states):
    kep = pyorb.cart_to_kep(states.T, mu=EARTH_MU, degrees=False)
    return {
        "semi_major_axis_m": kep[0],
        "eccentricity": kep[1],
        "inclination_rad": kep[2],
        "raan_rad": np.unwrap(kep[4]),
        "arg_perigee_rad": np.unwrap(kep[3]),
        "true_anomaly_rad": np.unwrap(kep[5]),
    }


def run():
    radius_m = 7_000_000.0
    inclination_rad = np.deg2rad(63.4)
    speed_mps = np.sqrt(EARTH_MU / radius_m)
    state0 = np.array(
        [
            radius_m,
            0.0,
            0.0,
            0.0,
            speed_mps * np.cos(inclination_rad),
            speed_mps * np.sin(inclination_rad),
        ]
    )
    period_s = 2.0 * np.pi * np.sqrt(radius_m**3 / EARTH_MU)
    t_j2, states_j2, _stats = propagate(
        state0,
        0.0,
        5.0 * period_s,
        gravity=GravityField(degree=2),
        options=Dop853Options(rtol=1e-10, atol=1e-4, max_step=120.0),
        return_history=True,
    )
    t_two_body, states_two_body, _stats = propagate(
        state0,
        0.0,
        5.0 * period_s,
        gravity=GravityField(degree=0, j2=0.0),
        options=Dop853Options(rtol=1e-10, atol=1e-4, max_step=120.0),
        return_history=True,
    )
    state_j2 = states_j2[-1]
    state_two_body = states_two_body[-1]
    normal_change = np.linalg.norm(
        _angular_momentum_unit(state_j2) - _angular_momentum_unit(state_two_body)
    )
    elements = _orbital_elements_from_pyorb(states_j2)
    return {
        "time_s": t_j2,
        "states": states_j2,
        "two_body_time_s": t_two_body,
        "two_body_states": states_two_body,
        "state": state_j2,
        "normal_change": normal_change,
        "elements": elements,
    }


if __name__ == "__main__":
    result = run()
    print(f"Angular momentum unit-vector change from two-body: {result['normal_change']:.6e}")

    import matplotlib.pyplot as plt

    states = result["states"]
    two_body_states = result["two_body_states"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(two_body_states[:, 0] / 1000.0, two_body_states[:, 1] / 1000.0, label="two-body")
    ax.plot(states[:, 0] / 1000.0, states[:, 1] / 1000.0, label="J2")
    ax.set_title("J2 perturbation over five orbits")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()

    elements = result["elements"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    time_orbits = result["time_s"] / result["time_s"][-1] * 5.0
    axes[0, 0].plot(
        time_orbits, (elements["semi_major_axis_m"] - elements["semi_major_axis_m"][0]) / 1000.0
    )
    axes[0, 0].set_ylabel("a - a0 [km]")
    axes[0, 1].plot(time_orbits, elements["eccentricity"])
    axes[0, 1].set_ylabel("e")
    axes[1, 0].plot(time_orbits, np.rad2deg(elements["inclination_rad"]))
    axes[1, 0].set_ylabel("i [deg]")
    axes[1, 1].plot(time_orbits, np.rad2deg(elements["raan_rad"]))
    axes[1, 1].set_ylabel("RAAN [deg]")
    axes[2, 0].plot(time_orbits, np.rad2deg(elements["arg_perigee_rad"]))
    axes[2, 0].set_ylabel("arg periapsis [deg]")
    axes[2, 0].set_xlabel("time [orbits]")
    axes[2, 1].plot(time_orbits, np.rad2deg(elements["true_anomaly_rad"]))
    axes[2, 1].set_ylabel("true anomaly [deg]")
    axes[2, 1].set_xlabel("time [orbits]")
    for axis in axes.flat:
        axis.grid(True)
    fig.suptitle("Osculating orbital elements under J2")
    plt.tight_layout()
    plt.show()
