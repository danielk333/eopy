"""Drag decay example with the lightweight DTM2000-like density model."""

import numpy as np

from eopy.atmosphere import AtmosphereConfig, SolarActivity
from eopy.constants import EARTH_MU, EARTH_RADIUS
from eopy.integrators import Dop853Options
from eopy.propagator import SpacecraftProperties, propagate


def _specific_energy(state):
    return 0.5 * np.dot(state[3:], state[3:]) - EARTH_MU / np.linalg.norm(state[:3])


def _specific_energy_history(states):
    return 0.5 * np.sum(states[:, 3:] * states[:, 3:], axis=1) - EARTH_MU / np.linalg.norm(states[:, :3], axis=1)


def run():
    radius_m = EARTH_RADIUS + 250_000.0
    speed_mps = np.sqrt(EARTH_MU / radius_m)
    state0 = np.array([radius_m, 0.0, 0.0, 0.0, speed_mps, 0.0])
    t, states, _stats = propagate(
        state0,
        0.0,
        1800.0,
        spacecraft=SpacecraftProperties(ballistic_coefficient_m2_per_kg=0.08),
        atmosphere=AtmosphereConfig(model="dtm2000", rho_ref=8e-10, h_ref_m=250_000.0, scale_height_m=70_000.0, co_rotate=False),
        activity=SolarActivity(f10=180.0, f10_mean=170.0, ap=12.0),
        options=Dop853Options(rtol=1e-10, atol=1e-4, max_step=30.0),
        return_history=True,
    )
    state = states[-1]
    return {"time_s": t, "states": states, "state": state, "energy_delta": _specific_energy(state) - _specific_energy(state0)}


if __name__ == "__main__":
    result = run()
    print(f"Specific orbital energy change: {result['energy_delta']:.6f} J/kg")

    import matplotlib.pyplot as plt

    t = result["time_s"]
    states = result["states"]
    altitude_km = (np.linalg.norm(states[:, :3], axis=1) - EARTH_RADIUS) / 1000.0
    energy = _specific_energy_history(states)
    energy_delta = energy - energy[0]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(t / 60.0, altitude_km)
    axes[0].set_ylabel("altitude [km]")
    axes[0].grid(True)
    axes[1].plot(t / 60.0, energy_delta)
    axes[1].set_xlabel("time [min]")
    axes[1].set_ylabel("energy change [J/kg]")
    axes[1].grid(True)
    fig.suptitle("Drag decay")
    plt.show()
