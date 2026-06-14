"""Solar radiation pressure example with cylindrical Earth shadowing."""

import numpy as np

from eopy.constants import ASTRONOMICAL_UNIT, EARTH_MU
from eopy.gravity import GravityField
from eopy.integrators import Dop853Options
from eopy.propagator import propagate
from eopy.radiation import RadiationPressure, cylindrical_shadow_factor


def run():
    radius_m = 7_200_000.0
    speed_mps = np.sqrt(EARTH_MU / radius_m)
    state0 = np.array([0.0, radius_m, 0.0, -speed_mps, 0.0, 0.0])
    sun = np.array([ASTRONOMICAL_UNIT, 0.0, 0.0])
    t, states, _stats = propagate(
        state0,
        0.0,
        600.0,
        gravity=GravityField(degree=0, j2=0.0),
        radiation=RadiationPressure(area_m2=20.0, reflection_coefficient=1.4, mass_kg=500.0),
        sun_position=sun,
        options=Dop853Options(rtol=1e-10, atol=1e-4, max_step=30.0),
        return_history=True,
    )
    shadow = cylindrical_shadow_factor(states[:, :3], sun)
    return {"time_s": t, "states": states, "state": states[-1], "shadow": shadow, "initial_shadow": shadow[0]}


if __name__ == "__main__":
    result = run()
    print(f"Initial shadow factor: {result['initial_shadow']:.1f}")

    import matplotlib.pyplot as plt

    t = result["time_s"]
    states = result["states"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=False)
    axes[0].plot(states[:, 0] / 1000.0, states[:, 1] / 1000.0)
    axes[0].set_title("SRP propagation")
    axes[0].set_xlabel("x [km]")
    axes[0].set_ylabel("y [km]")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True)
    axes[1].step(t / 60.0, result["shadow"], where="post")
    axes[1].set_xlabel("time [min]")
    axes[1].set_ylabel("shadow factor")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()
