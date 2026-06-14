"""Two-body circular orbit propagation example."""

import numpy as np

from eopy.constants import EARTH_MU
from eopy.gravity import GravityField
from eopy.integrators import Dop853Options
from eopy.propagator import propagate


def run():
    radius_m = 7_000_000.0
    speed_mps = np.sqrt(EARTH_MU / radius_m)
    period_s = 2.0 * np.pi * np.sqrt(radius_m**3 / EARTH_MU)
    state0 = np.array([radius_m, 0.0, 0.0, 0.0, speed_mps, 0.0])

    t, states, stats = propagate(
        state0,
        0.0,
        period_s,
        gravity=GravityField(degree=0, j2=0.0),
        options=Dop853Options(rtol=1e-11, atol=1e-6, max_step=60.0),
        return_history=True,
    )
    return {"time_s": t, "states": states, "final_error_m": np.linalg.norm(states[-1, :3] - state0[:3]), "stats": stats}


if __name__ == "__main__":
    result = run()
    print(f"Final position closure error: {result['final_error_m']:.6f} m")

    import matplotlib.pyplot as plt

    states = result["states"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(states[:, 0] / 1000.0, states[:, 1] / 1000.0)
    ax.scatter([states[0, 0] / 1000.0], [states[0, 1] / 1000.0], label="start")
    ax.set_title("Two-body circular orbit")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    plt.show()
