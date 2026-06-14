"""Third-body Sun perturbation example with a fixed Sun vector."""

import numpy as np

from eopy.constants import ASTRONOMICAL_UNIT, EARTH_MU, SUN_MU
from eopy.gravity import GravityField
from eopy.integrators import Dop853Options
from eopy.propagator import propagate
from eopy.third_body import ThirdBody


def run():
    radius_m = 42_164_000.0
    speed_mps = np.sqrt(EARTH_MU / radius_m)
    state0 = np.array([radius_m, 0.0, 0.0, 0.0, speed_mps, 0.0])
    sun = ThirdBody(mu=SUN_MU, position_m=np.array([ASTRONOMICAL_UNIT, 0.0, 0.0]), name="sun")
    t, states_with_sun, _stats = propagate(
        state0,
        0.0,
        3600.0,
        gravity=GravityField(degree=0, j2=0.0),
        third_bodies=[sun],
        options=Dop853Options(rtol=1e-11, atol=1e-4, max_step=60.0),
        return_history=True,
    )
    t_two_body, states_two_body, _stats = propagate(
        state0,
        0.0,
        3600.0,
        gravity=GravityField(degree=0, j2=0.0),
        options=Dop853Options(rtol=1e-11, atol=1e-4, max_step=60.0),
        return_history=True,
    )
    position_delta = np.linalg.norm(states_with_sun[:, :3] - states_two_body[:, :3], axis=1)
    return {
        "time_s": t,
        "states": states_with_sun,
        "two_body_time_s": t_two_body,
        "two_body_states": states_two_body,
        "position_delta_history_m": position_delta,
        "state": states_with_sun[-1],
        "position_delta_m": position_delta[-1],
    }


if __name__ == "__main__":
    result = run()
    print(f"Third-body position delta after one hour: {result['position_delta_m']:.6f} m")

    import matplotlib.pyplot as plt

    t = result["time_s"]
    states = result["states"]
    two_body_states = result["two_body_states"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 7))
    axes[0].plot(two_body_states[:, 0] / 1000.0, two_body_states[:, 1] / 1000.0, label="two-body")
    axes[0].plot(states[:, 0] / 1000.0, states[:, 1] / 1000.0, label="with Sun")
    axes[0].set_title("Fixed-Sun third-body perturbation")
    axes[0].set_xlabel("x [km]")
    axes[0].set_ylabel("y [km]")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(t / 60.0, result["position_delta_history_m"])
    axes[1].set_xlabel("time [min]")
    axes[1].set_ylabel("position delta [m]")
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()
