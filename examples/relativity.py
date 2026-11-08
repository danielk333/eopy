"""Compare a circular LEO with and without the Schwarzschild correction.

Both runs use the same initial state and central gravity, isolating the effect
of the first post-Newtonian correction over one day.
"""

import numpy as np

from eopy import EARTH_MU, Dop853Options, GravityField, propagate, schwarzschild_acceleration


def run(duration_s: float = 86_400.0) -> dict[str, np.ndarray | float]:
    radius_m = 6_378_137.0 + 400e3
    speed_mps = np.sqrt(EARTH_MU / radius_m)
    state0 = np.array([radius_m, 0.0, 0.0, 0.0, speed_mps, 0.0])
    sample_times = np.linspace(0.0, duration_s, 361)
    options = Dop853Options(rtol=1e-12, atol=1e-7, max_step=60.0)
    histories = []
    for relativity in (False, True):
        _, states, _ = propagate(
            state0,
            0.0,
            duration_s,
            gravity=GravityField(degree=0),
            options=options,
            sample_times=sample_times,
            relativity=relativity,
        )
        histories.append(states)
    newtonian, relativistic = histories
    position_delta = np.linalg.norm(relativistic[:, :3] - newtonian[:, :3], axis=1)
    return {
        "time_s": sample_times,
        "newtonian_states": newtonian,
        "relativistic_states": relativistic,
        "position_delta_history_m": position_delta,
        "position_delta_m": float(position_delta[-1]),
        "initial_correction_m_s2": float(schwarzschild_acceleration(state0[:3], state0[3:])[0]),
    }


if __name__ == "__main__":
    result = run()
    print(
        f"Initial outward relativistic acceleration: {result['initial_correction_m_s2']:.6g} m/s²"
    )
    print(f"Position difference after one day: {result['position_delta_m']:.6f} m")
