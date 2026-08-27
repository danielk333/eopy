"""Compare eopy propagation with SGP4 at two orbital altitudes.

SGP4 returns TEME coordinates in km and km/s.  This example uses
``spacecoords`` to transform those states to GCRS (an inertial frame) before
initializing eopy and computing position differences.
"""

import numpy as np
from astropy.time import Time
from sgp4.api import SGP4_ERRORS, Satrec
from spacecoords import celestial

from eopy.gravity import GravityField
from eopy.integrators import Dop853Options
from eopy.propagator import propagate


# The Vanguard 1 TLE is from the python-sgp4 documentation.  The second is a
# representative near-circular orbit at approximately 800 km altitude, where
# drag is much weaker.  Fixed TLEs make the comparison reproducible.
TLES = {
    "Vanguard 1 (elliptical LEO)": (
        "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
        "2 00005  34.2682 331.5174 1849677 331.7664  19.3264 10.82419157413667",
    ),
    "Circular orbit (about 800 km)": (
        "1 99999U 24001A   24150.50000000  .00000000  00000-0  00000-0 0  9990",
        "2 99999  98.6000 120.0000 0010000  90.0000 270.0000 14.27500000    01",
    ),
}


def _sgp4_states_gcrs(satellite, offsets_s):
    jd = np.full(offsets_s.shape, satellite.jdsatepoch, dtype=float)
    fr = satellite.jdsatepochF + offsets_s / 86_400.0
    errors, positions_km, velocities_km_s = satellite.sgp4_array(jd, fr)
    if np.any(errors):
        messages = sorted({SGP4_ERRORS[int(code)] for code in errors if code})
        raise RuntimeError("SGP4 propagation failed: " + "; ".join(messages))

    states_teme = np.vstack((positions_km.T * 1_000.0, velocities_km_s.T * 1_000.0))
    # Pass the two-part Julian date separately to retain SGP4's epoch precision.
    epochs = Time(jd, fr, format="jd", scale="utc")
    return celestial.convert(epochs, states_teme, "TEME", "GCRS").T


def _compare_tle(tle, offsets_s, step_s):
    satellite = Satrec.twoline2rv(*tle)
    sgp4_states = _sgp4_states_gcrs(satellite, offsets_s)

    eopy_states = np.empty_like(sgp4_states)
    eopy_states[0] = sgp4_states[0]
    options = Dop853Options(rtol=1e-10, atol=1e-3, max_step=min(120.0, step_s))
    for index in range(1, offsets_s.size):
        _time, eopy_states[index], _stats = propagate(
            eopy_states[index - 1],
            offsets_s[index - 1],
            offsets_s[index],
            gravity=GravityField(degree=2),
            options=options,
        )

    position_difference_m = np.linalg.norm(eopy_states[:, :3] - sgp4_states[:, :3], axis=1)
    return {
        "time_s": offsets_s,
        "sgp4_states_gcrs": sgp4_states,
        "eopy_states_gcrs": eopy_states,
        "position_difference_m": position_difference_m,
    }


def run(duration_s=86_400.0, step_s=600.0, tles=TLES):
    """Compare every supplied TLE using independently matched initial states."""

    if duration_s <= 0.0 or step_s <= 0.0:
        raise ValueError("duration_s and step_s must be positive")
    offsets_s = np.arange(0.0, duration_s + 0.5 * step_s, step_s)
    return {name: _compare_tle(tle, offsets_s, step_s) for name, tle in tles.items()}


if __name__ == "__main__":
    results = run()
    for name, result in results.items():
        difference_km = result["position_difference_m"][-1] / 1_000.0
        print(f"{name} final position difference: {difference_km:.3f} km")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, result in results.items():
        ax.semilogy(
            result["time_s"] / 3_600.0,
            result["position_difference_m"] / 1_000.0,
            label=name,
        )
    ax.set(title="eopy J2 propagation versus SGP4", xlabel="Time since TLE epoch [h]", ylabel="Position difference [km]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
