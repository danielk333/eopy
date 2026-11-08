"""Compare exponential, MSIS, and DTM2000 atmosphere densities."""

from datetime import datetime, timezone

import numpy as np

from eopy import (
    AtmosphereConfig,
    SolarActivity,
    density_dtm2000,
    density_exponential,
    density_msis,
)


def run() -> dict[str, float]:
    position = np.array([6_378_137.0 + 400e3, 0.0, 0.0])
    sun_fixed = np.array([149_597_870_700.0, 0.0, 0.0])
    epoch = datetime(2024, 7, 4, 12, tzinfo=timezone.utc)
    activity = SolarActivity()
    config = AtmosphereConfig(epoch=epoch)
    return {
        "exponential_kg_m3": float(density_exponential(position, activity, config)),
        "msis_kg_m3": float(density_msis(position, activity, config)),
        "dtm2000_kg_m3": float(density_dtm2000(position, activity, config, 0.0, sun_fixed)),
    }


if __name__ == "__main__":
    for name, value in run().items():
        print(f"{name}: {value:.6g}")
