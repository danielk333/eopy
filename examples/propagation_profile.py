"""Profile the cost of each enabled propagation effect."""

import numpy as np

from eopy.atmosphere import AtmosphereConfig, SolarActivity
from eopy.constants import ASTRONOMICAL_UNIT, EARTH_MU, EARTH_RADIUS, SUN_MU
from eopy.gravity import GravityField
from eopy.integrators import Dop853Options
from eopy.profiling import PropagationProfile
from eopy.propagator import SpacecraftProperties, propagate
from eopy.radiation import RadiationPressure
from eopy.third_body import ThirdBody


def run(duration_s=3600.0):
    """Run a representative LEO propagation and return its timing profile."""

    radius_m = EARTH_RADIUS + 500_000.0
    speed_mps = np.sqrt(EARTH_MU / radius_m)
    state0 = np.array([radius_m, 0.0, 0.0, 0.0, speed_mps, 0.0])
    sun_position = np.array([ASTRONOMICAL_UNIT, 0.0, 0.0])
    sun = ThirdBody(mu=SUN_MU, position_m=sun_position, name="Sun")
    profile = PropagationProfile()

    time_s, states, stats = propagate(
        state0,
        0.0,
        duration_s,
        gravity=GravityField(degree=2),
        spacecraft=SpacecraftProperties(
            mass_kg=500.0,
            drag_area_m2=4.0,
            drag_coefficient=2.2,
        ),
        atmosphere=AtmosphereConfig(model="dtm2000"),
        activity=SolarActivity(),
        radiation=RadiationPressure(
            mass_kg=500.0,
            area_m2=4.0,
            reflection_coefficient=1.3,
        ),
        sun_position=sun_position,
        third_bodies=[sun],
        options=Dop853Options(rtol=1e-10, atol=1e-3, max_step=60.0),
        return_history=True,
        profile=profile,
    )
    return {
        "time_s": time_s,
        "states": states,
        "stats": stats,
        "profile": profile,
        "report": profile.report(),
    }


if __name__ == "__main__":
    result = run()
    print(result["report"])
