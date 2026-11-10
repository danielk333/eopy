"""Predict GOCE's descending altitude crossing from ESA's final GPS-based orbit.

The comparison is a propagation to a specified geocentric altitude.
Default: 140 km, within the published orbit coverage.
Independent runs use ESA initial states at +0, +5, and +10 hours.

Orbit source (SP3 positions/velocities in J2000 axes, epochs in GPS time):
https://navigation-office.esa.int/GOCE_reentry_orbits.html
https://navigation-office.esa.int/products/goce/goce_13314.sp3

Pardini & Anselmo, GOCE Reentry Predictions for the Italian Civil Protection
Authorities, ESA SP-728 (2015): dry mass 1002 kg, campaign mean Cd*A 3.663 m^2.
https://openportal.isti.cnr.it/data/2015/342652/2015_%20342652.published.pdf

These give an illustrative constant Cd*A/m, not the true time-varying value
or a value calibrated for MSIS 2.1. The campaign mean includes later observations.

Example run:
```
python examples/reentry_prediction.py \
  --orbit-file ~/data/eopy/goce_13314.sp3 \
  --space-weather-file ~/data/eopy/SW-All.csv \
  --gravity-model-file ~/data/eopy/EIGEN-6S4_v2.gfc \
  --altitude-km 140 \
  --start-hours 0 5 10 \
  --ground-track-length 1800 \
  --plot
```
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import erfa
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq
from spacecoords import celestial

from eopy import (
    EARTH_RADIUS,
    AtmosphereConfig,
    CelestrakSpaceWeather,
    Dop853Options,
    GravityField,
    SpacecraftProperties,
    interpolated_earth_orientation,
    propagate,
    read_icgem_gfc,
)
from eopy.data import download_file
from eopy.events import altitude_crossing_event
from eopy.frames import ecef_to_geodetic_wgs84
from eopy.types import NDArray_NxM

if TYPE_CHECKING:
    from matplotlib.figure import Figure

ORBIT_URL = "https://navigation-office.esa.int/products/goce/goce_13314.sp3"
DEFAULT_BALLISTIC_COEFFICIENT = 3.663 / 1002.0


def read_goce_orbit(path: Path) -> tuple[datetime, NDArray_NxM, NDArray_NxM]:
    """Read this ESA SP3 product, returning UTC epoch, elapsed seconds, GCRS states.

    SP3-c positions are km; velocity records are decimetres/s. GPS = TAI - 19 s.
    J2000 denotes mean dynamical J2000 axes; ERFA's frame-bias matrix transforms
    GCRS to those axes, so right-multiplication converts row vectors back to GCRS.
    References: https://files.igs.org/pub/data/format/sp3c.txt;
    ERFA eraBp00 / IAU SOFA iauBp00.
    """
    lines = path.read_text().splitlines()
    if not lines[0].startswith("#cV") or "J2000" not in lines[0]:
        raise ValueError("Expected ESA's velocity-bearing J2000 SP3-c orbit")
    if not any(line.startswith("%c") and "GPS" in line.split() for line in lines):
        raise ValueError("Expected GPS time in the SP3 header")
    stamps, positions, velocities = [], [], []
    for line in lines:
        if line.startswith("*"):
            year, month, day, hour, minute, seconds = line[1:].split()
            stamps.append(
                datetime(int(year), int(month), int(day), int(hour), int(minute))
                + timedelta(seconds=float(seconds))
            )
        elif line.startswith("PL15"):
            positions.append([float(line[4:18]), float(line[18:32]), float(line[32:46])])
        elif line.startswith("VL15"):
            velocities.append([float(line[4:18]), float(line[18:32]), float(line[32:46])])
    if len(stamps) != len(positions) or len(stamps) != len(velocities) or len(stamps) < 2:
        raise ValueError("Expected one GOCE position and velocity per SP3 epoch")
    gps_seconds = (stamps[0] - datetime(1980, 1, 6)).total_seconds()
    epoch = Time(gps_seconds, format="gps").utc.to_datetime(timezone=timezone.utc)
    elapsed = np.array([(stamp - stamps[0]).total_seconds() for stamp in stamps])
    bias, _, _ = erfa.bp00(2451545.0, 0.0)
    states = np.column_stack(
        (np.asarray(positions) * 1000 @ bias, np.asarray(velocities) * 0.1 @ bias)
    )
    return epoch, elapsed, states


def reference_crossing(
    times_s: NDArray_NxM, states: NDArray_NxM, altitude_m: float
) -> tuple[float, tuple[float, float]] | None:
    """Interpolate the first descending crossing bracketed by reference samples.

    Radius and radial velocity define a local cubic Hermite interpolant. Return
    the sample bracket as well; interpolation does not supply measurement accuracy.
    """
    radius = np.linalg.norm(states[:, :3], axis=1)
    radial_velocity = np.sum(states[:, :3] * states[:, 3:], axis=1) / radius
    target = EARTH_RADIUS + altitude_m
    brackets = np.flatnonzero((radius[:-1] > target) & (radius[1:] <= target))
    if not len(brackets):
        return None
    index = brackets[0]
    segment = slice(index, index + 2)
    spline = CubicHermiteSpline(times_s[segment], radius[segment], radial_velocity[segment])
    crossing = brentq(lambda t: float(spline(t) - target), times_s[index], times_s[index + 1])
    return crossing, (float(times_s[index]), float(times_s[index + 1]))


def run(
    orbit_file: Path,
    space_weather_file: Path,
    cutoff_altitude_m: float = 140_000.0,
    ballistic_coefficient_m2_per_kg: float = DEFAULT_BALLISTIC_COEFFICIENT,
    gravity_model_file: Path | None = None,
    max_hours: float = 36.0,
    start_hours: float = 0.0,
) -> dict:
    """Predict from an ESA state at start_hours; elapsed times retain the file origin."""
    if cutoff_altitude_m < 120_000:
        raise ValueError("This example is limited to cutoffs at or above 120 km")
    epoch, reference_times, reference_states = read_goce_orbit(orbit_file)
    start_s = start_hours * 3600.0
    if not reference_times[0] <= start_s < reference_times[-1]:
        raise ValueError("Start time must lie within the published orbit coverage")
    reference = reference_crossing(reference_times, reference_states, cutoff_altitude_m)
    if reference is not None and start_s >= reference[0]:
        raise ValueError("Start time must precede the reference's first descending crossing")
    spline = CubicHermiteSpline(
        reference_times, reference_states[:, :3], reference_states[:, 3:], axis=0
    )
    initial_state = np.concatenate((spline(start_s), spline(start_s, 1)))
    if np.linalg.norm(initial_state[:3]) - EARTH_RADIUS <= cutoff_altitude_m:
        raise ValueError("Initial state must be above the selected cutoff")
    initial_epoch = epoch + timedelta(seconds=start_s)
    end_s = start_s + max_hours * 3600.0
    orientation = interpolated_earth_orientation(epoch, start_s, end_s)
    gravity = GravityField(degree=2)
    if gravity_model_file is not None:
        model = read_icgem_gfc(gravity_model_file, max_degree=16, max_order=16, epoch=initial_epoch)
        gravity = GravityField.from_model(model, backend="native")
    event = altitude_crossing_event(cutoff_altitude_m, direction=-1.0)
    times, states, stats = propagate(
        initial_state,
        start_s,
        end_s,
        gravity=gravity,
        spacecraft=SpacecraftProperties(
            ballistic_coefficient_m2_per_kg=ballistic_coefficient_m2_per_kg
        ),
        atmosphere=AtmosphereConfig(model="msis", epoch=epoch),
        activity=CelestrakSpaceWeather.from_csv(space_weather_file),
        earth_orientation=orientation,
        events=event,
        options=Dop853Options(rtol=1e-11, atol=1e-5, max_step=30.0),
        return_history=True,
    )
    if stats.status != 1:
        raise RuntimeError("Cutoff was not reached; increase --max-hours")
    predicted_s = float(stats.event_times[0][0])
    return {
        "epoch": epoch,
        "time_s": times,
        "states": states,
        "initial_state": initial_state,
        "initial_epoch": initial_epoch,
        "start_hours": start_hours,
        "cutoff_altitude_m": cutoff_altitude_m,
        "predicted_utc": epoch + timedelta(seconds=predicted_s),
        "reference_utc": None if reference is None else epoch + timedelta(seconds=reference[0]),
        "reference_bracket_s": None if reference is None else reference[1],
        "crossing_error_s": None if reference is None else predicted_s - reference[0],
        "reference_times_s": reference_times,
        "reference_states": reference_states,
        "ballistic_coefficient_m2_per_kg": ballistic_coefficient_m2_per_kg,
    }


def ground_track(
    epoch: datetime,
    times_s: NDArray_NxM,
    states: NDArray_NxM,
    end_s: float,
    length_s: float,
) -> tuple[NDArray_NxM, NDArray_NxM, NDArray_NxM]:
    """Sample the final interval and return elapsed seconds, longitude, latitude.

    Positions and velocities define a Cartesian Hermite interpolant, including
    the exact window endpoints. Each position is rotated into ITRS at its own
    epoch, then converted to WGS84 geodetic coordinates in degrees.
    """
    start_s = max(float(times_s[0]), end_s - length_s)
    count = max(2, int(np.ceil((end_s - start_s) / 10.0)) + 1)
    sample_times = np.linspace(start_s, end_s, count)
    spline = CubicHermiteSpline(times_s, states[:, :3], states[:, 3:], axis=0)
    interpolated = np.column_stack((spline(sample_times), spline(sample_times, 1)))
    epochs = Time(epoch) + TimeDelta(sample_times, format="sec")
    fixed = celestial.convert(epochs, interpolated.T, "GCRS", "ITRS").T
    latitude, longitude, _ = ecef_to_geodetic_wgs84(fixed[:, :3])
    return sample_times, np.asarray(longitude), np.asarray(latitude)


def plot_ground_tracks(results: list[dict] | dict, length_s: float = 300.0) -> Figure:
    """Overlay the final ground-track intervals with country borders."""
    # import here so one can comment out the ground track if cartopy does not work
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    figure = plt.figure(figsize=(12, 6))
    axes = figure.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    axes.set_global()
    axes.set_facecolor("#eaf4fb")
    axes.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f4f1e8")
    axes.coastlines(resolution="110m", linewidth=0.5)
    axes.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.5, edgecolor="gray")
    grid = axes.gridlines(draw_labels=True, linewidth=0.4, alpha=0.4)
    grid.top_labels = False
    grid.right_labels = False
    runs = [results] if isinstance(results, dict) else results
    result = runs[0]
    epoch = result["epoch"]
    reference_end = result["reference_utc"]
    reference_label = "ESA reference crossing"
    if reference_end is None:
        reference_end = epoch + timedelta(seconds=float(result["reference_times_s"][-1]))
        reference_label = "ESA reference, last available"
    colors = plt.get_cmap("tab10").colors
    trajectories = [
        (
            f"Prediction +{run.get('start_hours', 0):g} h",
            run["time_s"],
            run["states"],
            run["predicted_utc"],
            colors[index % len(colors)],
        )
        for index, run in enumerate(runs)
    ]
    trajectories.append(
        (
            reference_label,
            result["reference_times_s"],
            result["reference_states"],
            reference_end,
            "black",
        )
    )
    for label, times, states, endpoint, color in trajectories:
        end_s = (endpoint - epoch).total_seconds()
        _, longitude, latitude = ground_track(epoch, times, states, end_s, length_s)
        # Split longitude wrapping to avoid drawing a line across the world.
        jumps = np.flatnonzero(np.abs(np.diff(longitude)) > 180.0) + 1
        axes.plot(
            np.insert(longitude, jumps, np.nan),
            np.insert(latitude, jumps, np.nan),
            transform=ccrs.PlateCarree(),
            color=color,
            linewidth=1.8,
            label=f"{label}: {endpoint:%m-%d %H:%M:%S} UTC",
        )
        axes.plot(
            longitude[-1],
            latitude[-1],
            marker="o",
            markersize=6,
            color=color,
            transform=ccrs.PlateCarree(),
        )
    axes.set_title(f"Ground tracks: final {length_s:g} s before each endpoint (circles)")
    axes.legend(loc="lower left", fontsize=8)
    figure.text(0.99, 0.01, "Map: Natural Earth", ha="right", fontsize=8)
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--orbit-file", type=Path, help="Cached goce_13314.sp3; otherwise download from ESA"
    )
    parser.add_argument(
        "--space-weather-file",
        type=Path,
        required=True,
        help="Historical SW-All.csv covering November 2013",
    )
    parser.add_argument(
        "--gravity-model-file",
        type=Path,
        help="Optional ICGEM model; otherwise central gravity + J2",
    )
    parser.add_argument("--altitude-km", type=float, default=140.0)
    parser.add_argument(
        "--ballistic-coefficient",
        type=float,
        default=DEFAULT_BALLISTIC_COEFFICIENT,
        help="Cd*A/m in m^2/kg",
    )
    parser.add_argument(
        "--max-hours", type=float, default=36.0, help="Maximum propagation duration for each run"
    )
    parser.add_argument(
        "--start-hours",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 10.0],
        help="Initial state offsets from the first ESA epoch (default: 0 5 10 h)",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--ground-track-length",
        type=float,
        default=300.0,
        metavar="SECONDS",
        help="Ground-track duration before each crossing (default: 300 s; with --plot)",
    )
    args = parser.parse_args()
    if args.ground_track_length <= 0:
        parser.error("--ground-track-length must be positive")
    orbit = args.orbit_file or download_file(ORBIT_URL, "goce_13314.sp3")
    results = [
        run(
            orbit,
            args.space_weather_file,
            args.altitude_km * 1000,
            args.ballistic_coefficient,
            args.gravity_model_file,
            args.max_hours,
            start_hours,
        )
        for start_hours in args.start_hours
    ]
    first = results[0]
    print(f"Reference origin (UTC): {first['epoch'].isoformat()}")
    print(f"Constant Cd*A/m: {args.ballistic_coefficient:.7g} m²/kg (campaign-average assumption)")
    print(f"Geocentric cutoff: {args.altitude_km:g} km above radius {EARTH_RADIUS:.1f} m")
    if first["reference_utc"] is None:
        print(
            "No reference crossing in the published orbit; this cutoff is an unvalidated extrapolation."
        )
    else:
        print(f"Reference crossing (UTC, interpolated): {first['reference_utc'].isoformat()}")
        bracket = first["reference_bracket_s"]
        print(f"Reference sample bracket: {bracket[1] - bracket[0]:g} s")
    print(
        f"{'Start [h]':>9}  {'Initial epoch [UTC]':25}  {'Predicted crossing [UTC]':25}  {'Error [s]':>11}"
    )
    for result in results:
        error = result["crossing_error_s"]
        error_text = "unavailable" if error is None else f"{error:+.1f}"
        print(
            f"{result['start_hours']:9g}  {result['initial_epoch']:%Y-%m-%d %H:%M:%S}        "
            f"{result['predicted_utc']:%Y-%m-%d %H:%M:%S}        {error_text:>11}"
        )
    print("Each run starts from a fresh ESA state; the drag parameter is unchanged.")
    if args.plot:
        import matplotlib.pyplot as plt

        figure, (altitude_axes, error_axes) = plt.subplots(2, 1, figsize=(9, 8))
        for result in results:
            altitude = (np.linalg.norm(result["states"][:, :3], axis=1) - EARTH_RADIUS) / 1000
            (line,) = altitude_axes.plot(
                result["time_s"] / 3600, altitude, label=f"Prediction +{result['start_hours']:g} h"
            )
            altitude_axes.plot(
                result["time_s"][0] / 3600, altitude[0], marker="o", color=line.get_color()
            )
        reference_altitude = (
            np.linalg.norm(first["reference_states"][:, :3], axis=1) - EARTH_RADIUS
        ) / 1000
        altitude_axes.plot(
            first["reference_times_s"] / 3600,
            reference_altitude,
            color="black",
            label="ESA GPS-based reference",
        )
        altitude_axes.axhline(args.altitude_km, color="gray", linestyle="--", label="cutoff")
        altitude_axes.set(xlabel="Time from first ESA epoch [h]", ylabel="Geocentric altitude [km]")
        altitude_axes.legend()
        altitude_axes.grid()
        if first["reference_utc"] is not None:
            error_axes.plot(
                [r["start_hours"] for r in results],
                [r["crossing_error_s"] / 60 for r in results],
                "o-",
            )
            error_axes.axhline(0, color="black", linewidth=0.8)
        else:
            error_axes.text(
                0.5,
                0.5,
                "No reference crossing available",
                transform=error_axes.transAxes,
                ha="center",
            )
        error_axes.set(xlabel="Initial state offset [h]", ylabel="Prediction minus reference [min]")
        error_axes.grid()
        figure.tight_layout()
        plot_ground_tracks(results, args.ground_track_length)
        plt.show()


if __name__ == "__main__":
    main()
