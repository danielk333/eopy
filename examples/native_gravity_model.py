"""Propagate with an ICGEM gravity model through the native gravity backend."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eopy.data import DEFAULT_GRAVITY_MODEL_URL, download_gravity_model, read_icgem_gfc
from eopy.gravity import GravityField
from eopy.frames import interpolated_earth_orientation
from eopy.integrators import Dop853Options
from eopy.propagator import propagate


def resolve_gravity_model_path(
    *,
    gravity_model: str | os.PathLike[str] | None = None,
    gravity_url: str | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    filename: str | None = None,
    sha256: str | None = None,
    refresh: bool = False,
) -> Path:
    if gravity_model is not None:
        return Path(gravity_model)
    return download_gravity_model(
        gravity_url,
        filename=filename,
        target_dir=data_dir,
        sha256=sha256,
        refresh=refresh,
    )


def run(
    gravity_model_path: str | os.PathLike[str],
    *,
    degree: int | None = 4,
    order: int | None = 4,
    backend: str = "native",
    duration_s: float = 1800.0,
    max_step_s: float = 60.0,
    epoch: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc),
):
    model = read_icgem_gfc(
        gravity_model_path,
        max_degree=degree,
        max_order=order,
        epoch=epoch,
    )
    field = GravityField.from_model(model, degree=degree, order=order, backend=backend)
    radius_m = model.reference_radius + 700_000.0
    speed_mps = np.sqrt(model.mu / radius_m)
    state0 = np.array([radius_m, 0.0, 0.0, 0.0, speed_mps, 0.0])
    orientation = interpolated_earth_orientation(epoch, 0.0, duration_s)

    t, states, stats = propagate(
        state0,
        0.0,
        duration_s,
        gravity=field,
        earth_orientation=orientation,
        options=Dop853Options(rtol=1e-10, atol=1e-3, max_step=max_step_s),
        return_history=True,
    )
    altitude_m = np.linalg.norm(states[:, :3], axis=1) - model.reference_radius
    return {
        "model_name": model.name,
        "model_path": str(gravity_model_path),
        "backend": backend,
        "time_s": t,
        "states": states,
        "altitude_m": altitude_m,
        "stats": stats,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--gravity-model", help="Path to an existing ICGEM .gfc gravity model file.")
    source.add_argument("--gravity-url", default=DEFAULT_GRAVITY_MODEL_URL, help="URL to download an ICGEM .gfc gravity model file.")
    parser.add_argument("--data-dir", help="Directory used for downloaded gravity model files.")
    parser.add_argument("--filename", help="Cached filename to use with --gravity-url.")
    parser.add_argument("--sha256", help="Expected SHA256 for the downloaded gravity model.")
    parser.add_argument("--refresh", action="store_true", help="Redownload the model even if it already exists.")
    parser.add_argument("--native-library", help="Path to libeopykernels shared library from an out-of-tree build.")
    parser.add_argument("--backend", choices=["native", "auto", "python"], default="native", help="Gravity backend to use.")
    parser.add_argument("--degree", type=int, default=8, help="Maximum gravity degree to load.")
    parser.add_argument("--order", type=int, default=10, help="Maximum gravity order to load.")
    parser.add_argument("--duration", type=float, default=1800.0, help="Propagation duration in seconds.")
    parser.add_argument("--max-step", type=float, default=60.0, help="Integrator maximum step in seconds.")
    parser.add_argument("--epoch", default="2024-01-01T00:00:00+00:00", help="UTC coefficient and propagation epoch.")
    parser.add_argument("--no-plot", action="store_true", help="Do not display the propagation plot.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.native_library:
        os.environ["EOPY_NATIVE_LIBRARY"] = args.native_library

    gravity_model_path = resolve_gravity_model_path(
        gravity_model=args.gravity_model,
        gravity_url=args.gravity_url,
        data_dir=args.data_dir,
        filename=args.filename,
        sha256=args.sha256,
        refresh=args.refresh,
    )
    result = run(
        gravity_model_path,
        degree=args.degree,
        order=args.order,
        backend=args.backend,
        duration_s=args.duration,
        max_step_s=args.max_step,
        epoch=datetime.fromisoformat(args.epoch.replace("Z", "+00:00")),
    )

    print(f"Model: {result['model_name']}")
    print(f"Path: {result['model_path']}")
    print(f"Backend: {result['backend']}")
    print(f"Accepted steps: {result['stats'].accepted_steps}")
    print(f"Final altitude: {result['altitude_m'][-1] / 1000.0:.3f} km")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        states = result["states"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(states[:, 0] / 1000.0, states[:, 1] / 1000.0)
        axes[0].set_title("Native gravity propagation")
        axes[0].set_xlabel("x [km]")
        axes[0].set_ylabel("y [km]")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].grid(True)
        axes[1].plot(result["time_s"] / 60.0, result["altitude_m"] / 1000.0)
        axes[1].set_xlabel("time [min]")
        axes[1].set_ylabel("altitude [km]")
        axes[1].grid(True)
        plt.tight_layout()
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
