"""Download eopy input data files into a target folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from eopy.data import (
    DEFAULT_GRAVITY_MODEL_URL,
    download_eop,
    download_file,
    download_gravity_model,
    download_space_weather,
)


def download_requested_files(
    target_dir: str | Path,
    *,
    gravity_url: str | None = None,
    gravity_filename: str | None = None,
    gravity_sha256: str | None = None,
    include_gravity: bool = True,
    include_space_weather: bool = True,
    include_eop: bool = True,
    refresh: bool = False,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if include_gravity:
        paths["gravity"] = download_gravity_model(
            gravity_url,
            filename=gravity_filename,
            target_dir=target_dir,
            sha256=gravity_sha256,
            refresh=refresh,
        )
    if include_space_weather:
        paths["space_weather"] = download_space_weather(target_dir=target_dir, refresh=refresh)
    if include_eop:
        paths["eop"] = download_eop(target_dir=target_dir, refresh=refresh)
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_dir", help="Folder where downloaded files should be stored.")
    parser.add_argument(
        "--gravity-url",
        default=DEFAULT_GRAVITY_MODEL_URL,
        help="URL to an ICGEM-compatible .gfc gravity model.",
    )
    parser.add_argument("--gravity-filename", help="Filename to use for the cached gravity model.")
    parser.add_argument(
        "--gravity-sha256", help="Expected SHA256 for the downloaded gravity model."
    )
    parser.add_argument("--no-gravity", action="store_true", help="Skip gravity model download.")
    parser.add_argument(
        "--no-space-weather", action="store_true", help="Skip CelesTrak space-weather CSV download."
    )
    parser.add_argument("--no-eop", action="store_true", help="Skip CelesTrak EOP CSV download.")
    parser.add_argument(
        "--refresh", action="store_true", help="Redownload files even if they already exist."
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="Download an additional arbitrary file into target_dir using NAME as the filename.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = download_requested_files(
        args.target_dir,
        gravity_url=args.gravity_url,
        gravity_filename=args.gravity_filename,
        gravity_sha256=args.gravity_sha256,
        include_gravity=not args.no_gravity,
        include_space_weather=not args.no_space_weather,
        include_eop=not args.no_eop,
        refresh=args.refresh,
    )
    for item in args.url:
        if "=" not in item:
            raise ValueError("--url entries must use NAME=URL")
        name, url = item.split("=", 1)
        paths[name] = download_file(url, name, target_dir=args.target_dir, refresh=args.refresh)

    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
