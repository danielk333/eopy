"""Data discovery, downloading and parsing helpers."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np

DEFAULT_URLS = {
    "gravity_icgem_eigen_6s4_v2": "https://icgem.gfz.de/getmodel/gfc/4cc119d62d3f83adce857914bcabdfa7ca2f91677ed2905934cf52698584b4c9/EIGEN-6S4%20(v2).gfc",
    "space_weather_celestrak": "https://celestrak.org/SpaceData/SW-All.csv",
    "eop_celestrak": "https://celestrak.org/SpaceData/EOP-All.csv",
}
DEFAULT_GRAVITY_MODEL_URL = DEFAULT_URLS["gravity_icgem_eigen_6s4_v2"]
DEFAULT_GRAVITY_MODEL_FILENAME = "EIGEN-6S4_v2.gfc"


@dataclass(frozen=True, slots=True)
class DataStore:
    root: Path

    @classmethod
    def default(cls) -> "DataStore":
        return cls(resolve_data_path())

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)


@dataclass(slots=True)
class GravityModel:
    name: str
    mu: float
    reference_radius: float
    max_degree: int
    max_order: int
    c: np.ndarray
    s: np.ndarray
    normalized: bool = True


def resolve_data_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Return the eopy data directory, creating it if needed."""

    if path is not None:
        root = Path(path)
    elif "EOPY_DATA_PATH" in os.environ:
        root = Path(os.environ["EOPY_DATA_PATH"])
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "eopy"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    url: str,
    filename: str | None = None,
    *,
    target_dir: str | os.PathLike[str] | None = None,
    sha256: str | None = None,
    refresh: bool = False,
) -> Path:
    """Download a file into the eopy data cache using an atomic rename."""

    root = resolve_data_path(target_dir)
    out = root / (filename or url.rsplit("/", 1)[-1])
    if out.exists() and not refresh:
        if sha256 is not None and _sha256(out) != sha256:
            raise ValueError(f"checksum mismatch for existing file {out}")
        return out

    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=root)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with urlopen(url, timeout=60.0) as response, tmp.open("wb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
        if sha256 is not None and _sha256(tmp) != sha256:
            raise ValueError(f"checksum mismatch for downloaded file {url}")
        tmp.replace(out)
    finally:
        if tmp.exists():
            tmp.unlink()
    return out


def download_space_weather(*, target_dir: str | os.PathLike[str] | None = None, refresh: bool = False) -> Path:
    return download_file(DEFAULT_URLS["space_weather_celestrak"], "SW-All.csv", target_dir=target_dir, refresh=refresh)


def download_eop(*, target_dir: str | os.PathLike[str] | None = None, refresh: bool = False) -> Path:
    return download_file(DEFAULT_URLS["eop_celestrak"], "EOP-All.csv", target_dir=target_dir, refresh=refresh)


def download_gravity_model(
    url: str | None = None,
    *,
    filename: str | None = None,
    target_dir: str | os.PathLike[str] | None = None,
    sha256: str | None = None,
    refresh: bool = False,
) -> Path:
    """Download an ICGEM-compatible gravity model file."""

    if url is None:
        url = DEFAULT_GRAVITY_MODEL_URL
        filename = DEFAULT_GRAVITY_MODEL_FILENAME if filename is None else filename
    return download_file(url, filename, target_dir=target_dir, sha256=sha256, refresh=refresh)


def read_icgem_gfc(path: str | os.PathLike[str], *, max_degree: int | None = None, max_order: int | None = None) -> GravityModel:
    """Parse an ICGEM ``.gfc`` gravity model into dense coefficient arrays."""

    header: dict[str, str] = {}
    coeffs: list[tuple[int, int, float, float]] = []
    in_header = True
    with Path(path).open("rt", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if in_header:
                if parts[0].lower() == "end_of_head":
                    in_header = False
                    continue
                if len(parts) >= 2:
                    header[parts[0].lower()] = parts[1]
                continue
            if parts[0].lower() != "gfc" or len(parts) < 5:
                continue
            n = int(parts[1])
            m = int(parts[2])
            if max_degree is not None and n > max_degree:
                continue
            if max_order is not None and m > max_order:
                continue
            coeffs.append((n, m, float(parts[3].replace("D", "E")), float(parts[4].replace("D", "E"))))

    if not coeffs:
        raise ValueError(f"no gfc coefficients found in {path}")

    degree = max(n for n, _m, _c, _s in coeffs)
    order = max(m for _n, m, _c, _s in coeffs)
    c = np.zeros((degree + 1, order + 1), dtype=np.float64)
    s = np.zeros((degree + 1, order + 1), dtype=np.float64)
    for n, m, c_nm, s_nm in coeffs:
        c[n, m] = c_nm
        s[n, m] = s_nm

    norm = header.get("norm", "fully_normalized").lower()
    return GravityModel(
        name=header.get("modelname", Path(path).stem),
        mu=float(header.get("earth_gravity_constant", header.get("gm", "398600441800000.0")).replace("D", "E")),
        reference_radius=float(header.get("radius", "6378136.3").replace("D", "E")),
        max_degree=degree,
        max_order=order,
        c=np.ascontiguousarray(c),
        s=np.ascontiguousarray(s),
        normalized=norm in {"fully_normalized", "normalized"},
    )
