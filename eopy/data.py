"""Data discovery, downloading and parsing helpers."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from .types import NDArray_NxM

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
    c: NDArray_NxM
    s: NDArray_NxM
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


def download_space_weather(
    *, target_dir: str | os.PathLike[str] | None = None, refresh: bool = False
) -> Path:
    return download_file(
        DEFAULT_URLS["space_weather_celestrak"],
        "SW-All.csv",
        target_dir=target_dir,
        refresh=refresh,
    )


def download_eop(
    *, target_dir: str | os.PathLike[str] | None = None, refresh: bool = False
) -> Path:
    return download_file(
        DEFAULT_URLS["eop_celestrak"], "EOP-All.csv", target_dir=target_dir, refresh=refresh
    )


def download_gravity_model(
    url: str | None = None,
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


def _icgem_epoch(value: str) -> datetime:
    """Parse ICGEM's ``yyyymmdd.hhmm`` timestamp representation."""

    day, _, clock = value.partition(".")
    clock = (clock + "0000")[:4]
    midnight = datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc)
    # Some legacy files contain values such as ``.0060``.  Treating the clock
    # as a duration also accepts that equivalent spelling of 01:00.
    return midnight + timedelta(hours=int(clock[:2]), minutes=int(clock[2:]))


def _coefficient_epoch(value: datetime | date | str | object | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day)
    elif hasattr(value, "to_datetime"):
        parsed = value.to_datetime(timezone=timezone.utc)
    else:
        raise TypeError("epoch must be a datetime, date, ISO string, or Astropy Time")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_icgem_gfc(
    path: str | os.PathLike[str],
    max_degree: int | None = None,
    max_order: int | None = None,
    epoch: datetime | date | str | object | None = None,
) -> GravityModel:
    """Parse an ICGEM gravity model into dense coefficients.

    ICGEM 2.0 temporal records are evaluated at ``epoch``.  Supplying an
    epoch is mandatory when selected coefficients use ``gfct`` records; this
    prevents a time-variable model from silently degrading to central gravity.
    """

    header: dict[str, str] = {}
    static_coeffs: list[tuple[int, int, float, float]] = []
    temporal_records: list[
        tuple[str, int, int, float, float, datetime, datetime, float | None]
    ] = []
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
            kind = parts[0].lower()
            if kind not in {"gfc", "gfct", "trnd", "asin", "acos"} or len(parts) < 5:
                continue
            n = int(parts[1])
            m = int(parts[2])
            if max_degree is not None and n > max_degree:
                continue
            if max_order is not None and m > max_order:
                continue
            c_value = float(parts[3].replace("D", "E"))
            s_value = float(parts[4].replace("D", "E"))
            if kind == "gfc":
                static_coeffs.append((n, m, c_value, s_value))
                continue
            tail = 3 if kind in {"asin", "acos"} else 2
            if len(parts) < 5 + tail:
                raise ValueError(f"invalid {kind} record in {path}: {line}")
            t0 = _icgem_epoch(parts[-tail])
            t1 = _icgem_epoch(parts[-tail + 1])
            period = float(parts[-1].replace("D", "E")) if tail == 3 else None
            temporal_records.append((kind, n, m, c_value, s_value, t0, t1, period))

    if not static_coeffs and not temporal_records:
        raise ValueError(f"no gfc coefficients found in {path}")

    evaluation_epoch = _coefficient_epoch(epoch)
    if temporal_records and evaluation_epoch is None:
        raise ValueError(f"{path} contains time-variable coefficients; provide epoch")

    coeffs = list(static_coeffs)
    if temporal_records:
        assert evaluation_epoch is not None
        active: dict[tuple[int, int], list[tuple[str, float, float, datetime, float | None]]] = {}
        for kind, n, m, c_value, s_value, t0, t1, period in temporal_records:
            if t0 <= evaluation_epoch < t1:
                active.setdefault((n, m), []).append((kind, c_value, s_value, t0, period))
        temporal_keys = {(n, m) for _kind, n, m, _c, _s, _t0, _t1, _period in temporal_records}
        missing = sorted(temporal_keys - active.keys())
        if missing:
            first = missing[0]
            raise ValueError(
                f"no temporal gravity coefficients active at {evaluation_epoch.isoformat()} for {first}"
            )
        seconds_per_year = 365.25 * 86400.0
        for (n, m), records in active.items():
            base = [record for record in records if record[0] == "gfct"]
            if len(base) != 1:
                raise ValueError(f"expected one active gfct record for degree/order {(n, m)}")
            _, c_value, s_value, t0, _ = base[0]
            for kind, dc, ds, record_t0, period in records:
                if kind == "gfct":
                    continue
                record_years = (evaluation_epoch - record_t0).total_seconds() / seconds_per_year
                if kind == "trnd":
                    factor = record_years
                else:
                    assert period is not None
                    phase = 2.0 * np.pi * record_years / period
                    factor = np.sin(phase) if kind == "asin" else np.cos(phase)
                c_value += dc * factor
                s_value += ds * factor
            coeffs.append((n, m, c_value, s_value))

    degree = max(n for n, _m, _c, _s in coeffs)
    order = max(m for _n, m, _c, _s in coeffs)
    c = np.zeros((degree + 1, order + 1), dtype=np.float64)
    s = np.zeros((degree + 1, order + 1), dtype=np.float64)
    for n, m, c_nm, s_nm in coeffs:
        c[n, m] = c_nm
        s[n, m] = s_nm

    norm = header.get("norm", "fully_normalized").lower()
    if norm not in {"fully_normalized", "normalized", "unnormalized"}:
        raise ValueError(f"unsupported coefficient normalization {norm!r}")
    return GravityModel(
        name=header.get("modelname", Path(path).stem),
        mu=float(
            header.get("earth_gravity_constant", header.get("gm", "398600441800000.0")).replace(
                "D", "E"
            )
        ),
        reference_radius=float(header.get("radius", "6378136.3").replace("D", "E")),
        max_degree=degree,
        max_order=order,
        c=np.ascontiguousarray(c),
        s=np.ascontiguousarray(s),
        normalized=norm in {"fully_normalized", "normalized"},
    )
