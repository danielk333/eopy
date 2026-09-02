"""DTM2000 thermosphere density kernel.

This module is a Python adaptation of Orekit's DTM2000 implementation,
Copyright 2002-2026 CS GROUP, licensed under Apache-2.0. It was translated and
modified for eopy in 2026. The equations trace back to the original Fortran
routine by R. Biancale and S. Bruinsma. See ``assets/Orekit-APACHE-2.0.txt``
and ``assets/Orekit-NOTICE.txt``.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import math

import numpy as np


_ALEFA = np.array([0.0, -0.40, -0.38, 0.0, 0.0, 0.0, 0.0])
_MA = np.array([0.0, 1.0, 4.0, 16.0, 28.0, 32.0, 14.0])
_VMA = np.array([0.0, 1.6606e-24, 6.6423e-24, 26.569e-24, 46.4958e-24, 53.1381e-24, 23.2479e-24])
_RE = 6356.77
_ZLB0 = 120.0
_CPMG = 0.19081
_SPMG = 0.98163
_XLMG = -1.2392
_GSURF = 980.665
_RGAS = 831.4
_ROT = 0.017214206
_ROT2 = 0.034428412


@lru_cache(maxsize=1)
def _coefficients() -> tuple[np.ndarray, ...]:
    arrays = [np.zeros(97) for _ in range(9)]
    resource = files("eopy").joinpath("assets/dtm_2000.txt")
    with resource.open("rt", encoding="ascii") as handle:
        next(handle)
        next(handle)
        for line in handle:
            index = int(line[:4].replace(" ", "0"))
            rest = line[4:]
            for array in arrays:
                array[index] = float(rest[:13].replace(" ", "0"))
                rest = rest[22:]
    return tuple(arrays)


def _semestrial_correction(value: float, day: int) -> float:
    if day >= 100:
        return value - 2.0 * value * ((day - 244) / 40.0)
    return 2.0 * value * ((day - 59) / 40.0) - value


def _g_function(a: np.ndarray, day: int, lon: float, lat: float, local_time: float,
                flux: float, mean_flux: float, kp3: float, kp24: float,
                ff0: int, season: int) -> float:
    da = np.zeros(97)
    c = math.sin(lat)
    c2 = c * c
    c4 = c2 * c2
    s = math.cos(lat)
    s2 = s * s
    p10 = c
    p20 = 1.5 * c2 - 0.5
    p30 = c * (2.5 * c2 - 1.5)
    p40 = 4.375 * c4 - 3.75 * c2 + 0.375
    p50 = c * (7.875 * c4 - 8.75 * c2 + 1.875)
    p60 = (5.5 * c * p50 - 2.5 * p40) / 3.0
    p11 = s
    p21 = 3.0 * c * s
    p31 = s * (7.5 * c2 - 1.5)
    p41 = c * s * (17.5 * c2 - 7.5)
    p51 = s * (39.375 * c4 - 26.25 * c2 + 1.875)
    p22 = 3.0 * s2
    p32 = 15.0 * c * s2
    p42 = s2 * (52.5 * c2 - 7.5)
    p52 = 3.0 * c * p42 - 2.0 * p32
    p62 = 2.75 * c * p52 - 1.75 * p42
    p33 = 15.0 * s * s2
    cmg = s * _CPMG * math.cos(lon - _XLMG) + c * _SPMG
    p20mg = 1.5 * cmg * cmg - 0.5
    p40mg = 4.375 * cmg**4 - 3.75 * cmg * cmg + 0.375
    ch, sh = math.cos(local_time), math.sin(local_time)
    c2h, s2h = ch * ch - sh * sh, 2.0 * ch * sh
    c3h, s3h = c2h * ch - s2h * sh, s2h * ch + c2h * sh

    da[2], da[3], da[74] = p20, p40, p10
    a74, a77, a78 = a[74], a[77], a[78]
    if season == -1:
        a74, a77, a78 = -a74, -a77, -a78
    elif season == 0:
        a74 = _semestrial_correction(a74, day)
        a77 = _semestrial_correction(a77, day)
        a78 = _semestrial_correction(a78, day)
    da[77], da[78], da[79] = p30, p50, p60

    da[4] = flux - mean_flux
    da[6] = mean_flux - 150.0
    da[70] = da[71] = 0.0
    da[5], da[69] = da[4] ** 2, da[6] ** 2
    da[82], da[83], da[84] = da[4] * p10, da[4] * p20, da[4] * p30
    da[85], da[86], da[87] = da[6] * p20, da[6] * p30, da[6] * p40

    dkp = kp3
    dkpm = kp24
    da[7], da[8], da[68] = dkp, p20mg * dkp, p40mg * dkp
    da[60], da[61], da[75] = dkp**2, p20mg * dkp**2, dkp**4
    da[64], da[65], da[72] = dkpm, p20mg * dkpm, p40mg * dkpm
    da[66], da[73], da[76] = dkpm**2, p20mg * dkpm**2, dkpm**4

    nonperiodic = sum(a[i] * da[i] for i in (4, 5, 6, 69, 82, 83, 84, 85, 86, 87))
    flux_coupling = 1.0 + nonperiodic * ff0
    nonperiodic += sum(a[i] * da[i] for i in (2, 3, 7, 8, 60, 61, 64, 65, 66, 68, 72, 73, 75, 76, 79))
    nonperiodic += a74 * da[74] + a77 * da[77] + a78 * da[78]

    coste = math.cos(_ROT * (day - a[18]))
    da[9] = math.cos(_ROT * (day - a[11]))
    da[10] = p20 * da[9]
    da[12] = math.cos(_ROT2 * (day - a[14]))
    da[13] = p20 * da[12]
    da[15], da[16], da[17] = p10 * coste, p30 * coste, p50 * coste
    cos2te = math.cos(_ROT2 * (day - a[20]))
    da[19], da[39], da[59] = p10 * cos2te, p30 * cos2te, p50 * cos2te
    da[21:31] = (p11*ch, p31*ch, p51*ch, p11*ch*coste, p21*ch*coste,
                 p11*sh, p31*sh, p51*sh, p11*sh*coste, p21*sh*coste)
    da[31], da[37], da[32] = p22*c2h, p42*c2h, p32*c2h*coste
    da[33], da[38], da[34] = p22*s2h, p42*s2h, p32*s2h*coste
    da[88], da[89], da[90], da[91] = p32*c2h, p32*s2h, p52*c2h, p52*s2h
    a88, a89, a90, a91 = a[88], a[89], a[90], a[91]
    if season == -1:
        a88, a89, a90, a91 = -a88, -a89, -a90, -a91
    elif season == 0:
        a88 = _semestrial_correction(a88, day)
        a89 = _semestrial_correction(a89, day)
        a90 = _semestrial_correction(a90, day)
        a91 = _semestrial_correction(a91, day)
    da[92], da[93], da[35], da[36] = p62*c2h, p62*s2h, p33*c3h, p33*s3h
    periodic_indices = list(range(9, 40)) + [59, 92, 93]
    periodic = sum(a[i] * da[i] for i in periodic_indices)
    periodic += a88*da[88] + a89*da[89] + a90*da[90] + a91*da[91]
    da[40:49] = (p10*coste*dkp, p30*coste*dkp, p50*coste*dkp,
                 p11*ch*dkp, p31*ch*dkp, p51*ch*dkp,
                 p11*sh*dkp, p31*sh*dkp, p51*sh*dkp)
    periodic += sum(a[i] * da[i] for i in range(40, 49))
    clfl, slfl = math.cos(lon), math.sin(lon)
    da[49:59] = (p11*clfl, p21*clfl, p31*clfl, p41*clfl, p51*clfl,
                 p11*slfl, p21*slfl, p31*slfl, p41*slfl, p51*slfl)
    periodic += sum(a[i] * da[i] for i in range(49, 59))
    return nonperiodic + periodic * flux_coupling


def density_from_inputs(day: int, altitude_m: float, longitude_rad: float,
                        latitude_rad: float, local_solar_time_rad: float,
                        f10: float, f10_mean: float, kp_3h: float,
                        kp_24h: float) -> float:
    """Evaluate DTM2000 for one geodetic point and explicit model inputs."""
    if not 1 <= int(day) <= 366:
        raise ValueError("day must be in the range 1..366")
    if altitude_m < 120_000.0:
        raise ValueError("DTM2000 requires altitude at or above 120 km")
    values = (altitude_m, longitude_rad, latitude_rad, local_solar_time_rad,
              f10, f10_mean, kp_3h, kp_24h)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("DTM2000 inputs must contain only finite values")
    tt, h, he, oxygen, n2, o2, nitrogen, t0, tp = _coefficients()
    season = -1 if day < 59 or day > 284 else (1 if 99 < day < 244 else 0)
    tinf = tt[1] * (1.0 + _g_function(tt, day, longitude_rad, latitude_rad,
                                      local_solar_time_rad, f10, f10_mean,
                                      kp_3h, kp_24h, 1, 1))
    t120 = t0[1] + _g_function(t0, day, longitude_rad, latitude_rad,
                                local_solar_time_rad, f10, f10_mean,
                                kp_3h, kp_24h, 0, season)
    tp120 = tp[1] + _g_function(tp, day, longitude_rad, latitude_rad,
                                local_solar_time_rad, f10, f10_mean,
                                kp_3h, kp_24h, 0, season)
    bases = np.zeros(7)
    for index, coefficients in enumerate((h, he, oxygen, n2, o2, nitrogen), 1):
        flag = 0 if index in (1, 2) else 1
        bases[index] = coefficients[1] * math.exp(_g_function(
            coefficients, day, longitude_rad, latitude_rad, local_solar_time_rad,
            f10, f10_mean, kp_3h, kp_24h, flag, 1))
    altitude_km = altitude_m / 1000.0
    sigma = tp120 / (tinf - t120)
    zeta = (altitude_km - _ZLB0) * (_RE + _ZLB0) / (_RE + altitude_km)
    sigzeta = sigma * zeta
    temperature = tinf - (tinf - t120) * math.exp(-sigzeta)
    glb = (_GSURF / (1.0 + _ZLB0 / _RE) ** 2) / (sigma * _RGAS * tinf)
    density_g_cm3 = 0.0
    for index in range(1, 7):
        gamma = _MA[index] * glb
        exponent = math.log(t120 / temperature) * (1.0 + _ALEFA[index] + gamma) - sigzeta * gamma
        density_g_cm3 += bases[index] * math.exp(exponent) * _VMA[index]
    return density_g_cm3 * 1000.0
