#pragma once

#include <stddef.h>

#if defined(_WIN32)
#define EOPY_EXPORT __declspec(dllexport)
#else
#define EOPY_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

EOPY_EXPORT const char* eopy_native_version(void);

EOPY_EXPORT int eopy_central_j2_accel_batch(
    size_t count,
    const double* position_m,
    double mu,
    double reference_radius_m,
    double j2,
    double* acceleration_mps2
);

EOPY_EXPORT int eopy_spherical_harmonic_accel(
    size_t count,
    const double* position_m,
    size_t degree,
    size_t order,
    const double* c,
    const double* s,
    double mu,
    double reference_radius_m,
    int normalized,
    double* acceleration_mps2
);

}
