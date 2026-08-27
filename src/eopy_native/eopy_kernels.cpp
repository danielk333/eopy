#include "eopy_kernels.h"

#include <cmath>
#include <vector>

namespace {

constexpr double kEpsilon = 1.0e-300;

double associated_legendre(int n, int m, double x) {
    if (m < 0 || m > n) {
        return 0.0;
    }
    double pmm = 1.0;
    if (m > 0) {
        const double somx2 = std::sqrt(std::max(0.0, (1.0 - x) * (1.0 + x)));
        double fact = 1.0;
        for (int i = 1; i <= m; ++i) {
            // Geodesy/ICGEM associated Legendre functions do not include the
            // Condon-Shortley (-1)^m phase used by many physics libraries.
            pmm *= fact * somx2;
            fact += 2.0;
        }
    }
    if (n == m) {
        return pmm;
    }
    double pmmp1 = x * static_cast<double>(2 * m + 1) * pmm;
    if (n == m + 1) {
        return pmmp1;
    }
    double pll = 0.0;
    for (int ll = m + 2; ll <= n; ++ll) {
        pll = (static_cast<double>(2 * ll - 1) * x * pmmp1 - static_cast<double>(ll + m - 1) * pmm) /
              static_cast<double>(ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}

double fully_normalized_scale(int n, int m) {
    const double delta = m == 0 ? 1.0 : 0.0;
    return std::sqrt((2.0 - delta) * static_cast<double>(2 * n + 1)) *
           std::exp(0.5 * (std::lgamma(static_cast<double>(n - m + 1)) -
                           std::lgamma(static_cast<double>(n + m + 1))));
}

double harmonic_potential(
    const double* r,
    size_t degree,
    size_t order,
    const double* c,
    const double* s,
    double mu,
    double reference_radius,
    bool normalized
) {
    const double x = r[0];
    const double y = r[1];
    const double z = r[2];
    const double rho = std::sqrt(x * x + y * y + z * z);
    if (rho <= kEpsilon) {
        return 0.0;
    }
    const double sin_lat = z / rho;
    const double lon = std::atan2(y, x);
    const size_t stride = order + 1;
    double series = 0.0;
    double rr_n = 1.0;
    for (size_t n = 0; n <= degree; ++n) {
        if (n > 0) {
            rr_n *= reference_radius / rho;
        }
        const size_t max_m = std::min(order, n);
        for (size_t m = 0; m <= max_m; ++m) {
            const double c_nm = c[n * stride + m];
            const double s_nm = s == nullptr ? 0.0 : s[n * stride + m];
            if (c_nm == 0.0 && s_nm == 0.0) {
                continue;
            }
            const int ni = static_cast<int>(n);
            const int mi = static_cast<int>(m);
            double p_nm = associated_legendre(ni, mi, sin_lat);
            if (normalized) {
                p_nm *= fully_normalized_scale(ni, mi);
            }
            series += rr_n * p_nm * (c_nm * std::cos(static_cast<double>(m) * lon) +
                                     s_nm * std::sin(static_cast<double>(m) * lon));
        }
    }
    return mu / rho * series;
}

bool harmonic_acceleration(
    const double* r,
    size_t degree,
    size_t order,
    const double* c,
    const double* s,
    double mu,
    double reference_radius,
    bool normalized,
    double* acceleration
) {
    const double x_cart = r[0];
    const double y_cart = r[1];
    const double z_cart = r[2];
    const double rho = std::sqrt(x_cart * x_cart + y_cart * y_cart + z_cart * z_cart);
    if (rho <= kEpsilon) {
        return false;
    }
    const double sin_phi = z_cart / rho;
    const double cos_phi = std::hypot(x_cart, y_cart) / rho;
    // Polar passages are rare evaluation points and the finite-difference
    // potential is robust there, unlike longitude-based spherical formulas.
    if (cos_phi < 1.0e-10) {
        const double step = std::max(0.1, 1.0e-7 * rho);
        for (size_t axis = 0; axis < 3; ++axis) {
            double rp[3] = {r[0], r[1], r[2]};
            double rm[3] = {r[0], r[1], r[2]};
            rp[axis] += step;
            rm[axis] -= step;
            acceleration[axis] =
                (harmonic_potential(rp, degree, order, c, s, mu, reference_radius, normalized) -
                 harmonic_potential(rm, degree, order, c, s, mu, reference_radius, normalized)) /
                (2.0 * step);
        }
        return true;
    }

    const double longitude = std::atan2(y_cart, x_cart);
    const size_t width = degree + 1;
    std::vector<double> p(width * width, 0.0);
    auto at = [width, &p](size_t n, size_t m) -> double& { return p[n * width + m]; };
    at(0, 0) = 1.0;
    for (size_t m = 1; m <= degree; ++m) {
        at(m, m) = static_cast<double>(2 * m - 1) * cos_phi * at(m - 1, m - 1);
    }
    for (size_t m = 0; m < degree; ++m) {
        at(m + 1, m) = static_cast<double>(2 * m + 1) * sin_phi * at(m, m);
    }
    for (size_t m = 0; m <= degree; ++m) {
        for (size_t n = m + 2; n <= degree; ++n) {
            at(n, m) =
                (static_cast<double>(2 * n - 1) * sin_phi * at(n - 1, m) -
                 static_cast<double>(n + m - 1) * at(n - 2, m)) /
                static_cast<double>(n - m);
        }
    }

    double radial_sum = 0.0;
    double latitude_sum = 0.0;
    double longitude_sum = 0.0;
    double radius_power = 1.0;
    const size_t stride = order + 1;
    for (size_t n = 0; n <= degree; ++n) {
        if (n > 0) {
            radius_power *= reference_radius / rho;
        }
        const size_t max_m = std::min(order, n);
        for (size_t m = 0; m <= max_m; ++m) {
            const double c_nm = c[n * stride + m];
            const double s_nm = s == nullptr ? 0.0 : s[n * stride + m];
            if (c_nm == 0.0 && s_nm == 0.0) {
                continue;
            }
            const double scale = normalized
                ? fully_normalized_scale(static_cast<int>(n), static_cast<int>(m))
                : 1.0;
            const double p_nm = at(n, m) * scale;
            const double angle = static_cast<double>(m) * longitude;
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            const double trig = c_nm * cosine + s_nm * sine;
            radial_sum += static_cast<double>(n + 1) * radius_power * p_nm * trig;
            if (n > 0) {
                const double previous = n == m ? 0.0 : at(n - 1, m);
                const double dp_dx =
                    (static_cast<double>(n) * sin_phi * at(n, m) -
                     static_cast<double>(n + m) * previous) /
                    (sin_phi * sin_phi - 1.0);
                latitude_sum += radius_power * (cos_phi * dp_dx * scale) * trig;
            }
            if (m > 0) {
                longitude_sum += radius_power * p_nm * static_cast<double>(m) *
                    (-c_nm * sine + s_nm * cosine);
            }
        }
    }

    const double radial = -mu * radial_sum / (rho * rho);
    const double north = mu * latitude_sum / (rho * rho);
    const double east = mu * longitude_sum / (rho * rho * cos_phi);
    const double cos_lon = std::cos(longitude);
    const double sin_lon = std::sin(longitude);
    acceleration[0] = radial * cos_phi * cos_lon - north * sin_phi * cos_lon - east * sin_lon;
    acceleration[1] = radial * cos_phi * sin_lon - north * sin_phi * sin_lon + east * cos_lon;
    acceleration[2] = radial * sin_phi + north * cos_phi;
    return true;
}

}  // namespace

extern "C" {

const char* eopy_native_version(void) {
    return "0.1.0";
}

int eopy_central_j2_accel_batch(
    size_t count,
    const double* position_m,
    double mu,
    double reference_radius_m,
    double j2,
    double* acceleration_mps2
) {
    if (position_m == nullptr || acceleration_mps2 == nullptr) {
        return 1;
    }
    for (size_t i = 0; i < count; ++i) {
        const double* r = position_m + 3 * i;
        double* a = acceleration_mps2 + 3 * i;
        const double x = r[0];
        const double y = r[1];
        const double z = r[2];
        const double r2 = x * x + y * y + z * z;
        if (r2 <= kEpsilon) {
            return 2;
        }
        const double rho = std::sqrt(r2);
        const double inv_r3 = 1.0 / (r2 * rho);
        a[0] = -mu * x * inv_r3;
        a[1] = -mu * y * inv_r3;
        a[2] = -mu * z * inv_r3;
        if (j2 != 0.0) {
            const double z2_r2 = z * z / r2;
            const double factor = 1.5 * j2 * mu * reference_radius_m * reference_radius_m / (r2 * r2 * rho);
            a[0] += factor * x * (5.0 * z2_r2 - 1.0);
            a[1] += factor * y * (5.0 * z2_r2 - 1.0);
            a[2] += factor * z * (5.0 * z2_r2 - 3.0);
        }
    }
    return 0;
}

int eopy_holmes_featherstone_accel(
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
) {
    if (position_m == nullptr || c == nullptr || acceleration_mps2 == nullptr) {
        return 1;
    }
    const bool use_normalized = normalized != 0;
    for (size_t i = 0; i < count; ++i) {
        const double* r = position_m + 3 * i;
        double* a = acceleration_mps2 + 3 * i;
        if (!harmonic_acceleration(
                r, degree, order, c, s, mu, reference_radius_m, use_normalized, a)) {
            return 2;
        }
    }
    return 0;
}

}
