#include "eopy_kernels.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
constexpr double kEpsilon = 1.0e-300;

// Real geodesy (4pi) normalization, no Condon-Shortley phase.
// Standard three-term recurrence; normalization definitions:
// Wieczorek & Meschede (2018), doi:10.1029/2018GC007529.
// Differentiate the recurrence with respect to latitude to avoid division
// by sin(latitude)^2 - 1 near polar passages.
void legendre(size_t degree, double latitude, bool normalized,
              std::vector<double>& p, std::vector<double>& dp) {
    const size_t width = degree + 1;
    p.assign(width * width, 0.0);
    dp.assign(width * width, 0.0);
    const double x = std::sin(latitude), y = std::cos(latitude);
    p[0] = 1.0;
    for (size_t m = 0; m <= degree; ++m) {
        const size_t i = m * width + m;
        if (m > 0) {
            const size_t prev = (m - 1) * width + m - 1;
            const double factor = normalized
                ? (m == 1 ? std::sqrt(3.0) : std::sqrt((2.0*m+1)/(2.0*m)))
                : 2.0*m-1;
            p[i] = factor * y * p[prev];
            dp[i] = factor * (-x * p[prev] + y * dp[prev]);
        }
        if (m < degree) {
            const double factor = normalized ? std::sqrt(2.0*m+3) : 2.0*m+1;
            p[i + width] = factor * x * p[i];
            dp[i + width] = factor * (y * p[i] + x * dp[i]);
        }
        for (size_t n = m + 2; n <= degree; ++n) {
            const double a = normalized
                ? std::sqrt((4.0*n*n-1)/(double(n*n)-double(m*m)))
                : (2.0*n-1)/(n-m);
            const double b = normalized
                ? std::sqrt(((2.0*n+1)*((n-1.0)*(n-1.0)-m*m))/
                            ((2.0*n-3)*(double(n*n)-double(m*m))))
                : (n+m-1.0)/(n-m);
            const size_t j = n * width + m;
            p[j] = a*x*p[j-width] - b*p[j-2*width];
            dp[j] = a*(y*p[j-width]+x*dp[j-width]) - b*dp[j-2*width];
        }
    }
}

double potential(const double* r, size_t degree, size_t order,
                 const double* c, const double* s, double mu, double radius,
                 bool normalized) {
    const double rho = std::hypot(std::hypot(r[0], r[1]), r[2]);
    const double latitude = std::atan2(r[2], std::hypot(r[0], r[1]));
    const double longitude = std::atan2(r[1], r[0]);
    std::vector<double> p, dp;
    legendre(degree, latitude, normalized, p, dp);
    double sum = 0.0, power = 1.0;
    for (size_t n=0; n<=degree; ++n) {
        for (size_t m=0; m<=std::min(n,order); ++m) {
            const size_t i=n*(order+1)+m;
            sum += power*p[n*(degree+1)+m]*(c[i]*std::cos(m*longitude)+
                   (s ? s[i] : 0.0)*std::sin(m*longitude));
        }
        power *= radius/rho;
    }
    return mu/rho*sum;
}

bool harmonic_acceleration(const double* r, size_t degree, size_t order,
                           const double* c, const double* s, double mu,
                           double radius, bool normalized, double* acceleration) {
    const double horizontal=std::hypot(r[0],r[1]);
    const double rho=std::hypot(horizontal,r[2]);
    if (rho <= kEpsilon) return false;
    if (horizontal/rho < 1e-8) {
        const double step=std::max(0.1,1e-7*rho);
        for (size_t axis=0; axis<3; ++axis) {
            double rp[3]={r[0],r[1],r[2]}, rm[3]={r[0],r[1],r[2]};
            rp[axis]+=step; rm[axis]-=step;
            acceleration[axis]=(potential(rp,degree,order,c,s,mu,radius,normalized)-
                                potential(rm,degree,order,c,s,mu,radius,normalized))/(2*step);
        }
        return true;
    }
    const double latitude=std::atan2(r[2],horizontal), longitude=std::atan2(r[1],r[0]);
    std::vector<double> p, dp;
    legendre(degree,latitude,normalized,p,dp);
    double radial=0, north=0, east=0, power=1;
    for (size_t n=0; n<=degree; ++n) {
        for (size_t m=0; m<=std::min(n,order); ++m) {
            const size_t i=n*(order+1)+m, j=n*(degree+1)+m;
            const double cosine=std::cos(m*longitude), sine=std::sin(m*longitude);
            const double snm=s ? s[i] : 0.0, trig=c[i]*cosine+snm*sine;
            radial -= (n+1)*power*p[j]*trig;
            north += power*dp[j]*trig;
            east += power*p[j]*m*(-c[i]*sine+snm*cosine);
        }
        power *= radius/rho;
    }
    radial *= mu/(rho*rho); north *= mu/(rho*rho);
    east *= mu/(rho*horizontal);
    const double cp=horizontal/rho, sp=r[2]/rho, cl=std::cos(longitude), sl=std::sin(longitude);
    acceleration[0]=radial*cp*cl-north*sp*cl-east*sl;
    acceleration[1]=radial*cp*sl-north*sp*sl+east*cl;
    acceleration[2]=radial*sp+north*cp;
    return true;
}
} // namespace

extern "C" {

const char* eopy_native_version(void) {
    return "1.0.0rc1";
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

int eopy_spherical_harmonic_accel(
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
    try {
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
    } catch (...) {
        return 3; // Allocation or other C++ failure; never unwind across the C ABI.
    }
}

}
