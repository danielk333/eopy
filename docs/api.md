# Public interface

All physical inputs use SI units unless a name explicitly states otherwise. Array aliases describe
float64 arrays; shapes are part of the interface below.

## Propagation

`propagate(state0, t0, t1, ...)` returns `(time, state, Dop853Stats)`. States are `[x, y, z, vx, vy,
vz]`, shaped `(6,)` or `(N, 6)`. Normally these are geocentric GCRS states. Force callbacks must use
the propagated axes. `extra_acceleration(t, positions, velocities)` returns an `(N, 3)` array.

`Dop853Options` controls relative/absolute tolerances, initial step and maximum step, all in
propagation seconds and state units. `integrate_dop853` also accepts arbitrary state shapes and
preserves them when calling the RHS. Integration failure raises `RuntimeError`; unsupported sample
ordering raises `ValueError`. With `return_history=True` output adds a leading time dimension. With
`sample_times`, output contains only requested times reached before termination.

`Dop853Stats` contains evaluation counts, solver status/message, optional event times/states (each
event state history is `(K,6)` or `(K,N,6)`), and an optional `PropagationProfile`. Events follow
SciPy's `terminal`/`direction` callable convention. Built-in `altitude_crossing_event` and
`cylindrical_eclipse_event` are available from `eopy.events`.

## Force models

- `GravityField`, `gravity_acceleration` -> Positions `(3,)` or `(N,3)`; returns acceleration of the same shape. Default central Earth + J2. Coefficients take precedence over analytic J2. `from_model` transfers gravity-file metadata. 
- `SpacecraftProperties` -> Mass (kg), drag area (m^2), dimensionless drag coefficient, or explicit ballistic coefficient (m^2/kg). Scalars or per-spacecraft arrays. 
- `AtmosphereConfig` -> Exponential, MSIS, or DTM2000 selection; epoch and altitude limits. Model domains and clipping are distinct from integration tolerances. 
- `SolarActivity`, `SolarActivityProvider` -> Solar/geomagnetic indices or a callable from UTC datetime to indices. Exponential density is a configurable approximation, not an empirical thermosphere model. 
- `density_exponential`, `density_msis`, `density_dtm2000` -> Earth-fixed positions `(3,)` or `(N,3)`; returns kg/m^3 as scalar or `(N,)`. DTM2000 additionally needs Earth-fixed Sun position and valid altitude (at least 120 km). 
- `RadiationPressure`, `solar_radiation_pressure_acceleration` -> Area, mass, reflection coefficient, and shadow model; Earth-centred satellite/Sun positions in common axes. Returns m/s^2. 
- `conical_shadow_factor` -> Visible fraction of the Sun, including umbra, penumbra and annular geometry. Spherical Sun and Earth. 
- `ThirdBody`, `third_body_acceleration` -> Geocentric body position and gravitational parameter; differential acceleration relative to Earth's centre. 
- `schwarzschild_acceleration` -> Positions and velocities in inertial axes; first post-Newtonian correction in m/s^2. 

## Frames, data, and profiling

`interpolated_earth_orientation(epoch, t0, t1)` builds GCRS-to-ITRS rotations; `EarthOrientation`
transforms positions, states, and force vectors. `t1 > t0` is required when constructing providers,
even for backward integration. `astropy_geocentric_body_provider` returns a direct body-position
callback; `interpolated_astropy_body_providers` constructs bounded callbacks for several bodies.
Build coverage over the entire propagation interval or you may have out-of-range interpolation
`ValueError`s.

`read_icgem_gfc` returns `GravityModel` with contiguous coefficient arrays and model constants.
Temporal records require an evaluation epoch. `download_gravity_model`, `download_space_weather`,
and `download_eop` use the configured cache; downloads replace files atomically, with optional
checksum checking for gravity files. `CelestrakSpaceWeather.from_csv` builds time-dependent indices
and raises when required history is missing. `download_eop` caches a CSV; it does not configure
Astropy's IERS tables automatically.

`PropagationProfile` accumulates nested timings; `EffectTiming` returns calls and elapsed seconds.
Set `profile=True` in propagation to collect them.
