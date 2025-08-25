//! Global constants for the simulation.
//!
//! SI constants are provided for external input conversion. Internal physics
//! adopts geometric units with G = c = 1 once values are normalized.

// ---------------------------------------------------------------------------
// Mathematical Constants
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Physical Constants (SI)
// ---------------------------------------------------------------------------
pub const C_SI: f64 = 299_792_458.0; // Speed of light (m/s)
pub const G_SI: f64 = 6.674_30e-11; // Gravitational constant (m^3 kg^-1 s^-2)
pub const SOLAR_MASS: f64 = 1.988_47e30; // Solar mass (kg)

// ---------------------------------------------------------------------------
// Black Hole Constants / Limits
// ---------------------------------------------------------------------------
pub const THORNE_SPIN_LIMIT: f64 = 0.998; // Practical astrophysical max |a*|
pub const SCHWARZSCHILD_FACTOR: f64 = 2.0; // r_s = 2 M (geometric units)
pub const PHOTON_SPHERE_FACTOR: f64 = 1.5; // r_ph = 1.5 r_s (Schwarzschild)

// ---------------------------------------------------------------------------
// Numerical / Integration Constants
// ---------------------------------------------------------------------------
pub const MAX_INTEGRATION_STEPS: usize = 20_000; // Upper bound on geodesic steps
pub const DEFAULT_TIME_STEP: f64 = 1e-5; // Base step guess for integrators

// ---------------------------------------------------------------------------
// Accretion Disk / Astrophysical Model Constants
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Rendering Constants
// ---------------------------------------------------------------------------
// Rendering-related constants kept minimal (currently handled in renderer)

// ---------------------------------------------------------------------------
// Feature Flags (compile-time usage guidance)
// ---------------------------------------------------------------------------
// (No values; used via Cargo features: "kerr", "debug-physics")
