//! Global constants for the simulation.
//!
//! SI constants are provided for external input conversion. Internal physics
//! adopts geometric units with G = c = 1 once values are normalized.

// ---------------------------------------------------------------------------
// Mathematical Constants
// ---------------------------------------------------------------------------
pub use std::f64::consts::PI; // Re-export for convenience
pub const TWO_PI: f64 = 2.0 * PI;
pub const HALF_PI: f64 = PI / 2.0;

// ---------------------------------------------------------------------------
// Physical Constants (SI)
// ---------------------------------------------------------------------------
pub const C_SI: f64 = 299_792_458.0;          // Speed of light (m/s)
pub const G_SI: f64 = 6.674_30e-11;           // Gravitational constant (m^3 kg^-1 s^-2)
pub const SOLAR_MASS: f64 = 1.988_47e30;      // Solar mass (kg)
pub const H_PLANCK: f64 = 6.626_070_15e-34;   // Planck constant (J s)
pub const K_BOLTZMANN: f64 = 1.380_649e-23;   // Boltzmann constant (J/K)

// ---------------------------------------------------------------------------
// Black Hole Constants / Limits
// ---------------------------------------------------------------------------
pub const THORNE_SPIN_LIMIT: f64 = 0.998;     // Practical astrophysical max |a*|
pub const SCHWARZSCHILD_FACTOR: f64 = 2.0;    // r_s = 2 M (geometric units)
pub const PHOTON_SPHERE_FACTOR: f64 = 1.5;    // r_ph = 1.5 r_s for Schwarzschild
pub const ISCO_FACTOR: f64 = 3.0;             // r_isco = 3 r_s (photon sphere ref) placeholder

// ---------------------------------------------------------------------------
// Numerical / Integration Constants
// ---------------------------------------------------------------------------
pub const EPS: f64 = 1e-14;                   // Generic epsilon for comparisons
pub const MAX_INTEGRATION_STEPS: usize = 20_000; // Upper bound on geodesic steps
pub const INTEGRATION_TOLERANCE: f64 = 1e-10; // Adaptive RK tolerance (placeholder)
pub const DEFAULT_TIME_STEP: f64 = 1e-5;      // Base step guess for integrators
pub const RENDER_MAX_DISTANCE: f64 = 1_000.0; // Cutoff radius in geometric units

// ---------------------------------------------------------------------------
// Accretion Disk / Astrophysical Model Constants
// ---------------------------------------------------------------------------
pub const ACCRETION_DISK_INNER_FACTOR: f64 = 6.0; // r_in = 6 M default (Schwarzschild ISCO)
pub const ACCRETION_DISK_OUTER_FACTOR: f64 = 200.0; // Outer disk extent

// ---------------------------------------------------------------------------
// Rendering Constants
// ---------------------------------------------------------------------------
pub const DEFAULT_CAMERA_DISTANCE: f64 = 30.0; // In M units
pub const DEFAULT_FOV_RADIANS: f64 = PI / 3.0; // 60 degrees
pub const GAMMA_CORRECTION: f32 = 2.2;         // Gamma exponent (f32 for rendering path)
pub const TONEMAP_WHITE: f32 = 1.0;            // White point for Reinhard

// Near/Far planes used for any auxiliary depth-based helpers (physics uses real distances)
pub const NEAR_PLANE: f64 = 0.01;
pub const FAR_PLANE: f64 = 10_000.0;

// ---------------------------------------------------------------------------
// Feature Flags (compile-time usage guidance)
// ---------------------------------------------------------------------------
// (No values; used via Cargo features: "kerr", "debug-physics")

