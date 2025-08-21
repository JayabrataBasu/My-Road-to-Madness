//! Physical constants used throughout the simulation
//!
//! All constants are in SI units unless otherwise specified

// Re-export std PI
pub use std::f64::consts::PI;

// Fundamental constants
pub const C: f64 = 299_792_458.0; // Speed of light (m/s)
pub const G: f64 = 6.674_30e-11; // Gravitational constant (m³/kg⋅s²)
pub const SOLAR_MASS: f64 = 1.988_47e30; // Solar mass (kg)
pub const H: f64 = 6.626_070_15e-34; // Planck constant (J⋅s)
pub const K_B: f64 = 1.380_649e-23; // Boltzmann constant (J/K)

// Mathematical constants
pub const TWO_PI: f64 = 2.0 * PI;
pub const HALF_PI: f64 = PI / 2.0;

// Black hole specific constants
pub const SCHWARZSCHILD_COEFF: f64 = 2.0 * G / (C * C);
// Precomputed: 3 * sqrt(3) ≈ 5.196152422706632
pub const CRITICAL_IMPACT: f64 = 5.196152422706632;
pub const PHOTON_SPHERE_RADIUS: f64 = 1.5;
pub const ISCO_RADIUS: f64 = 3.0;

// Simulation parameters
pub const DEFAULT_BH_MASS_SOLAR: f64 = 10.0;
pub const DEFAULT_TIME_STEP: f64 = 1e-6;
pub const MAX_INTEGRATION_STEPS: usize = 10000;
pub const INTEGRATION_TOLERANCE: f64 = 1e-12;

// Rendering constants
pub const DEFAULT_CAMERA_DISTANCE: f64 = 10.0; // in Schwarzschild radii
pub const DEFAULT_FOV: f64 = PI / 3.0; // 60 degrees
pub const NEAR_PLANE: f64 = 0.1;
pub const FAR_PLANE: f64 = 1000.0;
