//! Physical constants used throughout the simulation
//!
//! All constants are in SI units unless otherwise specified

use std::f64::consts::PI;

// Fundamental constants
/// Speed of light in vacuum (m/s)
pub const C: f64 = 299_792_458.0;

/// Gravitational constant (m³/kg⋅s²)
pub const G: f64 = 6.674_30e-11;

/// Solar mass (kg)
pub const SOLAR_MASS: f64 = 1.988_47e30;

/// Planck constant (J⋅s)
pub const H: f64 = 6.626_070_15e-34;

/// Boltzmann constant (J/K)
pub const K_B: f64 = 1.380_649e-23;

// Mathematical constants
/// Pi
pub const PI: f64 = PI;

/// 2 * Pi
pub const TWO_PI: f64 = 2.0 * PI;

/// Pi / 2
pub const HALF_PI: f64 = PI / 2.0;

// Black hole specific constants
/// Schwarzschild radius coefficient (2G/c²)
pub const SCHWARZSCHILD_COEFF: f64 = 2.0 * G / (C * C);

/// Critical impact parameter for photon capture (in units of Schwarzschild radius)
pub const CRITICAL_IMPACT: f64 = 3.0 * (3.0_f64).sqrt();

/// Photon sphere radius (in units of Schwarzschild radius)
pub const PHOTON_SPHERE_RADIUS: f64 = 1.5;

/// Innermost stable circular orbit (ISCO) radius for Schwarzschild black hole
pub const ISCO_RADIUS: f64 = 3.0;

// Simulation parameters
/// Default black hole mass in solar masses
pub const DEFAULT_BH_MASS_SOLAR: f64 = 10.0;

/// Default simulation time step (s)
pub const DEFAULT_TIME_STEP: f64 = 1e-6;

/// Maximum integration steps for geodesics
pub const MAX_INTEGRATION_STEPS: usize = 10000;

/// Integration tolerance
pub const INTEGRATION_TOLERANCE: f64 = 1e-12;

// Rendering constants
/// Default camera distance from black hole (in Schwarzschild radii)
pub const DEFAULT_CAMERA_DISTANCE: f64 = 10.0;

/// Field of view in radians
pub const DEFAULT_FOV: f64 = PI / 3.0; // 60 degrees

/// Near clipping plane
pub const NEAR_PLANE: f64 = 0.1;

/// Far clipping plane
pub const FAR_PLANE: f64 = 1000.0;
