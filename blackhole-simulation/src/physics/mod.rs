//! Physics module for black hole simulation
//!
//! This module contains all the scientific calculations and physical
//! modeling for the black hole simulation, including spacetime geometry,
//! geodesics, and gravitational effects.

pub mod black_hole;
pub mod constants;
pub mod geodesics;
pub mod spacetime;

// Re-export commonly used items
pub use black_hole::BlackHole;
pub use constants::*;
pub use geodesics::*;
pub use spacetime::*;

/// Common 3D vector type for physics calculations
pub type Vec3 = nalgebra::Vector3<f64>;

/// Common 4D spacetime vector type
pub type Vec4 = nalgebra::Vector4<f64>;

/// Common matrix type for transformations
pub type Mat4 = nalgebra::Matrix4<f64>;

/// Physical units and conversion factors
#[derive(Debug, Clone, Copy)]
pub struct Units;

impl Units {
    /// Convert from geometric units to SI units
    pub const fn geometric_to_si() -> f64 {
        1.0
    }

    /// Convert solar masses to kilograms
    pub const fn solar_mass_to_kg() -> f64 {
        SOLAR_MASS
    }

    /// Convert Schwarzschild radii to meters
    pub fn schwarzschild_to_meters(rs: f64, mass_solar: f64) -> f64 {
        rs * 2.0 * G * mass_solar * SOLAR_MASS / (C * C)
    }
}
