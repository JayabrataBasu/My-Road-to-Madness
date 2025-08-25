use crate::physics::constants::*;
use anyhow::{Result, anyhow};

/// Representation of an astrophysical black hole.
///
/// Stored mass is in solar masses (mass_solar). All *derived* geometric
/// quantities convert solar masses -> geometric length units (M) using:
/// M (length) = G * M_kg / c^2. In geometric units we set G=c=1 so the
/// dimensionful distinction collapses; we retain conversion for clarity.
#[derive(Debug, Clone)]
pub struct BlackHole {
    mass_solar: f64,
    spin: f64,
}

impl BlackHole {
    #[inline]
    pub fn new(mass_solar: f64, mut spin: f64) -> Result<Self> {
        if mass_solar <= 0.0 {
            return Err(anyhow!("mass must be > 0"));
        }
        // Clamp spin to physical bounds
        if spin.abs() > THORNE_SPIN_LIMIT {
            spin = spin.signum() * THORNE_SPIN_LIMIT;
        }
        Ok(Self { mass_solar, spin })
    }

    #[inline]
    pub fn new_schwarzschild(mass_solar: f64) -> Result<Self> {
        Self::new(mass_solar, 0.0)
    }

    // ---------------- Getters ----------------
    // (mass_solar, spin getters removed as unused)
    // Removed unused getters (charge / position / accretion rate)

    // ---------------- Derived quantities ----------------
    /// Mass in geometric length units (M). Using M = G M_kg / c^2.
    #[inline]
    pub fn mass_geometric(&self) -> f64 {
        let mass_kg = self.mass_solar * SOLAR_MASS;
        G_SI * mass_kg / (C_SI * C_SI)
    }

    /// Schwarzschild radius r_s = 2 M (geometric units) for spin=0 definition.
    #[inline]
    pub fn schwarzschild_radius(&self) -> f64 {
        SCHWARZSCHILD_FACTOR * self.mass_geometric()
    }

    /// Event horizon radius r_+ (Kerr) = M + sqrt(M^2 - a^2).
    #[inline]
    pub fn event_horizon_radius(&self) -> f64 {
        let m = self.mass_geometric();
        #[cfg(feature = "kerr")]
        {
            let a = self.spin * m; // dimensional spin length a = a* M
            if a.abs() >= m {
                return m;
            }
            m + (m * m - a * a).sqrt()
        }
        #[cfg(not(feature = "kerr"))]
        {
            m * SCHWARZSCHILD_FACTOR / 2.0 // fallback: treat as Schwarzschild
        }
    }

    // (photon sphere helper removed)
    // Removed unused photon_sphere_radius / ergosphere_radius / is_extremal helpers
}
