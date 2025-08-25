use crate::physics::constants::*;
use anyhow::{Result, anyhow};
use nalgebra::Vector3;

/// Representation of an astrophysical black hole.
///
/// Stored mass is in solar masses (mass_solar). All *derived* geometric
/// quantities convert solar masses -> geometric length units (M) using:
/// M (length) = G * M_kg / c^2. In geometric units we set G=c=1 so the
/// dimensionful distinction collapses; we retain conversion for clarity.
#[derive(Debug, Clone)]
pub struct BlackHole {
    mass_solar: f64,        // Mass in solar masses
    spin: f64,              // Dimensionless a* (|a*| <= THORNE_SPIN_LIMIT)
    charge: f64,            // Currently unused (set 0.0 for neutrality)
    position: Vector3<f64>, // Spatial position in geometric units
    accretion_rate: f64,    // Dimensionless placeholder (Eddington fraction)
}

impl BlackHole {
    #[inline]
    pub fn new(
        mass_solar: f64,
        mut spin: f64,
        charge: f64,
        position: Vector3<f64>,
        accretion_rate: f64,
    ) -> Result<Self> {
        if mass_solar <= 0.0 {
            return Err(anyhow!("mass must be > 0"));
        }
        // Clamp spin to physical bounds
        if spin.abs() > THORNE_SPIN_LIMIT {
            spin = spin.signum() * THORNE_SPIN_LIMIT;
        }
        Ok(Self {
            mass_solar,
            spin,
            charge,
            position,
            accretion_rate,
        })
    }

    #[inline]
    pub fn new_schwarzschild(mass_solar: f64) -> Result<Self> {
        Self::new(mass_solar, 0.0, 0.0, Vector3::zeros(), 0.0)
    }
    #[inline]
    pub fn new_kerr(mass_solar: f64, spin: f64) -> Result<Self> {
        Self::new(mass_solar, spin, 0.0, Vector3::zeros(), 0.0)
    }

    // Convenience mass scale constructors (illustrative defaults)
    #[inline]
    pub fn stellar_mass() -> Result<Self> {
        Self::new_schwarzschild(10.0)
    }
    #[inline]
    pub fn intermediate_mass() -> Result<Self> {
        Self::new_schwarzschild(10_000.0)
    }
    #[inline]
    pub fn supermassive() -> Result<Self> {
        Self::new_kerr(4.0e6, 0.5)
    }

    // ---------------- Getters ----------------
    #[inline]
    pub fn mass_solar(&self) -> f64 {
        self.mass_solar
    }
    #[inline]
    pub fn spin(&self) -> f64 {
        self.spin
    }
    #[inline]
    pub fn charge(&self) -> f64 {
        self.charge
    }
    #[inline]
    pub fn position(&self) -> &Vector3<f64> {
        &self.position
    }
    #[inline]
    pub fn accretion_rate(&self) -> f64 {
        self.accretion_rate
    }

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

    /// Photon sphere radius (approx; exact for Schwarzschild). For Kerr this
    /// would depend on inclination & pro/retrograde; we provide Schwarzschild value.
    #[inline]
    pub fn photon_sphere_radius(&self) -> f64 {
        PHOTON_SPHERE_FACTOR * self.mass_geometric() * SCHWARZSCHILD_FACTOR / 2.0
    }

    /// Ergosphere outer boundary at polar angle theta.
    pub fn ergosphere_radius(&self, theta: f64) -> f64 {
        let m = self.mass_geometric();
        #[cfg(feature = "kerr")]
        {
            let a = self.spin * m;
            let cos_t = theta.cos();
            m + (m * m - a * a * cos_t * cos_t).sqrt()
        }
        #[cfg(not(feature = "kerr"))]
        {
            self.event_horizon_radius()
        }
    }

    /// Check if spin is near-extremal (informational).
    #[inline]
    pub fn is_extremal(&self) -> bool {
        (self.spin.abs() - THORNE_SPIN_LIMIT).abs() < 1e-6
    }
}
