// Physical constants (SI units)
const G: f64 = 6.67430e-11; // gravitational constant (m^3 kg^-1 s^-2)
const C: f64 = 2.99792458e8; // speed of light (m/s)
const HBAR: f64 = 1.054571817e-34; // reduced Planck constant (J s)
const KB: f64 = 1.380649e-23; // Boltzmann constant (J/K)
const SIGMA: f64 = 5.670374419e-8; // Stefan-Boltzmann constant (W m^-2 K^-4)
const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct BlackHole {
    pub mass: f64,   // in kg
    pub spin: f64,   // dimensionless (a/M, 0 <= spin < 1)
    pub charge: f64, // in Coulombs (usually 0 for astrophysical BHs)
}

impl BlackHole {
    /// Create a new black hole with mass (kg), spin (0..1), and charge (C)
    pub fn new(mass: f64, spin: f64, charge: f64) -> Self {
        BlackHole { mass, spin, charge }
    }

    /// Schwarzschild radius (event horizon for non-rotating, uncharged BH)
    pub fn schwarzschild_radius(&self) -> f64 {
        2.0 * G * self.mass / (C * C)
    }

    /// Ergosphere radius at equator (Kerr metric, spin in [0,1))
    pub fn ergosphere_radius(&self) -> f64 {
        let m = G * self.mass / (C * C);
        let a = self.spin * m;
        m + (m * m - a * a).sqrt()
    }

    /// Surface gravity at event horizon (Kerr, neglecting charge)
    pub fn surface_gravity(&self) -> f64 {
        let m = G * self.mass / (C * C);
        let a = self.spin * m;
        let r_plus = m + (m * m - a * a).sqrt();
        C * C * (r_plus - m) / (2.0 * m * r_plus)
    }

    /// Hawking temperature (K)
    pub fn temperature(&self) -> f64 {
        let m = G * self.mass / (C * C);
        let a = self.spin * m;
        let r_plus = m + (m * m - a * a).sqrt();
        HBAR * C / (4.0 * PI * KB * r_plus)
    }

    /// Accretion disk luminosity (Eddington limit, rough estimate, in Watts)
    pub fn luminosity(&self) -> f64 {
        // L_Edd = 1.26e31 * (M / M_sun) W
        let m_sun = 1.98847e30;
        1.26e31 * (self.mass / m_sun)
    }
}
