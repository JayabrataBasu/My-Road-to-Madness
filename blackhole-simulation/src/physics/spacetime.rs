use crate::physics::constants::*;
use nalgebra::{Matrix4, Vector3};

/// Coordinate systems used for metric evaluation & transformations.
pub enum CoordinateSystem {
    Cartesian,
    Spherical,
    BoyerLindquist,
}

/// Metric tensor wrapper using nalgebra for determinant/inversion convenience.
pub struct MetricTensor {
    pub components: Matrix4<f64>,
}

impl MetricTensor {
    /// Schwarzschild metric (t,r,theta,phi) with geometric units.
    pub fn schwarzschild(mass_geo: f64, r: f64, theta: f64) -> Self {
        let rs = SCHWARZSCHILD_FACTOR * mass_geo; // 2M
        let f = 1.0 - rs / r;
        let sin_t = theta.sin();
        let diag = [-f, 1.0 / f, r * r, r * r * sin_t * sin_t];
        let mut m = Matrix4::<f64>::zeros();
        for i in 0..4 {
            m[(i, i)] = diag[i];
        }
        Self { components: m }
    }

    /// Kerr metric in Boyer-Lindquist coordinates.
    #[cfg(feature = "kerr")]
    pub fn kerr(mass_geo: f64, a_dim: f64, r: f64, theta: f64) -> Self {
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let sin2 = sin_t * sin_t;
        let rho2 = r * r + a_dim * a_dim * cos_t * cos_t;
        let delta = r * r - 2.0 * mass_geo * r + a_dim * a_dim;
        let g00 = -(1.0 - 2.0 * mass_geo * r / rho2);
        let g03 = -2.0 * mass_geo * r * a_dim * sin2 / rho2;
        let g11 = rho2 / delta;
        let g22 = rho2;
        let g33 = (r * r + a_dim * a_dim + 2.0 * mass_geo * r * a_dim * a_dim * sin2 / rho2) * sin2;
        let mut m = Matrix4::<f64>::zeros();
        m[(0, 0)] = g00;
        m[(0, 3)] = g03;
        m[(3, 0)] = g03;
        m[(1, 1)] = g11;
        m[(2, 2)] = g22;
        m[(3, 3)] = g33;
        Self { components: m }
    }

    /// Determinant of the metric.
    pub fn determinant(&self) -> f64 {
        self.components.determinant()
    }

    /// Inverse metric.
    pub fn inverse(&self) -> Self {
        Self {
            components: self.components.try_inverse().unwrap_or_else(Matrix4::zeros),
        }
    }

    /// Christoffel symbols (very simplified finite-difference placeholder).
    /// Returns Gamma^mu_{nu,sigma} as [[[f64;4];4];4].
    pub fn christoffel_symbols(&self) -> [[[f64; 4]; 4]; 4] {
        // Placeholder zeros; full implementation requires partial derivatives of metric.
        [[[0.0; 4]; 4]; 4]
    }
}

// ---------------- Coordinate Transformations ----------------
pub fn cartesian_to_spherical(v: Vector3<f64>) -> (f64, f64, f64) {
    let r = v.norm();
    if r == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let theta = (v.z / r).acos();
    let phi = v.y.atan2(v.x);
    (r, theta, phi)
}

pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> Vector3<f64> {
    let sin_t = theta.sin();
    Vector3::new(
        r * sin_t * phi.cos(),
        r * sin_t * phi.sin(),
        r * theta.cos(),
    )
}

// Boyer-Lindquist <-> Cartesian (simplified ignoring frame dragging for placeholder conversions)
pub fn cartesian_to_boyer_lindquist(v: Vector3<f64>, _a_dim: f64) -> (f64, f64, f64) {
    cartesian_to_spherical(v) /* TODO refine for spin */
}
pub fn boyer_lindquist_to_cartesian(r: f64, theta: f64, phi: f64, _a_dim: f64) -> Vector3<f64> {
    spherical_to_cartesian(r, theta, phi) /* TODO refine */
}

// ---------------- Scalar Helper Functions ----------------
pub fn spacetime_interval(metric: &MetricTensor, dx: [f64; 4]) -> f64 {
    let mut ds2 = 0.0;
    for mu in 0..4 {
        for nu in 0..4 {
            ds2 += metric.components[(mu, nu)] * dx[mu] * dx[nu];
        }
    }
    ds2
}

pub fn proper_time_factor(metric: &MetricTensor, four_velocity: [f64; 4]) -> f64 {
    let ds2 = spacetime_interval(metric, four_velocity);
    (-ds2).sqrt()
}

pub fn gravitational_redshift(mass_geo: f64, r: f64, _theta: f64, a_opt: Option<f64>) -> f64 {
    if r <= 0.0 {
        return f64::INFINITY;
    }
    #[cfg(feature = "kerr")]
    {
        if let Some(_a_dim) = a_opt {
            // Simplified: use Schwarzschild-like factor as placeholder
            let rs = SCHWARZSCHILD_FACTOR * mass_geo;
            return 1.0 / (1.0 - rs / r).sqrt();
        }
    }
    let rs = SCHWARZSCHILD_FACTOR * mass_geo;
    1.0 / (1.0 - rs / r).sqrt()
}
