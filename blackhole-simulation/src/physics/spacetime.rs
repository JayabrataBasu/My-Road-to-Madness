use crate::physics::black_hole::BlackHole;

pub struct MetricTensor {
    pub components: [[f64; 4]; 4],
}

impl MetricTensor {
    /// Schwarzschild metric components at (t, r, theta, phi)
    pub fn schwarzschild(mass: f64, r: f64, theta: f64) -> Self {
        let rs = 2.0 * 6.67430e-11 * mass / (2.99792458e8 * 2.99792458e8);
        let g00 = -(1.0 - rs / r);
        let g11 = 1.0 / (1.0 - rs / r);
        let g22 = r * r;
        let g33 = r * r * theta.sin().powi(2);
        MetricTensor {
            components: [
                [g00, 0.0, 0.0, 0.0],
                [0.0, g11, 0.0, 0.0],
                [0.0, 0.0, g22, 0.0],
                [0.0, 0.0, 0.0, g33],
            ],
        }
    }

    /// Kerr metric components at (t, r, theta, phi)
    pub fn kerr(bh: &BlackHole, r: f64, theta: f64) -> Self {
        let m = 6.67430e-11 * bh.mass / (2.99792458e8 * 2.99792458e8);
        let a = bh.spin * m;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let rho2 = r * r + a * a * cos_theta * cos_theta;
        let delta = r * r - 2.0 * m * r + a * a;

        let g00 = -(1.0 - 2.0 * m * r / rho2);
        let g03 = -2.0 * m * r * a * sin_theta * sin_theta / rho2;
        let g11 = rho2 / delta;
        let g22 = rho2;
        let g33 = (r * r + a * a + 2.0 * m * r * a * a * sin_theta * sin_theta / rho2)
            * sin_theta
            * sin_theta;

        MetricTensor {
            components: [
                [g00, 0.0, 0.0, g03],
                [0.0, g11, 0.0, 0.0],
                [0.0, 0.0, g22, 0.0],
                [g03, 0.0, 0.0, g33],
            ],
        }
    }
}

/// Convert between coordinate systems
pub enum CoordinateSystem {
    Cartesian,
    Schwarzschild,
    BoyerLindquist,
}

/// Convert coordinates between systems
pub fn coordinate_transform(
    coords: [f64; 4],
    from: CoordinateSystem,
    to: CoordinateSystem,
    bh: Option<&BlackHole>,
) -> [f64; 4] {
    // Only basic Schwarzschild <-> Cartesian implemented for brevity
    match (from, to) {
        (CoordinateSystem::Cartesian, CoordinateSystem::Schwarzschild) => {
            let (x, y, z) = (coords[1], coords[2], coords[3]);
            let r = (x * x + y * y + z * z).sqrt();
            let theta = if r == 0.0 { 0.0 } else { (z / r).acos() };
            let phi = y.atan2(x);
            [coords[0], r, theta, phi]
        }
        (CoordinateSystem::Schwarzschild, CoordinateSystem::Cartesian) => {
            let (r, theta, phi) = (coords[1], coords[2], coords[3]);
            let x = r * theta.sin() * phi.cos();
            let y = r * theta.sin() * phi.sin();
            let z = r * theta.cos();
            [coords[0], x, y, z]
        }
        // Boyer-Lindquist <-> Cartesian not implemented
        _ => coords,
    }
}

/// Calculate spacetime interval ds^2 given metric and dx^mu
pub fn spacetime_interval(metric: &MetricTensor, dx: [f64; 4]) -> f64 {
    let mut ds2 = 0.0;
    for mu in 0..4 {
        for nu in 0..4 {
            ds2 += metric.components[mu][nu] * dx[mu] * dx[nu];
        }
    }
    ds2
}

/// Approximate Kretschmann scalar for Schwarzschild (curvature invariant)
pub fn curvature_scalar(mass: f64, r: f64) -> f64 {
    let rs = 2.0 * 6.67430e-11 * mass / (2.99792458e8 * 2.99792458e8);
    48.0 * (rs * rs) / (r * r * r * r)
}

/// Gravitational redshift factor z = (1 - rs/r)^(-1/2) - 1
pub fn redshift_factor(mass: f64, r: f64) -> f64 {
    let rs = 2.0 * 6.67430e-11 * mass / (2.99792458e8 * 2.99792458e8);
    if r <= rs {
        f64::INFINITY
    } else {
        1.0 / (1.0 - rs / r).sqrt() - 1.0
    }
}
