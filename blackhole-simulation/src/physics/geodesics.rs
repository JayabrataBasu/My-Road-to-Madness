use crate::physics::black_hole::BlackHole;
use crate::physics::constants::*;

// ---------------- Data Types ----------------
pub enum ParticleType {
    Photon,
    #[cfg(feature = "advanced-physics")]
    Massive,
}
pub enum IntegrationMethod {
    RK4,
    #[cfg(feature = "advanced-physics")]
    AdaptiveRK,
}

pub struct Geodesic {
    pub positions: Vec<[f64; 4]>,
    pub momenta: Vec<[f64; 4]>,
    pub terminated: bool,
    pub reason: Option<&'static str>,
}

#[cfg_attr(not(feature = "advanced-physics"), allow(dead_code))]
pub struct GeodesicIntegrator {
    pub method: IntegrationMethod,
    pub step: f64,
    pub min_step: f64,
    pub max_step: f64,
}

impl GeodesicIntegrator {
    pub fn new(method: IntegrationMethod) -> Self {
        Self {
            method,
            step: DEFAULT_TIME_STEP,
            min_step: DEFAULT_TIME_STEP * 0.01,
            max_step: DEFAULT_TIME_STEP * 100.0,
        }
    }
}

// ---------------- Core Integration Functions (Skeleton) ----------------
pub fn integrate_photon_geodesic(
    initial_pos: [f64; 4],
    initial_momentum: [f64; 4],
    bh: &BlackHole,
    max_lambda: f64,
    integrator: &GeodesicIntegrator,
) -> Geodesic {
    integrate_generic(
        initial_pos,
        initial_momentum,
        bh,
        max_lambda,
        integrator,
        ParticleType::Photon,
    )
}

#[cfg(feature = "advanced-physics")]
pub fn integrate_particle_orbit(
    initial_pos: [f64; 4],
    initial_momentum: [f64; 4],
    _mass: f64,
    bh: &BlackHole,
    max_lambda: f64,
    integrator: &GeodesicIntegrator,
) -> Geodesic {
    integrate_generic(
        initial_pos,
        initial_momentum,
        bh,
        max_lambda,
        integrator,
        ParticleType::Massive,
    )
}

fn integrate_generic(
    pos: [f64; 4],
    mom: [f64; 4],
    bh: &BlackHole,
    max_lambda: f64,
    integ: &GeodesicIntegrator,
    _ptype: ParticleType,
) -> Geodesic {
    let mut x = pos; // position 4-vector (t,x,y,z)
    let mut p = mom; // momentum surrogate (dt/dλ, dx/dλ, ...)
    let mut geod = Geodesic {
        positions: Vec::with_capacity(1024),
        momenta: Vec::with_capacity(1024),
        terminated: false,
        reason: None,
    };
    let mut lambda = 0.0;
    let horizon = bh.event_horizon_radius();
    let r_escape = 100.0 * bh.schwarzschild_radius();

    // Simple acceleration approximation: a = - M r_vec / r^3 (Newtonian) for spatial part, keep t component constant.
    let mass = bh.mass_geometric();
    let step = integ.step;

    for _ in 0..MAX_INTEGRATION_STEPS {
        geod.positions.push(x);
        geod.momenta.push(p);
        let r = (x[1] * x[1] + x[2] * x[2] + x[3] * x[3]).sqrt();
        if r <= horizon {
            geod.terminated = true;
            geod.reason = Some("horizon");
            break;
        }
        if r >= r_escape {
            geod.terminated = true;
            geod.reason = Some("escape");
            break;
        }
        if lambda >= max_lambda {
            geod.terminated = true;
            geod.reason = Some("lambda_limit");
            break;
        }
        if !x[0].is_finite() || !r.is_finite() {
            geod.terminated = true;
            geod.reason = Some("nan");
            break;
        }

        let accel = |pos: &[f64; 4]| -> [f64; 4] {
            let rr = (pos[1] * pos[1] + pos[2] * pos[2] + pos[3] * pos[3]).sqrt();
            if rr < 1e-8 {
                return [0.0, 0.0, 0.0, 0.0];
            }
            let inv_r3 = 1.0 / (rr * rr * rr);
            [
                0.0,
                -mass * pos[1] * inv_r3,
                -mass * pos[2] * inv_r3,
                -mass * pos[3] * inv_r3,
            ]
        };

        // RK4 on spatial components only (time evolves linearly by p[0])
        let a1 = accel(&x);
        let x2 = [
            x[0] + 0.5 * step * p[0],
            x[1] + 0.5 * step * p[1],
            x[2] + 0.5 * step * p[2],
            x[3] + 0.5 * step * p[3],
        ];
        let p2 = [
            p[0],
            p[1] + 0.5 * step * a1[1],
            p[2] + 0.5 * step * a1[2],
            p[3] + 0.5 * step * a1[3],
        ];
        let a2 = accel(&x2);
        let x3 = [
            x[0] + 0.5 * step * p2[0],
            x[1] + 0.5 * step * p2[1],
            x[2] + 0.5 * step * p2[2],
            x[3] + 0.5 * step * p2[3],
        ];
        let p3 = [
            p[0],
            p[1] + 0.5 * step * a2[1],
            p[2] + 0.5 * step * a2[2],
            p[3] + 0.5 * step * a2[3],
        ];
        let a3 = accel(&x3);
        let x4 = [
            x[0] + step * p3[0],
            x[1] + step * p3[1],
            x[2] + step * p3[2],
            x[3] + step * p3[3],
        ];
        let p4 = [
            p[0],
            p[1] + step * a3[1],
            p[2] + step * a3[2],
            p[3] + step * a3[3],
        ];
        let a4 = accel(&x4);

        for i in 0..4 {
            x[i] += step * (p[i] + 2.0 * p2[i] + 2.0 * p3[i] + p4[i]) / 6.0;
        }
        // update momentum spatial components with averaged acceleration
        for i in 1..4 {
            p[i] += step * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i]) / 6.0;
        }
        // keep p[0] constant (affine parameterization placeholder)

        lambda += step;
    }
    if !geod.terminated {
        geod.terminated = true;
        geod.reason = Some("step_cap");
    }
    geod
}

// ---------------- Utility Functions ----------------
// (Removed unused helper functions to reduce warnings)
