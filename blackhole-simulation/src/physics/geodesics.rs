use rayon::prelude::*;
use crate::physics::black_hole::BlackHole;
use crate::physics::constants::*;

// ---------------- Data Types ----------------
pub enum ParticleType { Photon, Massive }
pub enum IntegrationMethod { RK4, AdaptiveRK }

pub struct Geodesic {
    pub positions: Vec<[f64;4]>,
    pub momenta: Vec<[f64;4]>,
    pub terminated: bool,
    pub reason: Option<&'static str>,
}

pub struct GeodesicIntegrator {
    pub method: IntegrationMethod,
    pub step: f64,
    pub min_step: f64,
    pub max_step: f64,
}

impl GeodesicIntegrator {
    pub fn new(method: IntegrationMethod) -> Self { Self { method, step: DEFAULT_TIME_STEP, min_step: DEFAULT_TIME_STEP * 0.01, max_step: DEFAULT_TIME_STEP * 100.0 } }
}

// ---------------- Core Integration Functions (Skeleton) ----------------
pub fn integrate_photon_geodesic(initial_pos:[f64;4], initial_momentum:[f64;4], bh:&BlackHole, max_lambda:f64, integrator:&GeodesicIntegrator) -> Geodesic {
    integrate_generic(initial_pos, initial_momentum, bh, max_lambda, integrator, ParticleType::Photon)
}

pub fn integrate_particle_orbit(initial_pos:[f64;4], initial_momentum:[f64;4], _mass:f64, bh:&BlackHole, max_lambda:f64, integrator:&GeodesicIntegrator) -> Geodesic {
    integrate_generic(initial_pos, initial_momentum, bh, max_lambda, integrator, ParticleType::Massive)
}

fn integrate_generic(pos:[f64;4], mom:[f64;4], bh:&BlackHole, max_lambda:f64, integ:&GeodesicIntegrator, _ptype:ParticleType) -> Geodesic {
    let mut pos = pos;
    let mom = mom;
    let mut geod = Geodesic { positions: Vec::with_capacity(1024), momenta: Vec::with_capacity(1024), terminated:false, reason:None };
    let mut lambda = 0.0;
    let horizon = bh.event_horizon_radius();
    let r_escape = 100.0 * bh.schwarzschild_radius();
    for _step_i in 0..MAX_INTEGRATION_STEPS {
        geod.positions.push(pos); geod.momenta.push(mom);
        let r = (pos[1]*pos[1] + pos[2]*pos[2] + pos[3]*pos[3]).sqrt();
        if r <= horizon { geod.terminated=true; geod.reason=Some("horizon"); break; }
        if r >= r_escape { geod.terminated=true; geod.reason=Some("escape"); break; }
        if lambda >= max_lambda { geod.terminated=true; geod.reason=Some("lambda_limit"); break; }
        if !pos[0].is_finite() || !r.is_finite() { geod.terminated=true; geod.reason=Some("nan"); break; }
        // Very rough linear advance placeholder (replace with RK4 using Christoffel symbols)
        let dt = integ.step;
        for i in 0..4 { pos[i] += mom[i] * dt; }
        lambda += dt;
    }
    if !geod.terminated { geod.terminated=true; geod.reason=Some("step_cap"); }
    geod
}

// ---------------- Utility Functions ----------------
pub fn calculate_orbital_frequency(r:f64, mass:f64, _a:Option<f64>) -> f64 { (mass / (r*r*r)).sqrt() }
pub fn deflection_angle(impact_parameter:f64, bh:&BlackHole) -> f64 { 4.0 * bh.mass_geometric() / impact_parameter }
pub fn calculate_isco_orbit(bh:&BlackHole) -> f64 { 6.0 * bh.mass_geometric() } // Schwarzschild ISCO

pub fn normalize_null_momentum(_p:&mut [f64;4]) { /* placeholder */ }
pub fn normalize_timelike_momentum(_p:&mut [f64;4]) { /* placeholder */ }

// Batch integration skeleton
pub fn integrate_ray_batch(initials:&[[f64;8]], outputs:&mut [Geodesic], bh:&BlackHole, integrator:&GeodesicIntegrator) {
    outputs.par_iter_mut().zip(initials.par_iter()).for_each(|(out, init)| {
        let mut pos = [0.0;4]; let mut mom=[0.0;4];
        pos.copy_from_slice(&init[0..4]); mom.copy_from_slice(&init[4..8]);
        *out = integrate_photon_geodesic(pos, mom, bh, 1.0, integrator);
    });
}

