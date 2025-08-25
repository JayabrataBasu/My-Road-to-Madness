use crate::physics::black_hole::BlackHole;
use crate::physics::geodesics::{GeodesicIntegrator, IntegrationMethod, integrate_photon_geodesic};
use crate::rendering::camera::Camera;
use glam::Vec3;
use rayon::prelude::*;
use std::sync::Arc;

pub enum RayTracingQuality {
    Low,
    Medium,
    High,
}

pub struct RayTracer {
    pub quality: RayTracingQuality,
    pub camera: Arc<Camera>,
    pub black_hole: Arc<BlackHole>,
    integrator: GeodesicIntegrator,
    last_camera_version: std::sync::atomic::AtomicU64,
}

impl RayTracer {
    pub fn new(
        quality: RayTracingQuality,
        camera: Arc<Camera>,
        black_hole: Arc<BlackHole>,
    ) -> Self {
        Self {
            quality,
            camera,
            black_hole,
            integrator: GeodesicIntegrator::new(IntegrationMethod::RK4),
            last_camera_version: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn trace_frame(&self, framebuffer: &mut [[f32; 4]], width: u32, height: u32) {
        let cam = &self.camera;
        framebuffer.par_iter_mut().enumerate().for_each(|(i, px)| {
            let x = (i as u32) % width;
            let y = (i as u32) / width;
            if y >= height {
                return;
            }
            let (origin, dir) = cam.screen_to_world_ray(x, y, width, height);
            let color = self.trace_ray(origin, dir);
            let mapped = self.tonemap(color);
            px[0] = mapped[0];
            px[1] = mapped[1];
            px[2] = mapped[2];
            px[3] = 1.0;
        });
    }

    fn trace_ray(&self, origin: Vec3, dir: Vec3) -> [f32; 3] {
        // Quick radial distance to approximate horizon darkening
        let bh_r = self.black_hole.schwarzschild_radius() as f32;
        let to_origin = origin.length();
        let t = 0.5 * (dir.y + 1.0);
        let sky = Vec3::new(0.15, 0.2, 0.45).lerp(Vec3::new(0.9, 0.92, 1.0), t);
        // Very rough lensing hint: if ray nearly points to center, darken
        let focus = dir.normalize().dot((-origin).normalize()).clamp(-1.0, 1.0);
        let lens_factor = (focus.max(0.0)).powf(4.0);
        let mut col = sky * (1.0 - 0.7 * lens_factor);
        // If within ~event horizon radius along direction (cheap test)
        if to_origin < 50.0 * bh_r {
            col *= 0.9;
        }
        [col.x, col.y, col.z]
    }

    fn tonemap(&self, c: [f32; 3]) -> [f32; 3] {
        // Reinhard + gamma 2.2
        let reinhard = |v: f32| v / (1.0 + v);
        let gamma = 1.0 / 2.2;
        [
            reinhard(c[0]).powf(gamma),
            reinhard(c[1]).powf(gamma),
            reinhard(c[2]).powf(gamma),
        ]
    }
}
