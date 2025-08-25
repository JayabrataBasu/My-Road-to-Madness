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
    pub camera: Arc<parking_lot::RwLock<Camera>>,
    pub black_hole: Arc<BlackHole>,
    integrator: GeodesicIntegrator,
    pub last_camera_version: std::sync::atomic::AtomicU64,
    cached_dims: std::sync::atomic::AtomicU64, // (w<<32)|h
    ray_cache: parking_lot::RwLock<Vec<Vec3>>, // cached directions
}

impl RayTracer {
    pub fn new(
        _quality: RayTracingQuality,
        camera: Arc<parking_lot::RwLock<Camera>>,
        black_hole: Arc<BlackHole>,
    ) -> Self {
        Self {
            camera,
            black_hole,
            integrator: GeodesicIntegrator::new(IntegrationMethod::RK4),
            last_camera_version: std::sync::atomic::AtomicU64::new(0),
            cached_dims: std::sync::atomic::AtomicU64::new(0),
            ray_cache: parking_lot::RwLock::new(Vec::new()),
        }
    }

    pub fn trace_frame_accumulate(
        &self,
        accum: &mut [[f32; 3]],
        framebuffer: &mut [[f32; 4]],
        samples: u32,
        width: u32,
        height: u32,
        center_geod: bool,
    ) {
        let cam_lock = self.camera.read();
        let cam_version = cam_lock.version;
        let packed_dims = ((width as u64) << 32) | height as u64;
        let need_rebuild = self
            .last_camera_version
            .load(std::sync::atomic::Ordering::Relaxed)
            != cam_version
            || self.cached_dims.load(std::sync::atomic::Ordering::Relaxed) != packed_dims;
        if need_rebuild {
            let mut cache = self.ray_cache.write();
            cache.resize((width * height) as usize, Vec3::ZERO);
            for y in 0..height {
                for x in 0..width {
                    let (_, dir) = cam_lock.screen_to_world_ray(x, y, width, height);
                    cache[(y * width + x) as usize] = dir;
                }
            }
            self.last_camera_version
                .store(cam_version, std::sync::atomic::Ordering::Relaxed);
            self.cached_dims
                .store(packed_dims, std::sync::atomic::Ordering::Relaxed);
        }
        let cache = self.ray_cache.read();
        let inv = if samples == 0 {
            1.0
        } else {
            1.0 / (samples as f32 + 1.0)
        };
        accum
            .par_iter_mut()
            .zip(framebuffer.par_iter_mut())
            .enumerate()
            .for_each(|(i, (acc, fb))| {
                let x = (i as u32) % width;
                let y = (i as u32) / width;
                if y >= height {
                    return;
                }
                let origin = cam_lock.position();
                let dir = cache[i];
                let mut color = self.trace_ray(origin, dir);
                if center_geod && x == width / 2 && y == height / 2 {
                    let pos4 = [0.0_f64, origin.x as f64, origin.y as f64, origin.z as f64];
                    let mom4 = [1.0_f64, dir.x as f64, dir.y as f64, dir.z as f64];
                    let geod = integrate_photon_geodesic(
                        pos4,
                        mom4,
                        &self.black_hole,
                        0.01,
                        &self.integrator,
                    );
                    if geod.terminated {
                        color[0] *= 0.8;
                        color[1] *= 0.8;
                        color[2] = (color[2] * 0.8).min(1.0);
                    }
                }
                // Online mean: new_mean = old_mean + (sample - old_mean)/(n+1)
                acc[0] += (color[0] - acc[0]) * inv;
                acc[1] += (color[1] - acc[1]) * inv;
                acc[2] += (color[2] - acc[2]) * inv;
                let mapped = self.tonemap(*acc);
                fb[0] = mapped[0];
                fb[1] = mapped[1];
                fb[2] = mapped[2];
                fb[3] = 1.0;
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
