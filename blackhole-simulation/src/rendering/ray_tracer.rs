use crate::physics::black_hole::BlackHole;
use crate::physics::geodesics::{GeodesicIntegrator, IntegrationMethod, integrate_photon_geodesic};
use crate::rendering::camera::Camera;
use glam::Vec3;
use rayon::prelude::*;
use std::sync::Arc;
#[cfg(feature = "simd-opt")]
use wide::f32x4;

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
    sample_pattern: SamplePattern,
}

#[derive(Clone, Copy)]
pub enum SamplePattern {
    Halton,
    BlueNoise,
    HaltonBlueCombine,
}

// 8x8 Bayer matrix (ordered dithering) normalized to [0,1); serves as a cheap tileable blue-noise-like mask
const BAYER8: [u8; 64] = [
    0, 48, 12, 60, 3, 51, 15, 63, 32, 16, 44, 28, 35, 19, 47, 31, 8, 56, 4, 52, 11, 59, 7, 55, 40,
    24, 36, 20, 43, 27, 39, 23, 2, 50, 14, 62, 1, 49, 13, 61, 34, 18, 46, 30, 33, 17, 45, 29, 10,
    58, 6, 54, 9, 57, 5, 53, 42, 26, 38, 22, 41, 25, 37, 21,
];

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
            sample_pattern: SamplePattern::Halton,
        }
    }

    pub fn set_sample_pattern(&mut self, pattern: SamplePattern) {
        self.sample_pattern = pattern;
    }

    pub fn trace_frame_accumulate(
        &self,
        accum: &mut [[f32; 3]],
        m2: &mut [[f32; 3]],
        framebuffer: &mut [[f32; 4]],
        samples: u32,
        width: u32,
        height: u32,
        center_geod: bool,
        jitter_seed: Option<u64>,
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
        // Welford will use samples directly; no inv needed now.
        accum
            .par_iter_mut()
            .zip(m2.par_iter_mut())
            .zip(framebuffer.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((acc, m2v), fb))| {
                let x = (i as u32) % width;
                let y = (i as u32) / width;
                if y >= height {
                    return;
                }
                let origin = cam_lock.position();
                // Jitter sequence selection (Halton / BlueNoise / Combined)
                let dir = if let Some(seed) = jitter_seed {
                    let h = hash_u64(seed ^ ((i as u64) << 32));
                    let (jx, jy) = match self.sample_pattern {
                        SamplePattern::Halton => {
                            (halton(samples, 2) as f32, halton(samples, 3) as f32)
                        }
                        SamplePattern::BlueNoise => {
                            let bx = (x & 7) as usize;
                            let by = (y & 7) as usize;
                            let idx = by * 8 + bx;
                            // frame rolling to avoid static noise: add frame-based offset and hash scramble
                            let base = BAYER8[idx] as f32 / 64.0;
                            let foff = ((samples % 64) as f32) / 64.0;
                            let sx = ((h & 0xFFFF) as f32) / 65535.0; // small decorrelator
                            let sy = (((h >> 16) & 0xFFFF) as f32) / 65535.0;
                            (
                                (base + foff + sx * 0.25).fract(),
                                (base * 0.73 + foff + sy * 0.25).fract(),
                            )
                        }
                        SamplePattern::HaltonBlueCombine => {
                            let bx = (x & 7) as usize;
                            let by = (y & 7) as usize;
                            let idx = by * 8 + bx;
                            let b = BAYER8[idx] as f32 / 64.0;
                            let h2 = halton(samples, 2) as f32;
                            let h3 = halton(samples, 3) as f32;
                            // blend then scramble
                            let sx = ((h & 0xFFFF) as f32) / 65535.0;
                            let sy = (((h >> 16) & 0xFFFF) as f32) / 65535.0;
                            (
                                ((h2 + b) * 0.5 + sx * 0.25).fract(),
                                ((h3 + b * 0.73) * 0.5 + sy * 0.25).fract(),
                            )
                        }
                    };
                    // additional scramble via xor hash fractional
                    let sx = ((h >> 32) & 0xFFFF) as f32 / 65535.0;
                    let sy = ((h >> 48) & 0xFFFF) as f32 / 65535.0;
                    let (_, d) = cam_lock.screen_to_world_ray_offset(
                        x,
                        y,
                        width,
                        height,
                        (jx + sx * 0.5).fract(),
                        (jy + sy * 0.5).fract(),
                    );
                    d
                } else {
                    cache[i]
                };
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
                // Online mean update (SIMD optional)
                // Welford update per channel (mean in acc, M2 in m2v)
                // previous samples = samples, new count = samples+1
                let n1 = samples as f32; // previous count
                let n = n1 + 1.0;
                // update mean & M2
                for c in 0..3 {
                    let delta = color[c] - acc[c];
                    acc[c] += delta / n;
                    let delta2 = color[c] - acc[c];
                    m2v[c] += delta * delta2; // accumulate M2
                }
                let mapped = self.tonemap(*acc);
                fb[0] = mapped[0];
                fb[1] = mapped[1];
                fb[2] = mapped[2];
                // store average luminance variance (un-tonemapped) in alpha for now for debugging
                let var_r = if samples > 0 {
                    m2v[0] / (samples as f32)
                } else {
                    0.0
                };
                let var_g = if samples > 0 {
                    m2v[1] / (samples as f32)
                } else {
                    0.0
                };
                let var_b = if samples > 0 {
                    m2v[2] / (samples as f32)
                } else {
                    0.0
                };
                let avg_var = (var_r + var_g + var_b) / 3.0;
                fb[3] = (avg_var).min(1.0); // simple pack
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
        // Approximate gravitational redshift based on radius (static Schwarzschild)
        let r = origin.length();
        if r > 2.0 * bh_r {
            let rs = bh_r * 2.0; // since bh_r ~ r_s/2 earlier? ensure scale
            let factor = (1.0 - rs / r.max(rs + 1e-3)).sqrt().max(0.1);
            // shift color towards red & dim
            col = Vec3::new(col.x * 1.0, col.y * factor, col.z * factor * 0.5);
        }
        [col.x, col.y, col.z]
    }

    fn tonemap(&self, c: [f32; 3]) -> [f32; 3] {
        #[cfg(feature = "aces-tonemap")]
        {
            // ACES approximation by Krzysztof Narkowicz
            fn rrt_odt_fit(v: f32) -> f32 {
                let a = v * (2.51 * v + 0.03);
                let b = v * (2.43 * v + 0.59) + 0.14;
                (a / b).clamp(0.0, 1.0)
            }
            return [rrt_odt_fit(c[0]), rrt_odt_fit(c[1]), rrt_odt_fit(c[2])];
        }
        #[allow(unreachable_code)]
        {
            // Fallback Reinhard
            let reinhard = |v: f32| v / (1.0 + v);
            [reinhard(c[0]), reinhard(c[1]), reinhard(c[2])]
        }
    }
}

#[inline]
fn hash_u64(mut x: u64) -> u64 {
    // 64-bit mix (splitmix64)
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[inline]
fn halton(index: u32, base: u32) -> f64 {
    let mut i = index + 1; // start at 1
    let mut f = 1.0;
    let mut r = 0.0;
    let b = base as f64;
    while i > 0 {
        f /= b;
        r += f * (i % base) as f64;
        i /= base;
    }
    r
}
