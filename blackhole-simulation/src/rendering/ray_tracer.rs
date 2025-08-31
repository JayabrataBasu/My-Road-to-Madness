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
    mode: RayMode,
    frame_index: std::sync::atomic::AtomicU64,
}

#[derive(Clone, Copy)]
pub enum SamplePattern {
    Halton,
    BlueNoise,
    HaltonBlueCombine,
}

#[derive(Clone, Copy)]
pub enum RayMode {
    Approximate,
    Geodesic,
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
            mode: RayMode::Approximate,
            frame_index: std::sync::atomic::AtomicU64::new(0),
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
        self.frame_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        match self.mode {
            RayMode::Approximate => self.trace_ray_approx(origin, dir),
            RayMode::Geodesic => self.trace_ray_geodesic(origin, dir),
        }
    }

    fn trace_ray_approx(&self, origin: Vec3, dir: Vec3) -> [f32; 3] {
        // Background sky with subtle lens darkening
        let t = 0.5 * (dir.y + 1.0);
        let mut sky = Vec3::new(0.02, 0.025, 0.04).lerp(Vec3::new(0.25, 0.27, 0.30), t);
        // Add richer starfield BEFORE disk so disk occludes it naturally
        sky += rich_starfield(dir);
        let focus = dir.normalize().dot((-origin).normalize()).clamp(-1.0, 1.0);
        let lens_factor = focus.max(0.0).powf(3.5);
        sky *= 1.0 - 0.45 * lens_factor; // keep some brightness for stars

        // Black hole scale references
        const BH_VISUAL_SCALE: f32 = 3.5e-5;
        let r_s = (self.black_hole.schwarzschild_radius() as f32) * BH_VISUAL_SCALE;
        let photon_sphere = (self.black_hole.photon_sphere_radius() as f32) * BH_VISUAL_SCALE;

        // Simple sphere test (for horizon silhouette)
        let o = origin;
        let d = dir;
        let a = d.length_squared();
        let b = 2.0 * o.dot(d);
        let c = o.length_squared() - r_s * r_s;
        let disc = b * b - 4.0 * a * c;
        let mut hit_t = f32::INFINITY;
        if disc >= 0.0 {
            let sd = disc.sqrt();
            let t0 = (-b - sd) / (2.0 * a);
            let t1 = (-b + sd) / (2.0 * a);
            if t0 > 0.0 {
                hit_t = t0;
            } else if t1 > 0.0 {
                hit_t = t1;
            }
        }

        // Rotating accretion disk (reduced outer radius so we can SEE stars around it)
        let mut disk_col = Vec3::ZERO;
        if dir.y.abs() > 1e-6 {
            let frame = self.frame_index.load(std::sync::atomic::Ordering::Relaxed) as f32;
            let rot = frame * 0.008; // rotation speed
            let sin_a = rot.sin();
            let cos_a = rot.cos();
            let t_plane = -origin.y / dir.y; // plane y=0
            if t_plane > 0.0 && t_plane < hit_t {
                let p = origin + dir * t_plane;
                let mut xz = glam::Vec2::new(p.x, p.z);
                // Rotate texture space for disk animation
                xz = glam::Vec2::new(cos_a * xz.x - sin_a * xz.y, sin_a * xz.x + cos_a * xz.y);
                let r = xz.length();
                let inner = 1.15 * r_s;
                let outer = 140.0 * r_s; // narrower so background shows
                if r > inner && r < outer {
                    // Flared vertical thickness h(r) and ray intersection thickness weight
                    let h_r = (0.15 * r.powf(0.6)).min(30.0 * r_s).max(0.2 * r_s);
                    let y_local = (origin.y + dir.y * t_plane).abs();
                    let thick = (1.0 - (y_local / h_r)).clamp(0.0, 1.0).powf(2.0);
                    if thick > 0.0 {
                        // Emissivity ~ r^-q with soft inner core clamp
                        let q = 2.2;
                        let emiss = ((r / (4.0 * r_s)).max(0.35).powf(-q)).min(25.0) * thick;
                        // Temperature profile (normalized for color ramp)
                        let temp = (outer / r).powf(0.55).min(32.0);
                        let t_norm = (temp / 40.0).clamp(0.0, 1.0);
                        // Color ramp hot (inner) -> cool (outer)
                        let hot = Vec3::new(2.8, 2.4, 2.2);
                        let cool = Vec3::new(1.0, 0.9, 0.55);
                        let base = cool.lerp(hot, t_norm.powf(0.8));
                        // Approx Keplerian velocity & Doppler beaming
                        let beta = (r_s / (2.0 * r)).sqrt().min(0.7);
                        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
                        // Tangential velocity direction (counter-clockwise in rotated frame)
                        let vx = -xz.y / r;
                        let vz = xz.x / r;
                        let vel_dir =
                            Vec3::new(cos_a * vx - sin_a * vz, 0.0, sin_a * vx + cos_a * vz);
                        let view_dir = -dir.normalize();
                        let cos_th = vel_dir.dot(view_dir).clamp(-1.0, 1.0);
                        let doppler = (gamma * (1.0 - beta * cos_th)).max(0.05);
                        let boost = doppler.powf(-3.0);
                        // Saturation fades with radius
                        let sat = (1.0 - (r / outer)).clamp(0.0, 1.0).powf(0.3);
                        let base_sat = Vec3::new(
                            base.x,
                            base.y * (0.5 + 0.5 * sat),
                            base.z * (0.4 + 0.6 * sat),
                        );
                        // Radial fade so edge blends into space
                        let edge = ((outer - r) / (outer - inner)).clamp(0.0, 1.0).powf(1.5);
                        disk_col = base_sat * emiss * boost * 0.018 * edge;
                    }
                }
            }
        }

        // Photon ring heuristic (only if ray misses horizon) broadened for visibility
        let to_center = -o;
        let proj = to_center.dot(d);
        let closest = if proj > 0.0 {
            (to_center - d * (proj / d.length_squared())).length()
        } else {
            o.length()
        };
        let mut ring = 0.0;
        if closest > photon_sphere * 0.93 && closest < photon_sphere * 1.08 && hit_t.is_infinite() {
            let dist = (closest - photon_sphere).abs() / (0.08 * photon_sphere);
            ring = (1.0 - dist).clamp(0.0, 1.0).powf(1.3);
        }

        // Combine components
        let mut col = sky + disk_col + Vec3::new(2.6, 2.3, 1.7) * ring * 0.32;
        if hit_t.is_finite() {
            col = Vec3::splat(ring * 0.02);
        }

        // Gravitational redshift approximation (desaturate & dim green/blue based on camera radius)
        let r_cam = origin.length();
        if r_cam > 2.0 * r_s {
            let g_fac = (1.0 - r_s / r_cam.max(r_s + 1e-3)).sqrt().max(0.05);
            col = Vec3::new(col.x, col.y * g_fac, col.z * g_fac * 0.6);
        }
        [col.x.max(0.0), col.y.max(0.0), col.z.max(0.0)]
    }

    fn trace_ray_geodesic(&self, origin: Vec3, dir: Vec3) -> [f32; 3] {
        const BH_VISUAL_SCALE: f32 = 3.5e-5;
        let inv_scale = 1.0 / BH_VISUAL_SCALE;
        let mass = self.black_hole.mass_geometric() as f32;
        let horizon = self.black_hole.event_horizon_radius() as f32;
        let photon_sphere = self.black_hole.photon_sphere_radius() as f32;
        let o_phys = origin * inv_scale;
        let d_phys = dir.normalize();
        const STEPS: usize = 500;
        let mut x = glam::Vec4::new(0.0, o_phys.x, o_phys.y, o_phys.z);
        let mut p = glam::Vec4::new(1.0, d_phys.x, d_phys.y, d_phys.z);
        let mut min_r = f32::MAX;
        let mut disk_hit = false;
        let mut disk_r = 0.0;
        let mut last_y = x.y;
        for _ in 0..STEPS {
            let r = (x.y * x.y + x.z * x.z + x.w * x.w).sqrt();
            if r < min_r {
                min_r = r;
            }
            if r <= horizon {
                break;
            }
            if r > 300.0 * horizon {
                break;
            }
            if !disk_hit && (last_y > 0.0 && x.y <= 0.0 || last_y < 0.0 && x.y >= 0.0) {
                let rr = (x.z * x.z + x.w * x.w).sqrt();
                if rr > 1.1 * photon_sphere && rr < 60.0 * photon_sphere {
                    disk_hit = true;
                    disk_r = rr;
                }
            }
            last_y = x.y;
            let inv_r3 = if r > 1e-4 { 1.0 / (r * r * r) } else { 0.0 };
            let ax = -mass * x.y * inv_r3;
            let ay = -mass * x.z * inv_r3;
            let az = -mass * x.w * inv_r3;
            x.y += p.y * 0.01;
            x.z += p.z * 0.01;
            x.w += p.w * 0.01;
            p.y += ax * 0.01;
            p.z += ay * 0.01;
            p.w += az * 0.01;
        }
        let mut col = procedural_sky(dir);
        if min_r <= horizon {
            col = Vec3::ZERO;
        } else {
            if min_r > photon_sphere * 0.97 && min_r < photon_sphere * 1.03 {
                let t =
                    1.0 - ((min_r - photon_sphere).abs() / (0.03 * photon_sphere)).clamp(0.0, 1.0);
                col += Vec3::new(2.5, 2.0, 1.2) * t * 0.4;
            }
            if disk_hit {
                let temp = (photon_sphere / disk_r.max(photon_sphere)).powf(0.75);
                let base = blackbody_approx(temp);
                let gfac = (1.0 - (2.0 * mass as f32) / disk_r.max(2.0 * mass as f32 + 1e-3))
                    .sqrt()
                    .max(0.05);
                let shifted = Vec3::new(base.x, base.y * gfac, base.z * gfac * 0.7);
                col = col * 0.5 + shifted * 1.8;
            }
        }
        let focus = dir
            .normalize()
            .dot((-origin).normalize())
            .max(0.0)
            .powf(6.0);
        col *= 0.6 + 0.4 * (1.0 - focus);
        [col.x.max(0.0), col.y.max(0.0), col.z.max(0.0)]
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

fn procedural_sky(dir: Vec3) -> Vec3 {
    let t = 0.5 * (dir.y + 1.0);
    let base = Vec3::new(0.05, 0.07, 0.10).lerp(Vec3::new(0.65, 0.68, 0.72), t);
    let h = hash_u64(
        (dir.x.to_bits() as u64)
            ^ (dir.y.to_bits() as u64).rotate_left(13)
            ^ (dir.z.to_bits() as u64).rotate_left(27),
    );
    let star_sel = (h & 0xFFFF) as f32 / 65535.0;
    let star_density = 0.0025;
    let mut col = base;
    if star_sel < star_density {
        let mag = ((h >> 16) & 0xFF) as f32 / 255.0;
        let tint = Vec3::new(1.0, 0.8 + 0.2 * mag, 0.7 + 0.3 * mag);
        col += tint * (1.5 + 2.5 * (1.0 - star_sel / star_density));
    }
    col
}

// Enhanced multi-layer starfield with varied magnitudes and subtle color tints
fn rich_starfield(dir: Vec3) -> Vec3 {
    // Hash direction to pseudo-random
    let h = hash_u64(
        (dir.x.to_bits() as u64)
            ^ (dir.y.to_bits() as u64).rotate_left(17)
            ^ (dir.z.to_bits() as u64).rotate_left(29),
    );
    // Use several low-probability channels
    let r0 = (h & 0xFFFF) as f32 / 65535.0; // selection
    let r1 = ((h >> 16) & 0xFFFF) as f32 / 65535.0; // magnitude
    let r2 = ((h >> 32) & 0xFFFF) as f32 / 65535.0; // color mix
    let r3 = ((h >> 48) & 0xFFFF) as f32 / 65535.0; // layer

    // Base density
    let base_density = 0.006; // more stars
    if r0 > base_density {
        return Vec3::ZERO;
    }

    // Layer weighting: faint (most), medium, bright, rare super-bright
    let (mag_scale, color_bias, flicker) = if r3 < 0.80 {
        (1.0, 0.4, 0.3)
    } else if r3 < 0.95 {
        (2.2, 0.6, 0.5)
    } else if r3 < 0.995 {
        (4.0, 0.9, 0.8)
    } else {
        (8.0, 1.3, 1.0) // rare bright star
    };

    // Magnitude (brighter for lower r1)
    let mag = (1.0 - r1).powf(2.2) * mag_scale;
    // Temperature-ish tint: mix between warm and cool
    let warm = Vec3::new(1.0, 0.85, 0.7);
    let cool = Vec3::new(0.7, 0.8, 1.0);
    let tint = warm.lerp(cool, r2);
    // Add subtle flicker based on hashed variation (static for now)
    let intensity = 0.8 + 0.2 * flicker;
    tint * mag * intensity * color_bias * 0.5
}

fn blackbody_approx(t: f32) -> Vec3 {
    let x = (t * 3.0).clamp(0.0, 3.0);
    let r = (1.5 * x).min(2.5);
    let g = (x * x + 0.3).min(2.0);
    let b = (0.5 + 0.9 * (1.0 - (x / 3.0))).max(0.1);
    Vec3::new(r, g, b) * 0.4
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
