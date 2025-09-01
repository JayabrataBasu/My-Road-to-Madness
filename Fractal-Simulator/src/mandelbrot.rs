//! Mandelbrot fractal implementation with progressive iteration increase
//! and smooth coloring.

use crate::fractal::Fractal;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
#[cfg(feature = "simd")]
use wide::f32x4;

struct TileRequest {
    tile_x: u32,
    tile_y: u32,
    tx: u32,
    ty: u32,
    tw: u32,
    th: u32,
    min_x: f32,
    min_y: f32,
    dx: f32,
    dy: f32,
    iter: u32,
    scale: f32,
    color_enabled: bool,
    palette: Vec<[u8; 3]>,
}

pub struct Mandelbrot {
    pub center_x: f64,
    pub center_y: f64,
    /// Vertical span of the view in complex plane units.
    pub scale: f64,
    pub base_iterations: u32,
    pub max_iterations: u32,
    pub current_iterations: u32,
    pub iteration_step: u32,
    epoch: u64,
    gradient: Vec<[u8; 3]>,
    pub color_enabled: bool,
    pub tile_opt: bool,
    prev_iteration_step: Option<u32>,
    // animated zoom state
    animating: bool,
    anim_from_cx: f64,
    anim_from_cy: f64,
    anim_from_scale: f64,
    anim_to_cx: f64,
    anim_to_cy: f64,
    anim_to_scale: f64,
    anim_elapsed: f32,
    anim_duration: f32,
    // background tile cache and request channel
    tile_cache: Arc<Mutex<HashMap<(u32, u32, u32), Vec<u8>>>>,
    pending_requests: Arc<Mutex<HashSet<(u32, u32, u32)>>>,
    request_tx: Option<mpsc::Sender<TileRequest>>,
}

impl Mandelbrot {
    pub fn new() -> Self {
        let mut m = Self {
            center_x: -0.75,
            center_y: 0.0,
            scale: 2.8,
            base_iterations: 32,
            max_iterations: 256,
            current_iterations: 32,
            iteration_step: 8,
            epoch: 0,
            gradient: Vec::new(),
            color_enabled: true,
            tile_opt: true,
            prev_iteration_step: None,
            animating: false,
            anim_from_cx: 0.0,
            anim_from_cy: 0.0,
            anim_from_scale: 0.0,
            anim_to_cx: 0.0,
            anim_to_cy: 0.0,
            anim_to_scale: 0.0,
            anim_elapsed: 0.0,
            anim_duration: 0.0,
            tile_cache: Arc::new(Mutex::new(HashMap::new())),
            pending_requests: Arc::new(Mutex::new(HashSet::new())),
            request_tx: None,
        };
        m.spawn_worker();
        m
    }

    fn spawn_worker(&mut self) {
        if self.request_tx.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel::<TileRequest>();
        self.request_tx = Some(tx.clone());
        let cache = Arc::clone(&self.tile_cache);
        let pending = Arc::clone(&self.pending_requests);
        thread::spawn(move || {
            while let Ok(req) = rx.recv() {
                // compute tile into buffer
                let mut buf = vec![0u8; (req.tw * req.th * 4) as usize];
                for py in 0..req.th {
                    let cy = req.min_y + (req.ty + py) as f32 * req.dy;
                    for px in 0..req.tw {
                        let cx = req.min_x + (req.tx + px) as f32 * req.dx;
                        let (iter, zx, zy) = mandelbrot_point_hybrid(cx, cy, req.iter, req.scale);
                        let (r, g, b) = if req.color_enabled {
                            color_from_iter(iter, req.iter, zx, zy, &req.palette)
                        } else {
                            mono_from_iter(iter, req.iter)
                        };
                        let idx = ((py * req.tw + px) * 4) as usize;
                        buf[idx] = r;
                        buf[idx + 1] = g;
                        buf[idx + 2] = b;
                        buf[idx + 3] = 0xFF;
                    }
                }
                let key = (req.tile_x, req.tile_y, req.iter);
                if let Ok(mut c) = cache.lock() {
                    c.insert(key, buf);
                }
                if let Ok(mut p) = pending.lock() {
                    p.remove(&key);
                }
            }
        });
    }

    fn enqueue_tile_request(
        &self,
        tile_x: u32,
        tile_y: u32,
        tw: u32,
        th: u32,
        min_x: f32,
        min_y: f32,
        dx: f32,
        dy: f32,
        iter: u32,
        scale: f32,
        color_enabled: bool,
        palette: Vec<[u8; 3]>,
    ) {
        let key = (tile_x, tile_y, iter);
        {
            let mut p = self.pending_requests.lock().unwrap();
            if p.contains(&key) {
                return;
            }
            p.insert(key);
        }
        if let Some(tx) = &self.request_tx {
            let req = TileRequest {
                tile_x,
                tile_y,
                tx: tile_x * tw,
                ty: tile_y * th,
                tw,
                th,
                min_x,
                min_y,
                dx,
                dy,
                iter,
                scale,
                color_enabled,
                palette,
            };
            let _ = tx.send(req);
        }
    }

    pub fn pixel_to_complex(&self, px: u32, py: u32, width: u32, height: u32) -> (f64, f64) {
        let w_f = width as f64;
        let h_f = height as f64;
        let scale = self.scale;
        let half_h = scale * 0.5;
        let half_w = half_h * (w_f / h_f);
        let min_x = self.center_x - half_w;
        let min_y = self.center_y - half_h;
        let dx = (2.0 * half_w) / w_f;
        let dy = (2.0 * half_h) / h_f;
        let cx = min_x + px as f64 * dx;
        let cy = min_y + py as f64 * dy;
        (cx, cy)
    }

    // zoom_at removed; animated zoom replaces it

    /// Enter fast-zoom interaction mode: lower iterations and increase step to keep UI responsive.
    pub fn fast_zoom_start(&mut self) {
        if self.prev_iteration_step.is_none() {
            self.prev_iteration_step = Some(self.iteration_step);
            // reduce current iterations for quick renders
            self.current_iterations = self.base_iterations.min(self.current_iterations);
            // increase step so iterations ramp faster after interaction
            self.iteration_step = (self.iteration_step.max(2))
                .saturating_mul(8)
                .clamp(2, 1024);
            self.epoch += 1;
        }
    }

    /// End fast zoom and restore iteration stepping behavior.
    pub fn end_fast_zoom(&mut self) {
        if let Some(prev) = self.prev_iteration_step.take() {
            self.iteration_step = prev;
            self.recompute_max_iterations();
            self.epoch += 1;
        }
    }

    pub fn is_in_fast_mode(&self) -> bool {
        self.prev_iteration_step.is_some()
    }

    /// Start an animated zoom to keep the point under (px,py) fixed visually.
    pub fn start_animated_zoom(
        &mut self,
        factor: f64,
        px: u32,
        py: u32,
        width: u32,
        height: u32,
        duration_ms: u32,
    ) {
        let (before_x, before_y) = self.pixel_to_complex(px, py, width, height);
        let to_scale = (self.scale * factor).clamp(1.0e-12, 10.0);
        // compute what center would be after scaling so pixel remains same
        let half_h = to_scale * 0.5;
        let half_w = half_h * (width as f64 / height as f64);
        let min_x = self.center_x - half_w;
        let min_y = self.center_y - half_h;
        let dx = (2.0 * half_w) / (width as f64);
        let dy = (2.0 * half_h) / (height as f64);
        let after_x = min_x + px as f64 * dx;
        let after_y = min_y + py as f64 * dy;
        // setup animation from current to target
        self.animating = true;
        self.anim_from_cx = self.center_x;
        self.anim_from_cy = self.center_y;
        self.anim_from_scale = self.scale;
        self.anim_to_scale = to_scale;
        // adjust target center so the point under cursor stays fixed
        self.anim_to_cx = self.center_x + (before_x - after_x);
        self.anim_to_cy = self.center_y + (before_y - after_y);
        self.anim_elapsed = 0.0;
        self.anim_duration = duration_ms as f32 / 1000.0;
        self.epoch += 1;
    }

    pub fn recompute_iteration_step(&mut self) {
        // Dynamic step based on zoom. Smaller scale => deeper zoom => larger step.
        let ratio = (3.0 / self.scale).max(1.0);
        let step = (ratio.log2() * 4.0).ceil() as u32;
        self.iteration_step = step.clamp(2, 512);
    }

    fn recompute_max_iterations(&mut self) {
        let zoom_factor = (3.0 / self.scale).max(1.0);
        let target = (self.base_iterations as f64 * (zoom_factor.sqrt() * 1.2)).round() as u32;
        let clamped = target.clamp(64, 8000);
        if clamped != self.max_iterations {
            self.max_iterations = clamped;
            if self.color_enabled {
                self.ensure_gradient();
            }
        }
    }

    fn ensure_gradient(&mut self) {
        let needed = (self.max_iterations + 1) as usize;
        if self.gradient.len() != needed {
            self.gradient = (0..needed)
                .map(|i| palette_color(i as f32 / (needed - 1) as f32))
                .collect();
        }
    }

    pub fn toggle_color(&mut self) {
        self.color_enabled = !self.color_enabled;
        if self.color_enabled {
            self.ensure_gradient();
        }
        self.epoch += 1;
    }

    pub fn toggle_tile(&mut self) {
        self.tile_opt = !self.tile_opt;
        self.epoch += 1;
    }

    pub fn zoom(&mut self, factor: f64) {
        self.scale *= factor;
        self.scale = self.scale.clamp(1.0e-12, 10.0);
        if factor < 1.0 {
            // Encourage faster refinement when zooming in
            self.current_iterations = (self.current_iterations as f64 * 1.05) as u32;
        }
        self.recompute_iteration_step();
        self.recompute_max_iterations();
        self.epoch += 1;
    }

    pub fn pan(&mut self, dx: f64, dy: f64) {
        self.center_x += dx * self.scale;
        self.center_y += dy * self.scale;
        self.epoch += 1;
    }

    fn render_with_stride(&self, frame: &mut [u8], width: u32, height: u32, stride: u32) {
        let stride = stride.max(1);
        let w_f = width as f32;
        let h_f = height as f32;
        let scale = self.scale as f32;
        let half_h = scale * 0.5;
        let half_w = half_h * (w_f / h_f);
        let min_x = self.center_x as f32 - half_w;
        let min_y = self.center_y as f32 - half_h;
        let dx = (2.0 * half_w) / w_f;
        let dy = (2.0 * half_h) / h_f;
        let max_iter = self.current_iterations;

        // If final full-res pass and tile optimization enabled, attempt tile renderer.
        if stride == 1 && self.tile_opt {
            self.render_tiles(frame, width, height, min_x, min_y, dx, dy, max_iter, scale);
            return;
        }

        // Parallelize only for stride == 1 to avoid overhead dominating sparse passes.
        if stride == 1 {
            frame
                .par_chunks_exact_mut((width * 4) as usize)
                .enumerate()
                .for_each(|(y, row)| {
                    let y = y as u32;
                    let cy = min_y + y as f32 * dy;
                    let mut x = 0u32;
                    #[cfg(feature = "simd")]
                    {
                        while x + 4 <= width {
                            let xs = f32x4::from([
                                min_x + x as f32 * dx,
                                min_x + (x + 1) as f32 * dx,
                                min_x + (x + 2) as f32 * dx,
                                min_x + (x + 3) as f32 * dx,
                            ]);
                            let (iters, zxs, zys) = mandelbrot_point_simd(xs, cy, max_iter);
                            for lane in 0usize..4usize {
                                let xi = x + lane as u32;
                                let idx = (xi * 4) as usize;
                                let (r, g, b) = if self.color_enabled {
                                    color_from_iter(
                                        iters[lane],
                                        max_iter,
                                        zxs[lane],
                                        zys[lane],
                                        &self.gradient,
                                    )
                                } else {
                                    mono_from_iter(iters[lane], max_iter)
                                };
                                row[idx] = r;
                                row[idx + 1] = g;
                                row[idx + 2] = b;
                                row[idx + 3] = 0xFF;
                            }
                            x += 4;
                        }
                    }
                    // scalar tail
                    while x < width {
                        let cx = min_x + x as f32 * dx;
                        let (iter, zx, zy) = mandelbrot_point_hybrid(cx, cy, max_iter, scale);
                        let idx = (x * 4) as usize;
                        let (r, g, b) = if self.color_enabled {
                            color_from_iter(iter, max_iter, zx, zy, &self.gradient)
                        } else {
                            mono_from_iter(iter, max_iter)
                        };
                        row[idx] = r;
                        row[idx + 1] = g;
                        row[idx + 2] = b;
                        row[idx + 3] = 0xFF;
                        x += 1;
                    }
                });
        } else {
            // Sparse pass (no parallelism needed)
            let mut y = 0u32;
            while y < height {
                let cy = min_y + y as f32 * dy;
                let mut x = 0u32;
                while x < width {
                    let cx = min_x + x as f32 * dx;
                    let (iter, zx, zy) = mandelbrot_point_hybrid(cx, cy, max_iter, scale);
                    let idx = ((y * width + x) * 4) as usize;
                    let (r, g, b) = if self.color_enabled {
                        color_from_iter(iter, max_iter, zx, zy, &self.gradient)
                    } else {
                        mono_from_iter(iter, max_iter)
                    };
                    frame[idx] = r;
                    frame[idx + 1] = g;
                    frame[idx + 2] = b;
                    frame[idx + 3] = 0xFF;
                    x += stride;
                }
                y += stride;
            }
        }
    }

    #[inline(always)]
    fn render_tiles(
        &self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        min_x: f32,
        min_y: f32,
        dx: f32,
        dy: f32,
        max_iter: u32,
        scale: f32,
    ) {
        const TILE: u32 = 8; // tile size
        let w = width;
        let h = height;
        let color_enabled = self.color_enabled;
        for ty in (0..h).step_by(TILE as usize) {
            for tx in (0..w).step_by(TILE as usize) {
                let tw = (TILE).min(w - tx);
                let th = (TILE).min(h - ty);
                // sample corners + center
                let mut samples: [(u32, f32, f32); 5] = [(0, 0.0, 0.0); 5];
                let coords = [
                    (tx, ty),
                    (tx + tw - 1, ty),
                    (tx, ty + th - 1),
                    (tx + tw - 1, ty + th - 1),
                    (tx + tw / 2, ty + th / 2),
                ];
                let mut min_it = max_iter;
                let mut max_it_s = 0u32;
                for (i, (px, py)) in coords.iter().enumerate() {
                    let cx = min_x + *px as f32 * dx;
                    let cy = min_y + *py as f32 * dy;
                    let s = mandelbrot_point_hybrid(cx, cy, max_iter, scale);
                    samples[i] = s;
                    if s.0 < min_it {
                        min_it = s.0;
                    }
                    if s.0 > max_it_s {
                        max_it_s = s.0;
                    }
                }
                // If all inside set -> fill tile (black in color mode, black in mono)
                if min_it >= max_iter {
                    for py in 0..th {
                        for px in 0..tw {
                            let idx = (((ty + py) * w + (tx + px)) * 4) as usize;
                            frame[idx] = 0;
                            frame[idx + 1] = 0;
                            frame[idx + 2] = 0;
                            frame[idx + 3] = 0xFF;
                        }
                    }
                    continue;
                }
                // If variance small -> flat fill
                if max_it_s - min_it <= 2 {
                    let (r, g, b) = if color_enabled {
                        color_from_iter(
                            samples[0].0,
                            max_iter,
                            samples[0].1,
                            samples[0].2,
                            &self.gradient,
                        )
                    } else {
                        mono_from_iter(samples[0].0, max_iter)
                    };
                    for py in 0..th {
                        for px in 0..tw {
                            let idx = (((ty + py) * w + (tx + px)) * 4) as usize;
                            frame[idx] = r;
                            frame[idx + 1] = g;
                            frame[idx + 2] = b;
                            frame[idx + 3] = 0xFF;
                        }
                    }
                    continue;
                }
                // Else per-pixel fallback
                for py in 0..th {
                    let cy = min_y + (ty + py) as f32 * dy;
                    for px in 0..tw {
                        let cx = min_x + (tx + px) as f32 * dx;
                        let (iter, zx, zy) = mandelbrot_point_hybrid(cx, cy, max_iter, scale);
                        let (r, g, b) = if color_enabled {
                            color_from_iter(iter, max_iter, zx, zy, &self.gradient)
                        } else {
                            mono_from_iter(iter, max_iter)
                        };
                        let idx = (((ty + py) * w + (tx + px)) * 4) as usize;
                        frame[idx] = r;
                        frame[idx + 1] = g;
                        frame[idx + 2] = b;
                        frame[idx + 3] = 0xFF;
                    }
                }
            }
        }
    }
}

impl Fractal for Mandelbrot {
    fn name(&self) -> &'static str {
        "Mandelbrot"
    }
    fn detail_epoch(&self) -> u64 {
        self.epoch
    }
    fn update(&mut self, _dt: f32) {
        let before = self.current_iterations;
        // advance animation if active
        if self.animating {
            // clamp dt to frame budget; use a fixed small step for smoothness
            let dt = 1.0 / 60.0;
            self.anim_elapsed += dt;
            let t = (self.anim_elapsed / self.anim_duration).min(1.0);
            // smoothstep easing
            let tt = t * t * (3.0 - 2.0 * t);
            self.center_x = self.anim_from_cx + (self.anim_to_cx - self.anim_from_cx) * tt as f64;
            self.center_y = self.anim_from_cy + (self.anim_to_cy - self.anim_from_cy) * tt as f64;
            self.scale =
                self.anim_from_scale + (self.anim_to_scale - self.anim_from_scale) * tt as f64;
            if t >= 1.0 {
                self.animating = false;
            }
        }
        self.recompute_max_iterations();
        if self.current_iterations < self.max_iterations {
            let remaining = self.max_iterations - self.current_iterations;
            let dynamic_step = self.iteration_step.min(remaining.max(1)).max(2);
            self.current_iterations =
                (self.current_iterations + dynamic_step).min(self.max_iterations);
            if self.color_enabled {
                self.ensure_gradient();
            }
        }
        if self.current_iterations != before {
            self.epoch += 1;
        }
    }
    fn render_partial(&mut self, frame: &mut [u8], width: u32, height: u32, stride: u32) {
        self.render_with_stride(frame, width, height, stride);
    }
    fn pan(&mut self, dx: f64, dy: f64) {
        Mandelbrot::pan(self, dx, dy);
    }
    fn zoom(&mut self, factor: f64) {
        Mandelbrot::zoom(self, factor);
    }
    fn info_string(&self) -> String {
        format!(
            "iters {}/{} step {} scale {:.3e}",
            self.current_iterations, self.max_iterations, self.iteration_step, self.scale
        )
    }
}

fn mandelbrot_point(cx: f64, cy: f64, max_iter: u32) -> (u32, f64, f64) {
    let mut zx = 0.0;
    let mut zy = 0.0;
    let mut iter = 0;
    while zx * zx + zy * zy <= 4.0 && iter < max_iter {
        let xtemp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = xtemp;
        iter += 1;
    }
    (iter, zx, zy)
}

#[inline(always)]
fn mandelbrot_point_f(cx: f32, cy: f32, max_iter: u32) -> (u32, f32, f32) {
    // Cardioid / period-2 bulb quick rejection (in-set -> treat as max_iter)
    // Cardioid: (x - 1/4)^2 + y^2 <= (1/4)^2 - (x - 1/4)
    let x = cx;
    let y = cy;
    let x_minus = x - 0.25;
    let q = x_minus * x_minus + y * y;
    if q * (q + x_minus) <= 0.25 * y * y {
        return (max_iter, 0.0, 0.0);
    }
    // Period-2 bulb: (x+1)^2 + y^2 <= 1/16
    let x_plus = x + 1.0;
    if x_plus * x_plus + y * y <= 0.0625 {
        return (max_iter, 0.0, 0.0);
    }

    let mut zx = 0.0f32;
    let mut zy = 0.0f32;
    let mut iter = 0u32;
    while zx * zx + zy * zy <= 4.0 && iter < max_iter {
        // (zx + i zy)^2 + c
        let zx2 = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = zx2;
        iter += 1;
    }
    (iter, zx, zy)
}

#[cfg(feature = "simd")]
/// Compute escape iterations for four x coordinates at once (same cy)
fn mandelbrot_point_simd(xs: f32x4, cy: f32, max_iter: u32) -> ([u32; 4], [f32; 4], [f32; 4]) {
    let mut zx = f32x4::from(0.0);
    let mut zy = f32x4::from(0.0);
    let cx = xs;
    let mut iter = [0u32; 4];
    for _ in 0..max_iter {
        let zx2 = zx * zx - zy * zy + cx;
        zy = (zx * zy) * f32x4::from(2.0) + f32x4::from(cy);
        zx = zx2;
        let mag2 = zx * zx + zy * zy;
        let mag2_arr = mag2.to_array();
        let mut any_alive = false;
        for lane in 0usize..4usize {
            if mag2_arr[lane] <= 4.0 {
                iter[lane] += 1;
                any_alive = true;
            }
        }
        if !any_alive {
            break;
        }
    }
    let zxs = zx.to_array();
    let zys = zy.to_array();
    (iter, zxs, zys)
}

#[inline(always)]
fn mandelbrot_point_hybrid(cx: f32, cy: f32, max_iter: u32, scale: f32) -> (u32, f32, f32) {
    if scale < 5e-5 {
        // deep zoom, use f64 precision path
        let (it, zx, zy) = mandelbrot_point(cx as f64, cy as f64, max_iter);
        return (it, zx as f32, zy as f32);
    }
    mandelbrot_point_f(cx, cy, max_iter)
}

#[inline(always)]
fn smooth_color_f(iter: u32, max_iter: u32, zx: f32, zy: f32) -> f32 {
    if iter >= max_iter {
        return 0.0;
    }
    let modulus = (zx * zx + zy * zy).sqrt().max(1e-9);
    let nu = (modulus.ln()).ln() / std::f32::consts::LN_2;
    (iter as f32 + 1.0 - nu) / max_iter as f32
}

#[inline(always)]
fn hsv_to_rgb_f(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let h_prime = (h / 60.0) % 6.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let (r1, g1, b1) = if (0.0..1.0).contains(&h_prime) {
        (c, x, 0.0)
    } else if (1.0..2.0).contains(&h_prime) {
        (x, c, 0.0)
    } else if (2.0..3.0).contains(&h_prime) {
        (0.0, c, x)
    } else if (3.0..4.0).contains(&h_prime) {
        (0.0, x, c)
    } else if (4.0..5.0).contains(&h_prime) {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = v - c;
    let (r, g, b) = (r1 + m, g1 + m, b1 + m);
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

#[inline(always)]
fn palette_color(t: f32) -> [u8; 3] {
    let h = 360.0 * t;
    let (r, g, b) = hsv_to_rgb_f(h, 0.9, 0.9_f32.min(0.4 + 0.6 * t));
    [r, g, b]
}

#[inline(always)]
fn color_from_iter(
    iter: u32,
    max_iter: u32,
    zx: f32,
    zy: f32,
    gradient: &[[u8; 3]],
) -> (u8, u8, u8) {
    if iter >= max_iter {
        return (0, 0, 0);
    }
    let t = smooth_color_f(iter, max_iter, zx, zy).clamp(0.0, 1.0);
    let idx = (t * (gradient.len() - 1) as f32) as usize;
    let c = gradient[idx];
    (c[0], c[1], c[2])
}

#[inline(always)]
fn mono_from_iter(iter: u32, max_iter: u32) -> (u8, u8, u8) {
    if iter >= max_iter {
        return (0, 0, 0);
    }
    let v = ((iter as u64 * 255) / max_iter as u64) as u8; // simple grayscale
    (v, v, v)
}
