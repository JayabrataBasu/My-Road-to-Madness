//! Mandelbrot fractal implementation with progressive iteration increase
//! and smooth coloring.

use crate::fractal::Fractal;
use rayon::prelude::*;

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
}

impl Mandelbrot {
    pub fn new() -> Self {
        Self {
            center_x: -0.75,
            center_y: 0.0,
            scale: 2.8,
            base_iterations: 32,
            max_iterations: 256,
            current_iterations: 32,
            iteration_step: 8,
            epoch: 0,
            gradient: Vec::new(),
        }
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
            self.ensure_gradient();
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

        // Parallelize only for stride == 1 to avoid overhead dominating sparse passes.
        if stride == 1 {
            frame
                .par_chunks_exact_mut((width * 4) as usize)
                .enumerate()
                .for_each(|(y, row)| {
                    let y = y as u32;
                    let cy = min_y + y as f32 * dy;
                    let mut cx = min_x;
                    for x in 0..width {
                        let (iter, zx, zy) = mandelbrot_point_hybrid(cx, cy, max_iter, scale);
                        let idx = (x * 4) as usize;
                        let (r, g, b) = color_from_iter(iter, max_iter, zx, zy, &self.gradient);
                        row[idx] = r;
                        row[idx + 1] = g;
                        row[idx + 2] = b;
                        row[idx + 3] = 0xFF;
                        cx += dx;
                    }
                });
        } else {
            // Sparse pass (no parallelism needed)
            let mut y = 0u32;
            while y < height {
                let cy = min_y + y as f32 * dy;
                let mut x = 0u32;
                let mut cx = min_x;
                while x < width {
                    let (iter, zx, zy) =
                        mandelbrot_point_hybrid(cx + x as f32 * dx, cy, max_iter, scale);
                    let idx = ((y * width + x) * 4) as usize;
                    let (r, g, b) = color_from_iter(iter, max_iter, zx, zy, &self.gradient);
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
        self.recompute_max_iterations();
        if self.current_iterations < self.max_iterations {
            let remaining = self.max_iterations - self.current_iterations;
            let dynamic_step = self.iteration_step.min(remaining.max(1)).max(2);
            self.current_iterations =
                (self.current_iterations + dynamic_step).min(self.max_iterations);
            self.ensure_gradient();
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

#[inline(always)]
fn mandelbrot_point_hybrid(cx: f32, cy: f32, max_iter: u32, scale: f32) -> (u32, f32, f32) {
    if scale < 5e-5 {
        // deep zoom, use f64 precision path
        let (it, zx, zy) = mandelbrot_point(cx as f64, cy as f64, max_iter);
        return (it, zx as f32, zy as f32);
    }
    mandelbrot_point_f(cx, cy, max_iter)
}

// Smooth coloring based on normalized iteration count.
fn smooth_color(iter: u32, max_iter: u32, zx: f64, zy: f64) -> (u8, u8, u8) {
    if iter >= max_iter {
        return (0, 0, 0);
    }
    let modulus = (zx * zx + zy * zy).sqrt();
    let nu = (modulus.ln().ln() / std::f64::consts::LN_2).max(0.0);
    let smooth_iter = iter as f64 + 1.0 - nu;
    let t = smooth_iter / max_iter as f64; // 0..1
                                           // Map t through a simple palette (HSV rainbow)
    hsv_to_rgb(360.0 * t, 1.0, (t * 3.0).min(1.0))
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

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
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
