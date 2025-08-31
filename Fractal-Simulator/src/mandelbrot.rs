//! Mandelbrot fractal implementation with progressive iteration increase
//! and smooth coloring.

use crate::fractal::Fractal;

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
}

impl Mandelbrot {
    pub fn new() -> Self {
        Self {
            center_x: -0.75,
            center_y: 0.0,
            scale: 3.0,
            base_iterations: 32,
            max_iterations: 4000,
            current_iterations: 32,
            iteration_step: 8,
            epoch: 0,
        }
    }

    pub fn recompute_iteration_step(&mut self) {
        // Dynamic step based on zoom. Smaller scale => deeper zoom => larger step.
        let ratio = (3.0 / self.scale).max(1.0);
        let step = (ratio.log2() * 4.0).ceil() as u32;
        self.iteration_step = step.clamp(2, 512);
    }

    pub fn zoom(&mut self, factor: f64) {
        self.scale *= factor;
        self.scale = self.scale.clamp(1.0e-12, 10.0);
        if factor < 1.0 {
            // Encourage faster refinement when zooming in
            self.current_iterations = (self.current_iterations as f64 * 1.05) as u32;
        }
        self.recompute_iteration_step();
        self.epoch += 1;
    }

    pub fn pan(&mut self, dx: f64, dy: f64) {
        self.center_x += dx * self.scale;
        self.center_y += dy * self.scale;
        self.epoch += 1;
    }

    fn render_with_stride(&self, frame: &mut [u8], width: u32, height: u32, stride: u32) {
        let w_f = width as f64;
        let h_f = height as f64;
        let half_h = self.scale * 0.5;
        let half_w = half_h * (w_f / h_f);
        let min_x = self.center_x - half_w;
        let min_y = self.center_y - half_h;
        let dx = (2.0 * half_w) / w_f;
        let dy = (2.0 * half_h) / h_f;
        let max_iter = self.current_iterations;
        let stride = stride.max(1) as u32;
        let mut y = 0u32;
        while y < height {
            let cy = min_y + y as f64 * dy;
            let mut x = 0u32;
            while x < width {
                let cx = min_x + x as f64 * dx;
                let (iter, zx, zy) = mandelbrot_point(cx, cy, max_iter);
                let idx = ((y * width + x) * 4) as usize;
                let (r, g, b) = smooth_color(iter, max_iter, zx, zy);
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

impl Fractal for Mandelbrot {
    fn name(&self) -> &'static str {
        "Mandelbrot"
    }
    fn detail_epoch(&self) -> u64 {
        self.epoch
    }
    fn update(&mut self, _dt: f32) {
        let before = self.current_iterations;
        if self.current_iterations < self.max_iterations {
            self.current_iterations =
                (self.current_iterations + self.iteration_step).min(self.max_iterations);
        }
        if self.current_iterations != before {
            self.epoch += 1;
        }
    }
    fn render_partial(&mut self, frame: &mut [u8], width: u32, height: u32, stride: u32) {
        self.render_with_stride(frame, width, height, stride);
    }
    fn pan(&mut self, dx: f64, dy: f64) {
        self.pan(dx, dy);
    }
    fn zoom(&mut self, factor: f64) {
        self.zoom(factor);
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
