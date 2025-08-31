use crate::fractal::Fractal;

#[derive(Clone, Copy)]
struct Branch {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    depth: u32,
}

pub struct PythagoreanTree {
    branches: Vec<Branch>,
    frontier_depth: u32,
    max_depth: u32,
    epoch: u64,
}

impl PythagoreanTree {
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
            frontier_depth: 0,
            max_depth: 11,
            epoch: 0,
        }
    }
}

impl Fractal for PythagoreanTree {
    fn name(&self) -> &'static str {
        "Pythagorean Tree"
    }

    fn detail_epoch(&self) -> u64 {
        self.epoch
    }

    fn update(&mut self, _dt: f32) {
        if self.branches.is_empty() {
            // Initialize trunk (normalized coordinate space 0..1) bottom center upward
            self.branches.push(Branch {
                x0: 0.5,
                y0: 0.95,
                x1: 0.5,
                y1: 0.65,
                depth: 0,
            });
            self.frontier_depth = 0;
            self.epoch += 1;
            return;
        }
        // Grow next depth layer once per frame until max_depth reached
        if self.frontier_depth < self.max_depth {
            let current_depth = self.frontier_depth;
            let new: Vec<Branch> = self
                .branches
                .iter()
                .filter(|b| b.depth == current_depth)
                .flat_map(|b| {
                    let dx = b.x1 - b.x0;
                    let dy = b.y1 - b.y0; // vector
                                          // Left rotate 45°, right rotate -45°
                    let len = (dx * dx + dy * dy).sqrt();
                    let scale = len * 0.7;
                    let angle_left = (dy).atan2(dx) + std::f32::consts::FRAC_PI_4; // dx,dy angle plus 45°
                    let angle_right = (dy).atan2(dx) - std::f32::consts::FRAC_PI_4;
                    let lx = b.x1 + scale * angle_left.cos();
                    let ly = b.y1 + scale * angle_left.sin();
                    let rx = b.x1 + scale * angle_right.cos();
                    let ry = b.y1 + scale * angle_right.sin();
                    [
                        Branch {
                            x0: b.x1,
                            y0: b.y1,
                            x1: lx,
                            y1: ly,
                            depth: current_depth + 1,
                        },
                        Branch {
                            x0: b.x1,
                            y0: b.y1,
                            x1: rx,
                            y1: ry,
                            depth: current_depth + 1,
                        },
                    ]
                })
                .collect();
            if !new.is_empty() {
                self.branches.extend(new);
                self.frontier_depth += 1;
                self.epoch += 1;
            }
        }
    }

    fn render_partial(&mut self, frame: &mut [u8], width: u32, height: u32, _stride: u32) {
        // Draw over existing image (no clearing) -> we will clear on switching fractal.
        for b in &self.branches {
            let color = depth_color(b.depth, self.max_depth);
            draw_line(
                frame,
                width,
                height,
                (b.x0 * width as f32) as i32,
                (b.y0 * height as f32) as i32,
                (b.x1 * width as f32) as i32,
                (b.y1 * height as f32) as i32,
                color,
            );
        }
    }

    fn info_string(&self) -> String {
        format!("depth {} / {}", self.frontier_depth, self.max_depth)
    }
}

fn depth_color(depth: u32, max: u32) -> [u8; 3] {
    let t = depth as f32 / max.max(1) as f32;
    // simple gradient green to light
    [
        (20.0 + 200.0 * t) as u8,
        (100.0 + 120.0 * t) as u8,
        (20.0 + 100.0 * t) as u8,
    ]
}

fn draw_line(
    frame: &mut [u8],
    width: u32,
    height: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 3],
) {
    let mut x0 = x0;
    let mut y0 = y0;
    let mut x1 = x1;
    let mut y1 = y1;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x0 >= 0 && x0 < width as i32 && y0 >= 0 && y0 < height as i32 {
            let idx = ((y0 as u32 * width + x0 as u32) * 4) as usize;
            frame[idx] = color[0];
            frame[idx + 1] = color[1];
            frame[idx + 2] = color[2];
            frame[idx + 3] = 0xFF;
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}
