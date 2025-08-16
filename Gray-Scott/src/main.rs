use eframe::egui;
use egui::{ColorImage, TextureHandle, TextureOptions, Pos2, Vec2};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// Maximum history size for statistics
const MAX_HISTORY: usize = 1000;

/// Presets for different Gray-Scott patterns
#[derive(Debug, Clone)]
struct Preset {
    name: &'static str,
    du: f32,
    dv: f32,
    feed: f32,
    kill: f32,
    dt: f32,
}

impl Preset {
    const PRESETS: [Preset; 4] = [
        Preset {
            name: "Solitons",
            du: 0.16,
            dv: 0.08,
            feed: 0.035,
            kill: 0.065,
            dt: 1.0,
        },
        Preset {
            name: "Worms",
            du: 0.16,
            dv: 0.08,
            feed: 0.054,
            kill: 0.063,
            dt: 1.0,
        },
        Preset {
            name: "Chaos",
            du: 0.16,
            dv: 0.08,
            feed: 0.026,
            kill: 0.051,
            dt: 1.0,
        },
        Preset {
            name: "Custom",
            du: 0.16,
            dv: 0.08,
            feed: 0.055,
            kill: 0.062,
            dt: 1.0,
        },
    ];
}

/// Statistics data point for a single time step
#[derive(Debug, Clone)]
struct StatsPoint {
    time: f32,
    mean_u: f32,
    mean_v: f32,
    total_v: f32,
}

/// Gray-Scott simulation state
struct GrayScottSim {
    width: usize,
    height: usize,
    
    // Double buffering for simulation
    u_current: Vec<f32>,
    v_current: Vec<f32>,
    u_next: Vec<f32>,
    v_next: Vec<f32>,
    
    // Simulation parameters
    du: f32,    // Diffusion rate of U
    dv: f32,    // Diffusion rate of V
    feed: f32,  // Feed rate
    kill: f32,  // Kill rate
    dt: f32,    // Time step
    
    // Simulation state
    running: bool,
    time: f32,
    steps_per_frame: usize,
    
    // Statistics
    stats_history: VecDeque<StatsPoint>,
    
    // UI state
    texture: Option<TextureHandle>,
    mouse_painting: bool,
    paint_radius: f32,
    paint_strength: f32,
    
    // Performance tracking
    last_frame_time: Instant,
    fps: f32,
}

impl GrayScottSim {
    fn new() -> Self {
        let width = 256;
        let height = 256;
        let size = width * height;
        
        let mut sim = Self {
            width,
            height,
            u_current: vec![1.0; size],
            v_current: vec![0.0; size],
            u_next: vec![1.0; size],
            v_next: vec![0.0; size],
            du: 0.16,
            dv: 0.08,
            feed: 0.055,
            kill: 0.062,
            dt: 1.0,
            running: false,
            time: 0.0,
            steps_per_frame: 1,
            stats_history: VecDeque::new(),
            texture: None,
            mouse_painting: false,
            paint_radius: 10.0,
            paint_strength: 1.0,
            last_frame_time: Instant::now(),
            fps: 0.0,
        };
        
        sim.initialize_random();
        sim
    }
    
    /// Initialize the grid with random noise in a small central region
    fn initialize_random(&mut self) {
        let size = self.width * self.height;
        self.u_current = vec![1.0; size];
        self.v_current = vec![0.0; size];
        
        // Add some random V in a small central square
        let cx = self.width / 2;
        let cy = self.height / 2;
        let radius = 20.min(self.width / 8).min(self.height / 8);
        
        for y in (cy.saturating_sub(radius))..=(cy + radius).min(self.height - 1) {
            for x in (cx.saturating_sub(radius))..=(cx + radius).min(self.width - 1) {
                let idx = y * self.width + x;
                if fastrand::f32() < 0.1 {
                    self.v_current[idx] = 0.25 + fastrand::f32() * 0.5;
                    self.u_current[idx] = 0.5 - self.v_current[idx];
                }
            }
        }
        
        self.time = 0.0;
        self.stats_history.clear();
        self.update_statistics();
    }
    
    /// Resize the simulation grid
    fn resize(&mut self, new_width: usize, new_height: usize) {
        if new_width == self.width && new_height == self.height {
            return;
        }
        
        let new_size = new_width * new_height;
        
        // Create new grids
        let mut new_u = vec![1.0; new_size];
        let mut new_v = vec![0.0; new_size];
        
        // Copy existing data with scaling
        let scale_x = self.width as f32 / new_width as f32;
        let scale_y = self.height as f32 / new_height as f32;
        
        for y in 0..new_height {
            for x in 0..new_width {
                let old_x = ((x as f32 * scale_x) as usize).min(self.width - 1);
                let old_y = ((y as f32 * scale_y) as usize).min(self.height - 1);
                let old_idx = old_y * self.width + old_x;
                let new_idx = y * new_width + x;
                
                new_u[new_idx] = self.u_current[old_idx];
                new_v[new_idx] = self.v_current[old_idx];
            }
        }
        
        self.width = new_width;
        self.height = new_height;
        self.u_current = new_u;
        self.v_current = new_v;
        self.u_next = vec![1.0; new_size];
        self.v_next = vec![0.0; new_size];
        
        // Clear texture to force regeneration
        self.texture = None;
    }
    
    /// Apply a preset configuration
    fn apply_preset(&mut self, preset: &Preset) {
        self.du = preset.du;
        self.dv = preset.dv;
        self.feed = preset.feed;
        self.kill = preset.kill;
        self.dt = preset.dt;
    }
    
    /// Get the index for a given coordinate
    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
    
    /// Get wrapped coordinates (periodic boundary conditions)
    #[inline]
    fn wrap_coords(&self, x: i32, y: i32) -> (usize, usize) {
        let wx = ((x % self.width as i32 + self.width as i32) % self.width as i32) as usize;
        let wy = ((y % self.height as i32 + self.height as i32) % self.height as i32) as usize;
        (wx, wy)
    }
    
    /// Compute the Laplacian using a 5-point stencil
    #[inline]
    fn laplacian(&self, grid: &[f32], x: usize, y: usize) -> f32 {
        let center = grid[self.idx(x, y)];
        
        let (x_left, _) = self.wrap_coords(x as i32 - 1, y as i32);
        let (x_right, _) = self.wrap_coords(x as i32 + 1, y as i32);
        let (_, y_up) = self.wrap_coords(x as i32, y as i32 - 1);
        let (_, y_down) = self.wrap_coords(x as i32, y as i32 + 1);
        
        let left = grid[self.idx(x_left, y)];
        let right = grid[self.idx(x_right, y)];
        let up = grid[self.idx(x, y_up)];
        let down = grid[self.idx(x, y_down)];
        
        left + right + up + down - 4.0 * center
    }
    
    /// Perform one simulation step using forward Euler method
    fn step(&mut self) {
        let chunk_size = (self.height / rayon::current_num_threads()).max(1);
        
        // Process rows in parallel
        self.u_next
            .par_chunks_mut(self.width * chunk_size)
            .zip(self.v_next.par_chunks_mut(self.width * chunk_size))
            .enumerate()
            .for_each(|(chunk_idx, (u_chunk, v_chunk))| {
                let start_row = chunk_idx * chunk_size;
                let end_row = (start_row + chunk_size).min(self.height);
                
                for y in start_row..end_row {
                    for x in 0..self.width {
                        let local_idx = (y - start_row) * self.width + x;
                        
                        let u = self.u_current[self.idx(x, y)];
                        let v = self.v_current[self.idx(x, y)];
                        
                        let lapl_u = self.laplacian(&self.u_current, x, y);
                        let lapl_v = self.laplacian(&self.v_current, x, y);
                        
                        let uvv = u * v * v;
                        
                        let du_dt = self.du * lapl_u - uvv + self.feed * (1.0 - u);
                        let dv_dt = self.dv * lapl_v + uvv - (self.kill + self.feed) * v;
                        
                        u_chunk[local_idx] = (u + self.dt * du_dt).clamp(0.0, 1.0);
                        v_chunk[local_idx] = (v + self.dt * dv_dt).clamp(0.0, 1.0);
                    }
                }
            });
        
        // Swap buffers
        std::mem::swap(&mut self.u_current, &mut self.u_next);
        std::mem::swap(&mut self.v_current, &mut self.v_next);
        
        self.time += self.dt;
    }
    
    /// Update simulation for multiple steps
    fn update(&mut self) {
        if self.running {
            for _ in 0..self.steps_per_frame {
                self.step();
            }
            self.update_statistics();
        }
        
        // Update FPS
        let now = Instant::now();
        let frame_time = now.duration_since(self.last_frame_time).as_secs_f32();
        self.fps = 0.9 * self.fps + 0.1 / frame_time;
        self.last_frame_time = now;
    }
    
    /// Calculate and store statistics
    fn update_statistics(&mut self) {
        let total_pixels = self.u_current.len() as f32;
        
        let (sum_u, sum_v, total_v): (f32, f32, f32) = self.u_current
            .par_iter()
            .zip(self.v_current.par_iter())
            .map(|(u, v)| (*u, *v, *v))
            .reduce(
                || (0.0, 0.0, 0.0),
                |acc, val| (acc.0 + val.0, acc.1 + val.1, acc.2 + val.2),
            );
        
        let stats = StatsPoint {
            time: self.time,
            mean_u: sum_u / total_pixels,
            mean_v: sum_v / total_pixels,
            total_v,
        };
        
        self.stats_history.push_back(stats);
        
        if self.stats_history.len() > MAX_HISTORY {
            self.stats_history.pop_front();
        }
    }
    
    /// Create texture from current simulation state
    fn update_texture(&mut self, ctx: &egui::Context) {
        let mut pixels = vec![egui::Color32::BLACK; self.width * self.height];
        
        pixels
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, pixel)| {
                let u = self.u_current[i];
                let v = self.v_current[i];
                
                // Color scheme: U as grayscale background, V as blue intensity
                let gray = (u * 255.0) as u8;
                let blue = (v * 255.0) as u8;
                
                *pixel = egui::Color32::from_rgb(gray.saturating_sub(blue), gray.saturating_sub(blue), gray.saturating_add(blue));
            });
        
        let color_image = ColorImage {
            size: [self.width, self.height],
            pixels,
        };
        
        if let Some(texture) = &mut self.texture {
            texture.set(color_image, TextureOptions::NEAREST);
        } else {
            self.texture = Some(ctx.load_texture("simulation", color_image, TextureOptions::NEAREST));
        }
    }
    
    /// Handle mouse interaction for painting
    fn handle_mouse_painting(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        
        if response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        }
        
        if let Some(pos) = response.interact_pointer_pos() {
            if response.dragged() || response.clicked() {
                // Convert screen coordinates to grid coordinates
                let rel_pos = pos - rect.min;
                let grid_x = (rel_pos.x / rect.width() * self.width as f32) as usize;
                let grid_y = (rel_pos.y / rect.height() * self.height as f32) as usize;
                
                // Paint in a circular area
                let radius = self.paint_radius as usize;
                for dy in -(radius as i32)..=(radius as i32) {
                    for dx in -(radius as i32)..=(radius as i32) {
                        let distance = ((dx * dx + dy * dy) as f32).sqrt();
                        if distance <= self.paint_radius {
                            let (px, py) = self.wrap_coords(grid_x as i32 + dx, grid_y as i32 + dy);
                            let idx = self.idx(px, py);
                            
                            let strength = (1.0 - distance / self.paint_radius) * self.paint_strength;
                            self.v_current[idx] = (self.v_current[idx] + strength * 0.5).clamp(0.0, 1.0);
                            self.u_current[idx] = (self.u_current[idx] - strength * 0.25).clamp(0.0, 1.0);
                        }
                    }
                }
            }
        }
    }
    
    /// Export statistics to CSV file
    fn export_stats(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create("gray_scott_stats.csv")?;
        writeln!(file, "Time,Mean_U,Mean_V,Total_V")?;
        
        for stats in &self.stats_history {
            writeln!(file, "{},{},{},{}", stats.time, stats.mean_u, stats.mean_v, stats.total_v)?;
        }
        
        Ok(())
    }
}

/// Main application structure
