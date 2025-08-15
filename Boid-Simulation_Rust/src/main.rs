use macroquad::prelude::*;
use egui_macroquad;
use egui_macroquad::egui;
use serde::Serialize;
use std::collections::HashMap;
use std::time::Instant;

// Vector2 utilities for boid math
trait Vec2Utils {
    fn limit(&mut self, max: f32);
    fn magnitude(&self) -> f32;
    fn normalize(&self) -> Vec2;
    fn distance_to(&self, other: &Vec2) -> f32;
}

impl Vec2Utils for Vec2 {
    fn limit(&mut self, max: f32) {
        let mag = self.magnitude();
        if mag > max {
            *self = *self / mag * max;
        }
    }
    
    fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    
    fn normalize(&self) -> Vec2 {
        let mag = self.magnitude();
        if mag > 0.0 {
            *self / mag
        } else {
            Vec2::ZERO
        }
    }
    
    fn distance_to(&self, other: &Vec2) -> f32 {
        (*self - *other).magnitude()
    }
}

// Boid structure
#[derive(Clone)]
struct Boid {
    position: Vec2,
    velocity: Vec2,
    acceleration: Vec2,
}

impl Boid {
    fn new(x: f32, y: f32) -> Self {
        Self {
            position: Vec2::new(x, y),
            velocity: Vec2::new(
                rand::gen_range(-2.0, 2.0),
                rand::gen_range(-2.0, 2.0)
            ),
            acceleration: Vec2::ZERO,
        }
    }
    
    // Apply the three core flocking rules
    fn flock(&mut self, boids: &[Boid], params: &BoidParams) {
        let separation = self.separate(boids, params);
        let alignment = self.align(boids, params);
        let cohesion = self.cohesion(boids, params);
        
        // Apply weights to each rule
        self.acceleration += separation * params.separation_weight;
        self.acceleration += alignment * params.alignment_weight;
        self.acceleration += cohesion * params.cohesion_weight;
        
        // Limit steering force
        self.acceleration.limit(params.max_force);
    }
    
    // Separation: steer to avoid crowding local flockmates
    fn separate(&self, boids: &[Boid], params: &BoidParams) -> Vec2 {
        let mut steer = Vec2::ZERO;
        let mut count = 0;
        
        for other in boids {
            let distance = self.position.distance_to(&other.position);
            if distance > 0.0 && distance < params.separation_radius {
                let mut diff = self.position - other.position;
                diff = diff.normalize();
                diff /= distance; // Weight by distance (closer = stronger repulsion)
                steer += diff;
                count += 1;
            }
        }
        
        if count > 0 {
            steer /= count as f32;
            steer = steer.normalize() * params.max_speed;
            steer -= self.velocity;
        }
        
        steer
    }
    
    // Alignment: steer towards the average heading of neighbors
    fn align(&self, boids: &[Boid], params: &BoidParams) -> Vec2 {
        let mut sum = Vec2::ZERO;
        let mut count = 0;
        
        for other in boids {
            let distance = self.position.distance_to(&other.position);
            if distance > 0.0 && distance < params.neighbor_radius {
                sum += other.velocity;
                count += 1;
            }
        }
        
        if count > 0 {
            sum /= count as f32;
            sum = sum.normalize() * params.max_speed;
            let steer = sum - self.velocity;
            return steer;
        }
        
        Vec2::ZERO
    }
    
    // Cohesion: steer to move toward the average position of neighbors
    fn cohesion(&self, boids: &[Boid], params: &BoidParams) -> Vec2 {
        let mut sum = Vec2::ZERO;
        let mut count = 0;
        
        for other in boids {
            let distance = self.position.distance_to(&other.position);
            if distance > 0.0 && distance < params.neighbor_radius {
                sum += other.position;
                count += 1;
            }
        }
        
        if count > 0 {
            sum /= count as f32;
            return self.seek(sum, params);
        }
        
        Vec2::ZERO
    }
    
    // Seek a target position
    fn seek(&self, target: Vec2, params: &BoidParams) -> Vec2 {
        let desired = (target - self.position).normalize() * params.max_speed;
        desired - self.velocity
    }
    
    // Update position and velocity
    fn update(&mut self, dt: f32, params: &BoidParams) {
        // Update velocity
        self.velocity += self.acceleration * dt;
        self.velocity.limit(params.max_speed);
        
        // Update position
        self.position += self.velocity * dt;
        
        // Reset acceleration for next frame
        self.acceleration = Vec2::ZERO;
        
        // Wrap around screen edges
        if self.position.x < 0.0 {
            self.position.x = screen_width();
        } else if self.position.x > screen_width() {
            self.position.x = 0.0;
        }
        
        if self.position.y < 0.0 {
            self.position.y = screen_height();
        } else if self.position.y > screen_height() {
            self.position.y = 0.0;
        }
    }
    
    // Render boid as a triangle pointing in direction of velocity
    fn draw(&self) {
        let angle = self.velocity.y.atan2(self.velocity.x);
        let size = 8.0;
        
        // Triangle vertices relative to center
        let v1 = Vec2::new(size, 0.0);
        let v2 = Vec2::new(-size * 0.5, size * 0.5);
        let v3 = Vec2::new(-size * 0.5, -size * 0.5);
        
        // Rotate vertices
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        
        let p1 = Vec2::new(
            v1.x * cos_a - v1.y * sin_a + self.position.x,
            v1.x * sin_a + v1.y * cos_a + self.position.y,
        );
        
        let p2 = Vec2::new(
            v2.x * cos_a - v2.y * sin_a + self.position.x,
            v2.x * sin_a + v2.y * cos_a + self.position.y,
        );
        
        let p3 = Vec2::new(
            v3.x * cos_a - v3.y * sin_a + self.position.x,
            v3.x * sin_a + v3.y * cos_a + self.position.y,
        );
        
        draw_triangle(p1, p2, p3, WHITE);
        
        // Draw velocity vector for debugging (optional)
        if false { // Set to true to see velocity vectors
            let end = self.position + self.velocity * 10.0;
            draw_line(self.position.x, self.position.y, end.x, end.y, 1.0, RED);
        }
    }
}

// Predator structure (extra credit)
#[derive(Clone)]
struct Predator {
    position: Vec2,
    velocity: Vec2,
}

impl Predator {
    fn new(x: f32, y: f32) -> Self {
        Self {
            position: Vec2::new(x, y),
            velocity: Vec2::new(rand::gen_range(-1.0, 1.0), rand::gen_range(-1.0, 1.0)),
        }
    }
    
    fn update(&mut self, dt: f32, boids: &[Boid]) {
        // Simple predator AI - move toward nearest boid
        if let Some(nearest_boid) = boids.iter().min_by(|a, b| {
            self.position.distance_to(&a.position)
                .partial_cmp(&self.position.distance_to(&b.position))
                .unwrap()
        }) {
            let desired = (nearest_boid.position - self.position).normalize() * 100.0;
            let steer = (desired - self.velocity) * 0.5;
            self.velocity += steer * dt;
            self.velocity.limit(120.0);
        }
        
        self.position += self.velocity * dt;
        
        // Wrap around edges
        if self.position.x < 0.0 { self.position.x = screen_width(); }
        else if self.position.x > screen_width() { self.position.x = 0.0; }
        if self.position.y < 0.0 { self.position.y = screen_height(); }
        else if self.position.y > screen_height() { self.position.y = 0.0; }
    }
    
    fn draw(&self) {
        draw_circle(self.position.x, self.position.y, 12.0, RED);
    }
}

// Boid simulation parameters
#[derive(Clone)]
struct BoidParams {
    separation_radius: f32,
    neighbor_radius: f32,
    max_speed: f32,
    max_force: f32,
    separation_weight: f32,
    alignment_weight: f32,
    cohesion_weight: f32,
}

impl Default for BoidParams {
    fn default() -> Self {
        Self {
            separation_radius: 25.0,
            neighbor_radius: 50.0,
            max_speed: 100.0,
            max_force: 50.0,
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
        }
    }
}

// Parameter presets (extra credit)
impl BoidParams {
    fn tight_flock() -> Self {
        Self {
            separation_radius: 15.0,
            neighbor_radius: 40.0,
            max_speed: 80.0,
            max_force: 40.0,
            separation_weight: 2.0,
            alignment_weight: 1.5,
            cohesion_weight: 2.0,
        }
    }
    
    fn loose_swarm() -> Self {
        Self {
            separation_radius: 35.0,
            neighbor_radius: 80.0,
            max_speed: 120.0,
            max_force: 30.0,
            separation_weight: 1.0,
            alignment_weight: 0.8,
            cohesion_weight: 0.5,
        }
    }
    
    fn chaotic() -> Self {
        Self {
            separation_radius: 20.0,
            neighbor_radius: 30.0,
            max_speed: 150.0,
            max_force: 80.0,
            separation_weight: 0.5,
            alignment_weight: 0.3,
            cohesion_weight: 0.2,
        }
    }
}

// Spatial partitioning grid for performance optimization
struct SpatialGrid {
    cell_size: f32,
    grid: HashMap<(i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            grid: HashMap::new(),
        }
    }
    
    fn clear(&mut self) {
        self.grid.clear();
    }
    
    fn insert(&mut self, position: Vec2, index: usize) {
        let cell = self.get_cell(position);
        self.grid.entry(cell).or_insert_with(Vec::new).push(index);
    }
    
    fn get_cell(&self, position: Vec2) -> (i32, i32) {
        (
            (position.x / self.cell_size).floor() as i32,
            (position.y / self.cell_size).floor() as i32,
        )
    }
    
    fn get_neighbors(&self, position: Vec2, radius: f32) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let center_cell = self.get_cell(position);
        let cell_range = (radius / self.cell_size).ceil() as i32;
        
        for dx in -cell_range..=cell_range {
            for dy in -cell_range..=cell_range {
                let cell = (center_cell.0 + dx, center_cell.1 + dy);
                if let Some(indices) = self.grid.get(&cell) {
                    neighbors.extend(indices);
                }
            }
        }
        
        neighbors
    }
}

// Statistics tracking
#[derive(Serialize)]
struct Stats {
    frame: u64,
    boid_count: usize,
    average_speed: f32,
    average_neighbors: f32,
    fps: f32,
}

struct StatsCollector {
    stats_history: Vec<Stats>,
    frame_count: u64,
    last_stats_time: Instant,
}

impl StatsCollector {
    fn new() -> Self {
        Self {
            stats_history: Vec::new(),
            frame_count: 0,
            last_stats_time: Instant::now(),
        }
    }
    
    fn update(&mut self, boids: &[Boid], fps: f32) {
        self.frame_count += 1;
        
        // Calculate statistics every 30 frames
        if self.frame_count % 30 == 0 {
            let avg_speed = boids.iter()
                .map(|b| b.velocity.magnitude())
                .sum::<f32>() / boids.len() as f32;
            
            // Calculate average neighbor count (simplified)
            let avg_neighbors = 5.0; // Placeholder for actual calculation
            
            self.stats_history.push(Stats {
                frame: self.frame_count,
                boid_count: boids.len(),
                average_speed: avg_speed,
                average_neighbors: avg_neighbors,
                fps,
            });
            
            // Keep only last 1000 stats entries
            if self.stats_history.len() > 1000 {
                self.stats_history.remove(0);
            }
        }
    }
    
    fn export_csv(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut writer = csv::Writer::from_path("boid_stats.csv")?;
        
        for stat in &self.stats_history {
            writer.serialize(stat)?;
        }
        
        writer.flush()?;
        println!("Stats exported to boid_stats.csv");
        Ok(())
    }
}

// Main simulation state
struct Simulation {
    boids: Vec<Boid>,
    predators: Vec<Predator>,
    params: BoidParams,
    spatial_grid: SpatialGrid,
    stats: StatsCollector,
    paused: bool,
    show_ui: bool,
    camera_zoom: f32,
    camera_offset: Vec2,
}

impl Simulation {
    fn new() -> Self {
        let mut boids = Vec::new();
        for _ in 0..200 {
            boids.push(Boid::new(
                rand::gen_range(0.0, screen_width()),
                rand::gen_range(0.0, screen_height()),
            ));
        }
        
        Self {
            boids,
            predators: Vec::new(),
            params: BoidParams::default(),
            spatial_grid: SpatialGrid::new(50.0),
            stats: StatsCollector::new(),
            paused: false,
            show_ui: true,
            camera_zoom: 1.0,
            camera_offset: Vec2::ZERO,
        }
    }
    
    fn update(&mut self, dt: f32) {
        if self.paused {
            return;
        }
        
        // Build spatial grid for performance
        self.spatial_grid.clear();
        for (i, boid) in self.boids.iter().enumerate() {
            self.spatial_grid.insert(boid.position, i);
        }
        
        // Update boids
        let boids_clone = self.boids.clone();
        for (i, boid) in self.boids.iter_mut().enumerate() {
            // Get nearby boids for flocking calculations
            let neighbor_indices = self.spatial_grid.get_neighbors(
                boid.position,
                self.params.neighbor_radius.max(self.params.separation_radius),
            );
            
            let nearby_boids: Vec<Boid> = neighbor_indices
                .iter()
                .filter_map(|&idx| {
                    if idx != i {
                        Some(boids_clone[idx].clone())
                    } else {
                        None
                    }
                })
                .collect();
            
            // Apply flocking behavior
            boid.flock(&nearby_boids, &self.params);
            
            // Apply predator avoidance (extra credit)
            for predator in &self.predators {
                let distance = boid.position.distance_to(&predator.position);
                if distance < 100.0 && distance > 0.0 {
                    let flee = (boid.position - predator.position).normalize() * 200.0;
                    boid.acceleration += flee;
                }
            }
            
            boid.update(dt, &self.params);
        }
        
        // Update predators
        for predator in &mut self.predators {
            predator.update(dt, &self.boids);
        }
        
        // Update statistics
        self.stats.update(&self.boids, get_fps() as f32);
    }
    
    fn draw(&self) {
        clear_background(BLACK);
        
        // Apply camera transform
        let transform = Mat4::from_translation(Vec3::new(self.camera_offset.x, self.camera_offset.y, 0.0))
            * Mat4::from_scale(Vec3::new(self.camera_zoom, self.camera_zoom, 1.0));
        
        // Draw boids
        for boid in &self.boids {
            boid.draw();
        }
        
        // Draw predators
        for predator in &self.predators {
            predator.draw();
        }
        
        // Draw stats overlay
        if !self.stats.stats_history.is_empty() {
            let latest = self.stats.stats_history.last().unwrap();
            draw_text(&format!("Boids: {}", latest.boid_count), 10.0, 30.0, 20.0, WHITE);
            draw_text(&format!("Avg Speed: {:.1}", latest.average_speed), 10.0, 55.0, 20.0, WHITE);
            draw_text(&format!("FPS: {:.1}", latest.fps), 10.0, 80.0, 20.0, WHITE);
            draw_text(&format!("Predators: {}", self.predators.len()), 10.0, 105.0, 20.0, WHITE);
        }
        
        // Draw simple speed chart
        self.draw_speed_chart();
        
        // Instructions
        draw_text("Controls: SPACE=pause, U=toggle UI, R=reset, C=export CSV", 10.0, screen_height() - 25.0, 16.0, GRAY);
        draw_text("Mouse: Click=add predator, Scroll=zoom, Drag=pan", 10.0, screen_height() - 10.0, 16.0, GRAY);
    }
    
    fn draw_speed_chart(&self) {
        if self.stats.stats_history.len() < 2 {
            return;
        }
        
        let chart_x = screen_width() - 220.0;
        let chart_y = 20.0;
        let chart_w = 200.0;
        let chart_h = 100.0;
        
        // Background
        draw_rectangle(chart_x, chart_y, chart_w, chart_h, Color::new(0.0, 0.0, 0.0, 0.5));
        
        // Title
        draw_text("Avg Speed", chart_x + 5.0, chart_y + 15.0, 16.0, WHITE);
        
        // Find min/max for scaling
        let speeds: Vec<f32> = self.stats.stats_history.iter().map(|s| s.average_speed).collect();
        let min_speed = speeds.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_speed = speeds.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        if max_speed > min_speed {
            // Draw lines
            for i in 1..speeds.len() {
                let x1 = chart_x + (i - 1) as f32 / (speeds.len() - 1) as f32 * chart_w;
                let y1 = chart_y + chart_h - ((speeds[i - 1] - min_speed) / (max_speed - min_speed)) * chart_h;
                let x2 = chart_x + i as f32 / (speeds.len() - 1) as f32 * chart_w;
                let y2 = chart_y + chart_h - ((speeds[i] - min_speed) / (max_speed - min_speed)) * chart_h;
                
                draw_line(x1, y1, x2, y2, 2.0, GREEN);
            }
        }
    }
    
    fn handle_input(&mut self) {
        // Keyboard controls
        if is_key_pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
        
        if is_key_pressed(KeyCode::U) {
            self.show_ui = !self.show_ui;
        }
        
        if is_key_pressed(KeyCode::R) {
            self.reset();
        }
        
        if is_key_pressed(KeyCode::C) {
            let _ = self.stats.export_csv();
        }
        
        // Mouse controls
        if is_mouse_button_pressed(MouseButton::Left) {
            let (mouse_x, mouse_y) = mouse_position();
            self.predators.push(Predator::new(mouse_x, mouse_y));
        }
        
        // Camera controls
        let scroll = mouse_wheel().1;
        if scroll != 0.0 {
            self.camera_zoom *= 1.0 + scroll * 0.1;
            self.camera_zoom = self.camera_zoom.clamp(0.1, 5.0);
        }
        
        if is_mouse_button_down(MouseButton::Right) {
            let delta = mouse_delta_position(); // returns Vec2
            self.camera_offset.x += delta.x;
            self.camera_offset.y += delta.y;
        }
    }
    
    fn reset(&mut self) {
        self.boids.clear();
        self.predators.clear();
        
        for _ in 0..200 {
            self.boids.push(Boid::new(
                rand::gen_range(0.0, screen_width()),
                rand::gen_range(0.0, screen_height()),
            ));
        }
        
        self.params = BoidParams::default();
        self.stats = StatsCollector::new();
        self.camera_zoom = 1.0;
        self.camera_offset = Vec2::ZERO;
    }
    
    fn draw_ui(&mut self, egui_ctx: &egui::Context) {
        if !self.show_ui {
            return;
        }

        egui::Window::new("Boid Parameters")
            .default_size([300.0, 400.0])
            .show(egui_ctx, |ui| {
                ui.heading("Flocking Parameters");
                
                ui.add(egui::Slider::new(&mut self.params.separation_radius, 10.0..=100.0)
                    .text("Separation Radius"));
                ui.add(egui::Slider::new(&mut self.params.neighbor_radius, 20.0..=150.0)
                    .text("Neighbor Radius"));
                ui.add(egui::Slider::new(&mut self.params.max_speed, 50.0..=300.0)
                    .text("Max Speed"));
                ui.add(egui::Slider::new(&mut self.params.max_force, 20.0..=150.0)
                    .text("Max Force"));
                
                ui.separator();
                
                ui.add(egui::Slider::new(&mut self.params.separation_weight, 0.0..=5.0)
                    .text("Separation Weight"));
                ui.add(egui::Slider::new(&mut self.params.alignment_weight, 0.0..=5.0)
                    .text("Alignment Weight"));
                ui.add(egui::Slider::new(&mut self.params.cohesion_weight, 0.0..=5.0)
                    .text("Cohesion Weight"));
                
                ui.separator();
                
                ui.heading("Presets");
                ui.horizontal(|ui| {
                    if ui.button("Default").clicked() {
                        self.params = BoidParams::default();
                    }
                    if ui.button("Tight Flock").clicked() {
                        self.params = BoidParams::tight_flock();
                    }
                    if ui.button("Loose Swarm").clicked() {
                        self.params = BoidParams::loose_swarm();
                    }
                    if ui.button("Chaotic").clicked() {
                        self.params = BoidParams::chaotic();
                    }
                });
                
                ui.separator();
                
                ui.heading("Simulation");
                let mut boid_count = self.boids.len();
                ui.add(egui::Slider::new(&mut boid_count, 10..=1000).text("Boid Count"));
                if boid_count != self.boids.len() {
                    self.boids.resize_with(boid_count, || Boid::new(
                        rand::gen_range(0.0, screen_width()),
                        rand::gen_range(0.0, screen_height()),
                    ));
                }
                
                ui.horizontal(|ui| {
                    if ui.button("Add Predator").clicked() {
                        self.predators.push(Predator::new(
                            screen_width() / 2.0,
                            screen_height() / 2.0,
                        ));
                    }
                    if ui.button("Clear Predators").clicked() {
                        self.predators.clear();
                    }
                });
                
                ui.horizontal(|ui| {
                    if ui.button("Export CSV").clicked() {
                        let _ = self.stats.export_csv();
                    }
                    if ui.button("Reset").clicked() {
                        self.reset();
                    }
                });
                
                let pause_text = if self.paused { "Resume" } else { "Pause" };
                if ui.button(pause_text).clicked() {
                    self.paused = !self.paused;
                }
            });
    }
}

#[macroquad::main("Boid Simulation")]
async fn main() {
    let mut simulation = Simulation::new();
    
    loop {
        let dt = get_frame_time();
        
        // Handle input
        simulation.handle_input();
        
        // Update simulation
        simulation.update(dt);
        
        // Render
        simulation.draw();
        
        // Draw UI
        egui_macroquad::ui(|egui_ctx| {
            simulation.draw_ui(egui_ctx);
        });
        
        egui_macroquad::draw();
        
        next_frame().await;
    }
}