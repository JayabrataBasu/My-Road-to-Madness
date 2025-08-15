use macroquad::prelude::*;
use egui_macroquad::egui;
use egui::{DragValue, TextEdit};
use std::io::{self, BufWriter, Write};
use std::fs::{File, OpenOptions};
use serde::Serialize;

#[derive(Clone, Copy, Serialize)]
struct Params {
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
    dt: f64,
    x0: f64,
    y0: f64,
    tol: f64, // for adaptive
}

#[derive(Serialize)]
struct Stats {
    sum_x: f64,
    sum_y: f64,
    sum_x2: f64,
    sum_y2: f64,
    count: usize,
    max_x: f64,
    max_y: f64,
    min_x: f64,
    min_y: f64,
}

impl Stats {
    fn new(x: f64, y: f64) -> Self {
        Stats {
            sum_x: x,
            sum_y: y,
            sum_x2: x * x,
            sum_y2: y * y,
            count: 1,
            max_x: x,
            max_y: y,
            min_x: x,
            min_y: y,
        }
    }

    fn update(&mut self, x: f64, y: f64) {
        self.sum_x += x;
        self.sum_y += y;
        self.sum_x2 += x * x;
        self.sum_y2 += y * y;
        self.count += 1;
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
    }

    fn mean_x(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum_x / self.count as f64 }
    }

    fn mean_y(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum_y / self.count as f64 }
    }

    fn std_x(&self) -> f64 {
        if self.count < 2 { 0.0 } else {
            let mean = self.mean_x();
            ((self.sum_x2 / self.count as f64 - mean * mean) as f64).sqrt()
        }
    }

    fn std_y(&self) -> f64 {
        if self.count < 2 { 0.0 } else {
            let mean = self.mean_y();
            ((self.sum_y2 / self.count as f64 - mean * mean) as f64).sqrt()
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)] // <-- Add Debug here
enum Integrator {
    Euler,
    RK4,
    AdaptiveRK,
}

struct Simulation {
    params: Params,
    edit_params: Params,
    integrator: Integrator,
    current_t: f64,
    x: f64,
    y: f64,
    history: Vec<(f64, f64, f64)>,
    stats: Stats,
    paused: bool,
    show_params: bool,
    show_help: bool,
    zoom_level: f64,
    csv_filename: String,
    append_csv: bool,
    png_filename: String,
    json_filename: String,
    error_msg: String,
}

impl Simulation {
    fn new(params: Params) -> Self {
        let x = params.x0;
        let y = params.y0;
        let history = vec![(0.0, x, y)];
        let stats = Stats::new(x, y);
        Simulation {
            params,
            edit_params: params,
            integrator: Integrator::RK4,
            current_t: 0.0,
            x,
            y,
            history,
            stats,
            paused: false,
            show_params: false,
            show_help: false,
            zoom_level: 1.0,
            csv_filename: "lotka_volterra.csv".to_string(),
            append_csv: false,
            png_filename: "lotka_volterra.png".to_string(),
            json_filename: "lotka_volterra.json".to_string(),
            error_msg: String::new(),
        }
    }

    fn reset(&mut self) {
        self.current_t = 0.0;
        self.x = self.params.x0;
        self.y = self.params.y0;
        self.history = vec![(0.0, self.x, self.y)];
        self.stats = Stats::new(self.x, self.y);
    }

    fn reset_stats(&mut self) {
        self.stats = Stats::new(0.0, 0.0); // or recompute from history, but for simplicity reset to zero
    }

    fn update(&mut self) {
        if self.paused {
            return;
        }
        let dt = self.params.dt;
        let new_state = match self.integrator {
            Integrator::Euler => euler_step(&self.params, [self.x, self.y], dt),
            Integrator::RK4 => rk4_step(&self.params, self.current_t, [self.x, self.y], dt),
            Integrator::AdaptiveRK => adaptive_rk_step(&self.params, self.current_t, [self.x, self.y]),
        };
        self.current_t += dt; // for adaptive, actual dt may vary, but for simplicity use fixed
        self.x = new_state[0];
        self.y = new_state[1];
        self.history.push((self.current_t, self.x, self.y));
        self.stats.update(self.x, self.y);
        const MAX_HISTORY: usize = 10000;
        if self.history.len() > MAX_HISTORY {
            self.history.remove(0);
        }
    }

    fn export_csv(&mut self) -> io::Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(self.append_csv)
            .truncate(!self.append_csv)
            .open(&self.csv_filename)?;
        let mut writer = BufWriter::new(file);
        if !self.append_csv {
            writeln!(writer, "time,prey,predator")?;
        }
        for &(t, x, y) in &self.history {
            writeln!(writer, "{},{},{}", t, x, y)?;
        }
        Ok(())
    }

    fn export_json(&self) -> io::Result<()> {
        let data = serde_json::json!({
            "params": self.params,
            "stats": self.stats,
        });
        let file = File::create(&self.json_filename)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &data)?;
        Ok(())
    }

    fn export_png(&self) -> Result<(), image::ImageError> {
        let data = get_screen_data();
        let pixels = data.bytes; // <-- Use .bytes instead of .export()
        let img = image::RgbaImage::from_vec(data.width as u32, data.height as u32, pixels.to_vec()).unwrap();
        img.save(&self.png_filename)?;
        Ok(())
    }
}

// Equations
fn lotka_volterra(params: &Params, state: [f64; 2]) -> [f64; 2] {
    let [x, y] = state;
    [params.alpha * x - params.beta * x * y, params.delta * x * y - params.gamma * y]
}

// Euler step
fn euler_step(params: &Params, state: [f64; 2], dt: f64) -> [f64; 2] {
    let k = lotka_volterra(params, state);
    [state[0] + dt * k[0], state[1] + dt * k[1]]
}

// RK4 step (same as before)
fn rk4_step(params: &Params, _t: f64, state: [f64; 2], dt: f64) -> [f64; 2] {
    let f = |s: [f64; 2]| lotka_volterra(params, s);
    let k1 = f(state);
    let k2 = f([state[0] + dt / 2.0 * k1[0], state[1] + dt / 2.0 * k1[1]]);
    let k3 = f([state[0] + dt / 2.0 * k2[0], state[1] + dt / 2.0 * k2[1]]);
    let k4 = f([state[0] + dt * k3[0], state[1] + dt * k3[1]]);
    [
        state[0] + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        state[1] + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
    ]
}

// Simple adaptive RK: use RK4 and RK3 for error estimate
fn adaptive_rk_step(params: &Params, t: f64, state: [f64; 2]) -> [f64; 2] {
    let mut dt = params.dt;
    loop {
        let rk4 = rk4_step(params, t, state, dt);
        // RK3 for estimate
        let f = |s: [f64; 2]| lotka_volterra(params, s);
        let k1 = f(state);
        let k2 = f([state[0] + dt / 2.0 * k1[0], state[1] + dt / 2.0 * k1[1]]);
        let k3 = f([state[0] + dt * ( -k2[0] + 2.0 * k2[0]), state[1] + dt * ( -k2[1] + 2.0 * k2[1])]); // simplified
        let rk3 = [
            state[0] + dt / 6.0 * (k1[0] + 4.0 * k2[0] + k3[0]),
            state[1] + dt / 6.0 * (k1[1] + 4.0 * k2[1] + k3[1]),
        ];
        let err = ((rk4[0] - rk3[0]).abs() + (rk4[1] - rk3[1]).abs()) / 2.0;
        if err < params.tol {
            return rk4;
        }
        dt *= 0.9 * (params.tol / err).powf(0.25);
        if dt < 1e-6 {
            return rk4; // avoid too small
        }
    }
}

// Rendering
fn draw_time_series(sim: &Simulation, left: f32, bottom: f32, width: f32, height: f32, zoom_level: f64) {
    if sim.history.is_empty() || sim.current_t == 0.0 {
        return;
    }
    let actual_max = sim.history.iter().fold(0.0_f64, |acc, &(_, x, y)| acc.max(x).max(y)); // <-- Specify type
    let display_max = (actual_max / zoom_level).max(1.0);

    // Grid and ticks
    for i in 1..10 {
        let gx = left + i as f32 * width / 10.0;
        draw_line(gx, bottom, gx, bottom - height, 0.5, GRAY);
        let gy = bottom - i as f32 * height / 10.0;
        draw_line(left, gy, left + width, gy, 0.5, GRAY);
    }

    // Axes
    draw_line(left, bottom, left + width, bottom, 1.0, BLACK);
    draw_line(left, bottom, left, bottom - height, 1.0, BLACK);

    // Tick labels
    for i in 0..=10 {
        let tx = sim.current_t * i as f64 / 10.0;
        draw_text(&format!("{:.1}", tx), left + i as f32 * width / 10.0 - 10.0, bottom + 15.0, 14.0, BLACK);
        let py = display_max * i as f64 / 10.0;
        draw_text(&format!("{:.1}", py), left - 30.0, bottom - i as f32 * height / 10.0 + 5.0, 14.0, BLACK);
    }

    // Labels and legends (same)
    draw_text("Time", left + width / 2.0, bottom + 30.0, 20.0, BLACK);
    draw_text("Population", left - 60.0, bottom - height / 2.0, 20.0, BLACK);
    draw_text("Prey", left + 10.0, bottom - height + 20.0, 18.0, BLUE);
    draw_text("Predator", left + 10.0, bottom - height + 40.0, 18.0, RED);

    // Plot
    let mut prev_sx = None;
    let mut prev_sy_x = None;
    let mut prev_sy_y = None;
    for &(t, x, y) in &sim.history {
        let sx = left + ((t / sim.current_t) as f32 * width);
        let sy_x = bottom - ((x / display_max) as f32 * height);
        let sy_y = bottom - ((y / display_max) as f32 * height);
        if let (Some(psx), Some(psy_x), Some(psy_y)) = (prev_sx, prev_sy_x, prev_sy_y) {
            draw_line(psx, psy_x, sx, sy_x, 2.0, BLUE);
            draw_line(psx, psy_y, sx, sy_y, 2.0, RED);
        }
        prev_sx = Some(sx);
        prev_sy_x = Some(sy_x);
        prev_sy_y = Some(sy_y);
    }
}

fn draw_phase_portrait(sim: &Simulation, left: f32, bottom: f32, width: f32, height: f32, zoom_level: f64) {
    if sim.history.is_empty() {
        return;
    }
    let actual_max = sim.history.iter().fold(0.0_f64, |acc, &(_, x, y)| acc.max(x).max(y)); // <-- Specify type
    let display_max = (actual_max / zoom_level).max(1.0);

    // Grid and ticks
    for i in 1..10 {
        let gx = left + i as f32 * width / 10.0;
        draw_line(gx, bottom, gx, bottom - height, 0.5, GRAY);
        let gy = bottom - i as f32 * height / 10.0;
        draw_line(left, gy, left + width, gy, 0.5, GRAY);
    }

    // Axes
    draw_line(left, bottom, left + width, bottom, 1.0, BLACK);
    draw_line(left, bottom, left, bottom - height, 1.0, BLACK);

    // Tick labels
    for i in 0..=10 {
        let px = display_max * i as f64 / 10.0;
        draw_text(&format!("{:.1}", px), left + i as f32 * width / 10.0 - 10.0, bottom + 15.0, 14.0, BLACK);
        let py = display_max * i as f64 / 10.0;
        draw_text(&format!("{:.1}", py), left - 30.0, bottom - i as f32 * height / 10.0 + 5.0, 14.0, BLACK);
    }

    // Labels
    draw_text("Prey", left + width / 2.0, bottom + 30.0, 20.0, BLACK);
    draw_text("Predator", left - 60.0, bottom - height / 2.0, 20.0, BLACK);

    // Plot
    let mut prev_sx = None;
    let mut prev_sy = None;
    for &(_, x, y) in &sim.history {
        let sx = left + ((x / display_max) as f32 * width);
        let sy = bottom - ((y / display_max) as f32 * height);
        if let (Some(psx), Some(psy)) = (prev_sx, prev_sy) {
            draw_line(psx, psy, sx, sy, 2.0, GREEN);
        }
        prev_sx = Some(sx);
        prev_sy = Some(sy);
    }
}

#[macroquad::main("Lotka-Volterra Simulator")]
async fn main() {
    let params = Params {
        alpha: 1.5,
        beta: 0.02,
        delta: 0.01,
        gamma: 0.5,
        dt: 0.01,
        x0: 100.0,
        y0: 15.0,
        tol: 1e-4,
    };
    let mut sim = Simulation::new(params);

    loop {
        // Keyboard inputs (reduced, as many moved to UI)
        if is_key_pressed(KeyCode::Space) {
            sim.paused = !sim.paused;
        }
        if is_key_pressed(KeyCode::R) {
            sim.reset();
        }
        if is_key_pressed(KeyCode::H) {
            sim.show_help = !sim.show_help;
        }
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        // Mouse wheel for zoom
        let (_, wheel_y) = mouse_wheel();
        if wheel_y != 0.0 {
            sim.zoom_level += wheel_y as f64 * 0.1;
            sim.zoom_level = sim.zoom_level.max(0.1);
        }

        // Update simulation
        sim.update();

        // Egui UI
        egui_macroquad::ui(|egui_ctx| {
            egui::TopBottomPanel::top("hud").show(egui_ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("Time: {:.2}", sim.current_t));
                    ui.label(format!("Prey: {:.2}", sim.x));
                    ui.label(format!("Predator: {:.2}", sim.y));
                    ui.label(format!("Mean Prey: {:.2}", sim.stats.mean_x()));
                    ui.label(format!("Mean Predator: {:.2}", sim.stats.mean_y()));
                    ui.label(format!("Std Prey: {:.2}", sim.stats.std_x()));
                    ui.label(format!("Std Predator: {:.2}", sim.stats.std_y()));
                    ui.label(format!("Min Prey: {:.2}", sim.stats.min_x));
                    ui.label(format!("Min Predator: {:.2}", sim.stats.min_y));
                    ui.label(format!("Max Prey: {:.2}", sim.stats.max_x));
                    ui.label(format!("Max Predator: {:.2}", sim.stats.max_y));
                    ui.label(format!("dt: {:.4}", sim.params.dt));
                    if sim.paused {
                        ui.label("PAUSED");
                    }
                });
            });

            egui::SidePanel::right("controls").show(egui_ctx, |ui| {
                ui.heading("Controls");
                if ui.button(if sim.paused { "Resume" } else { "Pause" }).clicked() {
                    sim.paused = !sim.paused;
                }
                if ui.button("Reset Simulation").clicked() {
                    sim.reset();
                }
                if ui.button("Reset Stats").clicked() {
                    sim.reset_stats();
                }
                egui::ComboBox::from_label("Integrator")
                    .selected_text(format!("{:?}", sim.integrator))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut sim.integrator, Integrator::Euler, "Euler");
                        ui.selectable_value(&mut sim.integrator, Integrator::RK4, "RK4");
                        ui.selectable_value(&mut sim.integrator, Integrator::AdaptiveRK, "AdaptiveRK");
                    });
                ui.add(egui::Slider::new(&mut sim.params.dt, 0.001..=0.1).text("dt"));
                ui.add(egui::Slider::new(&mut sim.zoom_level, 0.1..=10.0).text("Zoom"));
                if ui.button("Auto Zoom").clicked() {
                    let actual_max = sim.history.iter().fold(0.0_f64, |acc, &(_, x, y)| acc.max(x).max(y));
                    sim.zoom_level = if actual_max > 0.0 { actual_max / sim.stats.max_x.max(sim.stats.max_y) } else { 1.0 };
                }

                ui.heading("Parameters");
                ui.add(DragValue::new(&mut sim.edit_params.alpha).prefix("α: ").speed(0.01));
                ui.add(DragValue::new(&mut sim.edit_params.beta).prefix("β: ").speed(0.0001));
                ui.add(DragValue::new(&mut sim.edit_params.delta).prefix("δ: ").speed(0.0001));
                ui.add(DragValue::new(&mut sim.edit_params.gamma).prefix("γ: ").speed(0.01));
                ui.add(DragValue::new(&mut sim.edit_params.x0).prefix("Initial Prey: ").speed(1.0));
                ui.add(DragValue::new(&mut sim.edit_params.y0).prefix("Initial Predator: ").speed(1.0));
                ui.add(DragValue::new(&mut sim.edit_params.tol).prefix("Tolerance: ").speed(0.0001));
                if ui.button("Apply Parameters (No Reset)").clicked() {
                    sim.params = sim.edit_params;
                }

                ui.heading("Export");
                ui.add(TextEdit::singleline(&mut sim.csv_filename).hint_text("CSV Filename"));
                ui.checkbox(&mut sim.append_csv, "Append to CSV");
                if ui.button("Export CSV").clicked() {
                    if let Err(e) = sim.export_csv() {
                        sim.error_msg = format!("CSV Export Error: {}", e);
                    }
                }
                ui.add(TextEdit::singleline(&mut sim.png_filename).hint_text("PNG Filename"));
                if ui.button("Export PNG").clicked() {
                    if let Err(e) = sim.export_png() {
                        sim.error_msg = format!("PNG Export Error: {}", e);
                    }
                }
                ui.add(TextEdit::singleline(&mut sim.json_filename).hint_text("JSON Filename"));
                if ui.button("Export JSON").clicked() {
                    if let Err(e) = sim.export_json() {
                        sim.error_msg = format!("JSON Export Error: {}", e);
                    }
                }
            });

            if sim.show_help {
                egui::Window::new("Help").show(egui_ctx, |ui| {
                    ui.label("Controls:");
                    ui.label("Space: Pause/Resume");
                    ui.label("R: Reset");
                    ui.label("H: Toggle Help");
                    ui.label("Esc: Quit");
                    ui.label("Mouse Wheel: Zoom");
                    ui.label("Use side panel for more controls");
                });
            }

            if !sim.error_msg.is_empty() {
                egui::Window::new("Error").show(egui_ctx, |ui| {
                    ui.label(&sim.error_msg);
                    if ui.button("Close").clicked() {
                        sim.error_msg.clear();
                    }
                });
            }
        });

        // Render graphics
        clear_background(LIGHTGRAY);

        // Adjusted margins and plot area for better axis visibility
        let left_margin = 50.0;
        let top_margin = 60.0;
        let right_margin = 320.0; // account for side panel and extra space
        let bottom_margin = 40.0;
        let plot_width = (screen_width() - left_margin - right_margin) / 2.0;
        let plot_height = screen_height() - top_margin - bottom_margin;

        draw_time_series(
            &sim,
            left_margin,
            screen_height() - bottom_margin,
            plot_width,
            plot_height,
            sim.zoom_level,
        );
        draw_phase_portrait(
            &sim,
            screen_width() / 2.0 + left_margin / 2.0,
            screen_height() - bottom_margin,
            plot_width,
            plot_height,
            sim.zoom_level,
        );

        egui_macroquad::draw();

        next_frame().await;
    }
}