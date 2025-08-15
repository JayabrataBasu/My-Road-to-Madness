use macroquad::prelude::*;
use egui_macroquad::egui::DragValue;
use std::io::{self, BufWriter, Write};
use std::fs::File;

#[derive(Clone, Copy)]
struct Params {
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
    dt: f64,
    x0: f64,
    y0: f64,
}

struct Stats {
    sum_x: f64,
    sum_y: f64,
    count: usize,
    max_x: f64,
    max_y: f64,
}

impl Stats {
    fn mean_x(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_x / self.count as f64
        }
    }

    fn mean_y(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_y / self.count as f64
        }
    }
}

struct Simulation {
    params: Params,
    edit_params: Params,
    current_t: f64,
    x: f64,
    y: f64,
    history: Vec<(f64, f64, f64)>,
    stats: Stats,
    paused: bool,
    show_params: bool,
    zoom_level: f64,
}

impl Simulation {
    fn new(params: Params) -> Self {
        let x = params.x0;
        let y = params.y0;
        let history = vec![(0.0, x, y)];
        let stats = Stats {
            sum_x: x,
            sum_y: y,
            count: 1,
            max_x: x,
            max_y: y,
        };
        Simulation {
            params,
            edit_params: params,
            current_t: 0.0,
            x,
            y,
            history,
            stats,
            paused: false,
            show_params: false,
            zoom_level: 1.0,
        }
    }

    fn reset(&mut self) {
        self.current_t = 0.0;
        self.x = self.params.x0;
        self.y = self.params.y0;
        self.history = vec![(0.0, self.x, self.y)];
        self.stats = Stats {
            sum_x: self.x,
            sum_y: self.y,
            count: 1,
            max_x: self.x,
            max_y: self.y,
        };
    }

    fn update(&mut self) {
        if !self.paused {
            let new_state = rk4_step(&self.params, self.current_t, [self.x, self.y], self.params.dt);
            self.current_t += self.params.dt;
            self.x = new_state[0];
            self.y = new_state[1];
            self.history.push((self.current_t, self.x, self.y));
            self.stats.sum_x += self.x;
            self.stats.sum_y += self.y;
            self.stats.count += 1;
            self.stats.max_x = self.stats.max_x.max(self.x);
            self.stats.max_y = self.stats.max_y.max(self.y);
        }
    }

    fn export(&self) -> io::Result<()> {
        let file = File::create("lotka_volterra.csv")?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "time,prey,predator")?;
        for &(t, x, y) in &self.history {
            writeln!(writer, "{},{},{}", t, x, y)?;
        }
        Ok(())
    }
}

// Lotka-Volterra equations
fn lotka_volterra(params: &Params, state: [f64; 2]) -> [f64; 2] {
    let [x, y] = state;
    let dx = params.alpha * x - params.beta * x * y;
    let dy = params.delta * x * y - params.gamma * y;
    [dx, dy]
}

// RK4 integration step
fn rk4_step(params: &Params, _t: f64, state: [f64; 2], dt: f64) -> [f64; 2] {
    let f = |s: [f64; 2]| lotka_volterra(params, s);
    let k1 = f(state);
    let k2 = f([
        state[0] + dt / 2.0 * k1[0],
        state[1] + dt / 2.0 * k1[1],
    ]);
    let k3 = f([
        state[0] + dt / 2.0 * k2[0],
        state[1] + dt / 2.0 * k2[1],
    ]);
    let k4 = f([
        state[0] + dt * k3[0],
        state[1] + dt * k3[1],
    ]);
    [
        state[0] + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        state[1] + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
    ]
}

// Rendering functions
fn draw_time_series(sim: &Simulation, left: f32, bottom: f32, width: f32, height: f32, zoom_level: f64) {
    if sim.history.is_empty() || sim.current_t == 0.0 {
        return;
    }
    let actual_max = sim.history.iter().fold(0.0_f64, |acc, &(_, x, y)| acc.max(x).max(y));
    let display_max = (actual_max / zoom_level).max(1.0); // avoid div by zero or too small

    // Axes
    draw_line(left, bottom, left + width, bottom, 1.0, BLACK);
    draw_line(left, bottom, left, bottom - height, 1.0, BLACK);

    // Labels
    draw_text("Time", left + width / 2.0, bottom + 15.0, 20.0, BLACK);
    draw_text("Population", left - 60.0, bottom - height / 2.0, 20.0, BLACK);

    // Legends
    draw_text("Prey", left + 10.0, bottom - height + 20.0, 18.0, BLUE);
    draw_text("Predator", left + 10.0, bottom - height + 40.0, 18.0, RED);

    // Plot lines
    let mut prev_sx: Option<f32> = None;
    let mut prev_sy_x: Option<f32> = None;
    let mut prev_sy_y: Option<f32> = None;

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
    let actual_max = sim.history.iter().fold(0.0_f64, |acc, &(_, x, y)| acc.max(x).max(y));
    let display_max = (actual_max / zoom_level).max(1.0);

    // Axes
    draw_line(left, bottom, left + width, bottom, 1.0, BLACK);
    draw_line(left, bottom, left, bottom - height, 1.0, BLACK);

    // Labels
    draw_text("Prey", left + width / 2.0, bottom + 15.0, 20.0, BLACK);
    draw_text("Predator", left - 60.0, bottom - height / 2.0, 20.0, BLACK);

    // Plot line
    let mut prev_sx: Option<f32> = None;
    let mut prev_sy: Option<f32> = None;

    for &(_, x, y) in &sim.history {
        let sx = left + ((x / display_max) as f32 * width);
        let sy = bottom - ((y / display_max) as f32 * height);

        if let (Some(psx), Some(psy)) = (prev_sx, prev_sy) {
            draw_line(psx, psy, sx, sy, 2.0, GREEN); // Use green for phase portrait trajectory
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
    };
    let mut sim = Simulation::new(params);

    loop {
        // Handle keyboard inputs
        if is_key_pressed(KeyCode::Space) {
            sim.paused = !sim.paused;
        }
        if is_key_pressed(KeyCode::R) {
            sim.reset();
        }
        if is_key_pressed(KeyCode::E) {
            let _ = sim.export();
        }
        if is_key_pressed(KeyCode::Up) {
            sim.params.dt *= 1.2;
        }
        if is_key_pressed(KeyCode::Down) {
            sim.params.dt /= 1.2;
            if sim.params.dt < 0.001 {
                sim.params.dt = 0.001;
            }
        }
        if is_key_pressed(KeyCode::P) {
            sim.edit_params = sim.params;
            sim.show_params = true;
        }
        if is_key_pressed(KeyCode::LeftBracket) {
            sim.zoom_level = (sim.zoom_level - 0.1).max(0.1);
        }
        if is_key_pressed(KeyCode::RightBracket) {
            sim.zoom_level += 0.1;
        }
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        // Update simulation
        sim.update();

        // Egui UI
        egui_macroquad::ui(|egui_ctx| {
            egui_macroquad::egui::TopBottomPanel::top("hud").exact_height(30.0).show(egui_ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("Time: {:.2}", sim.current_t));
                    ui.label(format!("Mean Prey: {:.2}", sim.stats.mean_x()));
                    ui.label(format!("Mean Predator: {:.2}", sim.stats.mean_y()));
                    ui.label(format!("Max Prey: {:.2}", sim.stats.max_x));
                    ui.label(format!("Max Predator: {:.2}", sim.stats.max_y));
                    ui.label(format!("dt: {:.4}", sim.params.dt));
                    if sim.paused {
                        ui.label("PAUSED");
                    }
                });
            });

            if sim.show_params {
                egui_macroquad::egui::Window::new("Edit Parameters").show(egui_ctx, |ui| {
                    ui.add(DragValue::new(&mut sim.edit_params.alpha).prefix("α: ").speed(0.01));
                    ui.add(DragValue::new(&mut sim.edit_params.beta).prefix("β: ").speed(0.0001));
                    ui.add(DragValue::new(&mut sim.edit_params.delta).prefix("δ: ").speed(0.0001));
                    ui.add(DragValue::new(&mut sim.edit_params.gamma).prefix("γ: ").speed(0.01));
                    ui.add(DragValue::new(&mut sim.edit_params.dt).prefix("dt: ").speed(0.001));
                    ui.add(DragValue::new(&mut sim.edit_params.x0).prefix("Initial Prey: ").speed(1.0));
                    ui.add(DragValue::new(&mut sim.edit_params.y0).prefix("Initial Predator: ").speed(1.0));

                    ui.horizontal(|ui| {
                        if ui.button("Apply and Reset").clicked() {
                            sim.params = sim.edit_params;
                            sim.reset();
                            sim.show_params = false;
                        }
                        if ui.button("Cancel").clicked() {
                            sim.show_params = false;
                        }
                    });
                });
            }
        });

        // Render graphics
        clear_background(LIGHTGRAY);

        let top = 40.0;
        let height = screen_height() - top - 20.0;
        let width = screen_width() / 2.0 - 20.0;

        // Time series graph
        draw_time_series(&sim, 10.0, screen_height() - 10.0, width, height, sim.zoom_level);

        // Phase portrait graph
        draw_phase_portrait(&sim, screen_width() / 2.0 + 10.0, screen_height() - 10.0, width, height, sim.zoom_level);

        // Draw egui on top
        egui_macroquad::draw();

        next_frame().await;
    }
}