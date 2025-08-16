use eframe::egui;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Gray-Scott Reaction-Diffusion Simulator"),
        ..Default::default()
    };

    eframe::run_native(
        "Gray-Scott Simulator",
        options,
        Box::new(|_cc| Ok(Box::new(GrayScottApp::new()))), // <-- wrap in Ok
    )
}

struct GrayScottApp {
    // Simulation state
    grid_size: usize,
    read_grid: Vec<[f32; 3]>, // [u, v, w]
    write_grid: Vec<[f32; 3]>,

    // Simulation parameters
    du: f32,   // Diffusion rate of U
    dv: f32,   // Diffusion rate of V
    dw: f32,   // Diffusion rate of W
    feed: f32, // Feed rate (f)
    kill: f32, // Kill rate (k)
    fw: f32,   // Feed rate for W
    kw: f32,   // Kill rate for W
    dt: f32,   // Time step
    steps_per_frame: u32,

    // Rendering
    texture_handle: Option<egui::TextureHandle>,
    color_image: egui::ColorImage,

    // UI state
    running: bool,
    mouse_painting: bool,
    paint_radius: f32,

    // Statistics
    current_stats: Statistics,
    stats_history: Vec<HistoryEntry>,
    frame_count: u64,

    // Grid resize
    new_grid_size: usize,

    // Color mapping: indices 0=U, 1=V, 2=W
    color_map: [usize; 3], // [R, G, B]
}

#[derive(Default)]
struct Statistics {
    mean_u: f32,
    mean_v: f32,
    total_v: f32,
}

struct HistoryEntry {
    time: f32,
    mean_u: f32,
    mean_v: f32,
    total_v: f32,
}

impl GrayScottApp {
    fn new() -> Self {
        let grid_size = 256;
        let grid_len = grid_size * grid_size;

        let mut app = Self {
            grid_size,
            read_grid: vec![[1.0, 0.0, 0.0]; grid_len],
            write_grid: vec![[1.0, 0.0, 0.0]; grid_len],

            // Default parameters (Solitons preset)
            du: 0.16,
            dv: 0.08,
            feed: 0.0367,
            kill: 0.0649,
            dt: 1.0,
            steps_per_frame: 1,

            dw: 0.08,
            fw: 0.04,
            kw: 0.06,

            color_map: [0, 2, 1], // Default: U=R, W=G, V=B

            texture_handle: None,
            color_image: egui::ColorImage::new([grid_size, grid_size], egui::Color32::BLACK),

            running: false,
            mouse_painting: false,
            paint_radius: 8.0,

            current_stats: Statistics::default(),
            stats_history: Vec::new(),
            frame_count: 0,

            new_grid_size: grid_size,
        };

        app.reset_grid();
        app.update_texture_data();
        app
    }

    fn reset_grid(&mut self) {
        // Initialize grid with U=1.0, V=0.0
        let grid_len = self.grid_size * self.grid_size;
        self.read_grid = vec![[1.0, 0.0, 0.0]; grid_len];
        self.write_grid = vec![[1.0, 0.0, 0.0]; grid_len];

        // Add a small central patch of V
        let center = self.grid_size / 2;
        let patch_size = 20;

        for y in (center - patch_size / 2)..(center + patch_size / 2) {
            for x in (center - patch_size / 2)..(center + patch_size / 2) {
                if x < self.grid_size && y < self.grid_size {
                    let idx = y * self.grid_size + x;
                    self.read_grid[idx][1] = 0.25; // Set V concentration
                    self.read_grid[idx][0] = 0.5; // Reduce U concentration
                    self.read_grid[idx][2] = 0.5; // Add W in center patch
                }
            }
        }

        self.frame_count = 0;
        self.stats_history.clear();
    }

    fn randomize_grid(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for cell in &mut self.read_grid {
            cell[0] = rng.gen_range(0.5..1.0); // U between 0.5 and 1.0
            cell[1] = rng.gen_range(0.0..0.5); // V between 0.0 and 0.5
            cell[2] = rng.gen_range(0.0..0.5); // W between 0.0 and 0.5
        }

        self.frame_count = 0;
        self.stats_history.clear();
    }

    fn resize_grid(&mut self, new_size: usize) {
        if new_size == self.grid_size {
            return;
        }

        let old_size = self.grid_size;
        self.grid_size = new_size;
        let grid_len = new_size * new_size;

        // Create new grids
        let mut new_read_grid = vec![[1.0, 0.0, 0.0]; grid_len];
        let mut new_write_grid = vec![[1.0, 0.0, 0.0]; grid_len];

        // Copy data from old grid (simple nearest neighbor sampling)
        for y in 0..new_size {
            for x in 0..new_size {
                let old_x = (x * old_size) / new_size;
                let old_y = (y * old_size) / new_size;

                if old_x < old_size && old_y < old_size {
                    let old_idx = old_y * old_size + old_x;
                    let new_idx = y * new_size + x;
                    new_read_grid[new_idx] = self.read_grid[old_idx];
                }
            }
        }

        self.read_grid = new_read_grid;
        self.write_grid = new_write_grid;
        self.color_image = egui::ColorImage::new([new_size, new_size], egui::Color32::BLACK);
        self.texture_handle = None; // Force texture recreation
    }

    fn step_simulation(&mut self) {
        let size = self.grid_size;

        for y in 0..size {
            for x in 0..size {
                let idx = y * size + x;
                let u = self.read_grid[idx][0];
                let v = self.read_grid[idx][1];
                let w = self.read_grid[idx][2];

                // Laplacians
                let laplacian_u = self.laplacian(x, y, 0);
                let laplacian_v = self.laplacian(x, y, 1);
                let laplacian_w = self.laplacian(x, y, 2);

                // Gray-Scott for U/V, simple interaction for W
                let reaction = u * v * v;
                let du_dt = self.du * laplacian_u - reaction + self.feed * (1.0 - u);
                let dv_dt = self.dv * laplacian_v + reaction - (self.feed + self.kill) * v;

                // W: similar to V, but interacts with U and V
                let reaction_w = u * w * w;
                let dw_dt = self.dw * laplacian_w + reaction_w - (self.fw + self.kw) * w;

                let new_u = (u + self.dt * du_dt).clamp(0.0, 1.0);
                let new_v = (v + self.dt * dv_dt).clamp(0.0, 1.0);
                let new_w = (w + self.dt * dw_dt).clamp(0.0, 1.0);

                self.write_grid[idx][0] = new_u;
                self.write_grid[idx][1] = new_v;
                self.write_grid[idx][2] = new_w;
            }
        }
        std::mem::swap(&mut self.read_grid, &mut self.write_grid);
    }

    fn laplacian(&self, x: usize, y: usize, component: usize) -> f32 {
        let size = self.grid_size;
        let center = self.read_grid[y * size + x][component];

        let mut sum = -4.0 * center; // Center weight

        // Add neighbors with periodic boundary conditions
        sum += self.read_grid[((y + size - 1) % size) * size + x][component]; // Top
        sum += self.read_grid[((y + 1) % size) * size + x][component]; // Bottom
        sum += self.read_grid[y * size + (x + size - 1) % size][component]; // Left
        sum += self.read_grid[y * size + (x + 1) % size][component]; // Right

        sum
    }

    fn update_texture_data(&mut self) {
        let pixels = &mut self.color_image.pixels;
        for (i, cell) in self.read_grid.iter().enumerate() {
            let r = (cell[self.color_map[0]] * 255.0) as u8;
            let g = (cell[self.color_map[1]] * 255.0) as u8;
            let b = (cell[self.color_map[2]] * 255.0) as u8;
            pixels[i] = egui::Color32::from_rgb(r, g, b);
        }
    }

    fn calculate_statistics(&mut self) {
        let total_cells = self.read_grid.len() as f32;
        let mut sum_u = 0.0;
        let mut sum_v = 0.0;

        for cell in &self.read_grid {
            sum_u += cell[0];
            sum_v += cell[1];
            // Optionally, add W stats if desired
        }

        self.current_stats.mean_u = sum_u / total_cells;
        self.current_stats.mean_v = sum_v / total_cells;
        self.current_stats.total_v = sum_v;

        // Add to history every 10 frames to avoid memory bloat
        if self.frame_count % 10 == 0 {
            let time = self.frame_count as f32 * self.dt;
            self.stats_history.push(HistoryEntry {
                time,
                mean_u: self.current_stats.mean_u,
                mean_v: self.current_stats.mean_v,
                total_v: self.current_stats.total_v,
            });
        }
    }

    fn export_statistics(&self) -> Result<(), std::io::Error> {
        let mut file = File::create("statistics.csv")?;
        writeln!(file, "Time,Mean_U,Mean_V,Total_V")?;

        for entry in &self.stats_history {
            writeln!(
                file,
                "{:.3},{:.6},{:.6},{:.6}",
                entry.time, entry.mean_u, entry.mean_v, entry.total_v
            )?;
        }

        Ok(())
    }

    fn paint_v(&mut self, pos: egui::Pos2, canvas_rect: egui::Rect) {
        let rel_x = (pos.x - canvas_rect.min.x) / canvas_rect.width();
        let rel_y = (pos.y - canvas_rect.min.y) / canvas_rect.height();

        if rel_x < 0.0 || rel_x > 1.0 || rel_y < 0.0 || rel_y > 1.0 {
            return;
        }

        let grid_x = (rel_x * self.grid_size as f32) as i32;
        let grid_y = (rel_y * self.grid_size as f32) as i32;

        let radius = self.paint_radius as i32;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x = grid_x + dx;
                let y = grid_y + dy;

                if x >= 0 && x < self.grid_size as i32 && y >= 0 && y < self.grid_size as i32 {
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq <= radius * radius {
                        let idx = y as usize * self.grid_size + x as usize;
                        // Paint V chemical
                        self.read_grid[idx][1] = (self.read_grid[idx][1] + 0.5).min(1.0);
                        self.read_grid[idx][0] = (self.read_grid[idx][0] * 0.5).max(0.0);
                    }
                }
            }
        }
    }

    // New: Paint W chemical
    fn paint_w(&mut self, pos: egui::Pos2, canvas_rect: egui::Rect) {
        let rel_x = (pos.x - canvas_rect.min.x) / canvas_rect.width();
        let rel_y = (pos.y - canvas_rect.min.y) / canvas_rect.height();

        if rel_x < 0.0 || rel_x > 1.0 || rel_y < 0.0 || rel_y > 1.0 {
            return;
        }

        let grid_x = (rel_x * self.grid_size as f32) as i32;
        let grid_y = (rel_y * self.grid_size as f32) as i32;

        let radius = self.paint_radius as i32;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x = grid_x + dx;
                let y = grid_y + dy;

                if x >= 0 && x < self.grid_size as i32 && y >= 0 && y < self.grid_size as i32 {
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq <= radius * radius {
                        let idx = y as usize * self.grid_size + x as usize;
                        // Paint W chemical
                        self.read_grid[idx][2] = (self.read_grid[idx][2] + 0.5).min(1.0);
                        self.read_grid[idx][0] = (self.read_grid[idx][0] * 0.5).max(0.0);
                    }
                }
            }
        }
    }

    fn load_preset(&mut self, preset: &str) {
        match preset {
            "Solitons" => {
                self.feed = 0.0367;
                self.kill = 0.0649;
                self.du = 0.16;
                self.dv = 0.08;
                self.fw = 0.04;
                self.kw = 0.06;
                self.dw = 0.08;
            }
            "Worms" => {
                self.feed = 0.078;
                self.kill = 0.061;
                self.du = 0.16;
                self.dv = 0.08;
                self.fw = 0.04;
                self.kw = 0.06;
                self.dw = 0.08;
            }
            "Chaos" => {
                self.feed = 0.025;
                self.kill = 0.050;
                self.du = 0.16;
                self.dv = 0.08;
                self.fw = 0.04;
                self.kw = 0.06;
                self.dw = 0.08;
            }
            _ => {}
        }
    }
}

impl eframe::App for GrayScottApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle grid resize
        if self.new_grid_size != self.grid_size {
            self.resize_grid(self.new_grid_size);
        }

        // Run simulation steps
        if self.running {
            for _ in 0..self.steps_per_frame {
                self.step_simulation();
                self.frame_count += 1;
            }
        }

        // Update statistics and texture
        self.calculate_statistics();
        self.update_texture_data();

        // Create/update texture
        if self.texture_handle.is_none() {
            self.texture_handle = Some(ctx.load_texture(
                "simulation",
                self.color_image.clone(),
                egui::TextureOptions::NEAREST,
            ));
        } else if let Some(handle) = &self.texture_handle {
            // Fix: use get_mut() to get a mutable reference to the handle
            if let Some(mut_handle) = self.texture_handle.as_mut() {
                mut_handle.set(self.color_image.clone(), egui::TextureOptions::NEAREST);
            }
        }

        // Side panel for controls
        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("Gray-Scott Simulator");
            ui.separator();

            // Simulation controls
            ui.heading("Simulation");
            ui.horizontal(|ui| {
                if ui
                    .button(if self.running { "Pause" } else { "Run" })
                    .clicked()
                {
                    self.running = !self.running;
                }
                if ui.button("Step").clicked() && !self.running {
                    self.step_simulation();
                    self.frame_count += 1;
                }
            });

            ui.separator();

            // Parameter sliders
            ui.heading("Parameters");
            ui.add(egui::Slider::new(&mut self.du, 0.0..=0.2).text("Du (U Diffusion)"));
            ui.add(egui::Slider::new(&mut self.dv, 0.0..=0.1).text("Dv (V Diffusion)"));
            ui.add(egui::Slider::new(&mut self.dw, 0.0..=0.1).text("Dw (W Diffusion)"));
            ui.add(egui::Slider::new(&mut self.feed, 0.0..=0.1).text("Feed Rate (f)"));
            ui.add(egui::Slider::new(&mut self.kill, 0.0..=0.1).text("Kill Rate (k)"));
            ui.add(egui::Slider::new(&mut self.fw, 0.0..=0.1).text("Feed Rate W (fw)"));
            ui.add(egui::Slider::new(&mut self.kw, 0.0..=0.1).text("Kill Rate W (kw)"));
            ui.add(egui::Slider::new(&mut self.dt, 0.1..=2.0).text("Time Step"));
            ui.add(egui::Slider::new(&mut self.steps_per_frame, 1..=64).text("Steps per Frame"));

            ui.separator();

            // Color mapping controls
            ui.heading("Color Mapping");
            let chemicals = ["U", "V", "W"];
            for (i, channel) in ["Red", "Green", "Blue"].iter().enumerate() {
                egui::ComboBox::from_label(format!("{channel} channel"))
                    .selected_text(chemicals[self.color_map[i]])
                    .show_ui(ui, |cb| {
                        for (idx, &chem) in chemicals.iter().enumerate() {
                            cb.selectable_value(&mut self.color_map[i], idx, chem);
                        }
                    });
            }

            ui.separator();

            // Grid management
            ui.heading("Grid Management");
            ui.add(
                egui::Slider::new(&mut self.new_grid_size, 64..=1024)
                    .text("Grid Size")
                    .step_by(64.0),
            );

            ui.horizontal(|ui| {
                if ui.button("Reset").clicked() {
                    self.reset_grid();
                }
                if ui.button("Randomize").clicked() {
                    self.randomize_grid();
                }
            });

            ui.separator();

            // Painting controls
            ui.heading("Interactive Painting");
            ui.add(egui::Slider::new(&mut self.paint_radius, 1.0..=20.0).text("Paint Radius"));
            ui.label("Click and drag on simulation to paint V chemical");
            ui.label("Hold Shift and drag to paint W chemical");

            ui.separator();

            // Presets
            ui.heading("Presets");
            ui.horizontal(|ui| {
                if ui.button("Solitons").clicked() {
                    self.load_preset("Solitons");
                }
                if ui.button("Worms").clicked() {
                    self.load_preset("Worms");
                }
            });
            if ui.button("Chaos").clicked() {
                self.load_preset("Chaos");
            }

            ui.separator();

            // Statistics
            ui.heading("Live Statistics");
            ui.label(format!("Frame: {}", self.frame_count));
            ui.label(format!("Mean U: {:.4}", self.current_stats.mean_u));
            ui.label(format!("Mean V: {:.4}", self.current_stats.mean_v));
            ui.label(format!("Total V: {:.2}", self.current_stats.total_v));

            if ui.button("Export Statistics").clicked() {
                match self.export_statistics() {
                    Ok(()) => println!("Statistics exported to statistics.csv"),
                    Err(e) => println!("Failed to export statistics: {}", e),
                }
            }
        });

        // Main panel for simulation display
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Simulation Canvas");
            ui.label(format!(
                "R = {}, G = {}, B = {}",
                ["U", "V", "W"][self.color_map[0]],
                ["U", "V", "W"][self.color_map[1]],
                ["U", "V", "W"][self.color_map[2]]
            ));

            if let Some(texture) = &self.texture_handle {
                let available_size = ui.available_size();
                let image_size = available_size.min_elem();
                let image_rect = egui::Rect::from_center_size(
                    available_size.to_pos2() * 0.5,
                    egui::Vec2::splat(image_size),
                );

                let response = ui.allocate_rect(image_rect, egui::Sense::click_and_drag());

                ui.put(
                    image_rect,
                    egui::Image::from_texture(texture)
                        .fit_to_exact_size(image_rect.size())
                        .rounding(egui::Rounding::same(4.0)),
                );

                // Paint V or W depending on Shift key
                if response.dragged() || response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        if ctx.input(|i| i.modifiers.shift) {
                            self.paint_w(pos, image_rect);
                        } else {
                            self.paint_v(pos, image_rect);
                        }
                    }
                }
            }
        });

        // Request continuous repaints
        ctx.request_repaint();
    }
}
