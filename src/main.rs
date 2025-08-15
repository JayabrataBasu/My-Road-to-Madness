use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use rand::Rng;
use std::collections::HashSet;
use std::time::{Duration, Instant};

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Conway's Game of Life"),
        ..Default::default()
    };

    eframe::run_native(
        "Conway's Game of Life",
        options,
        Box::new(|_cc| Ok(Box::new(GameOfLifeApp::new()))),
    )
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Cell {
    x: i32,
    y: i32,
}

impl Cell {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    fn neighbors(&self) -> Vec<Cell> {
        vec![
            Cell::new(self.x - 1, self.y - 1),
            Cell::new(self.x, self.y - 1),
            Cell::new(self.x + 1, self.y - 1),
            Cell::new(self.x - 1, self.y),
            Cell::new(self.x + 1, self.y),
            Cell::new(self.x - 1, self.y + 1),
            Cell::new(self.x, self.y + 1),
            Cell::new(self.x + 1, self.y + 1),
        ]
    }
}

struct GameOfLifeApp {
    cells: HashSet<Cell>,
    is_running: bool,
    generation: u64,
    last_update: Instant,
    update_interval: Duration,
    cell_size: f32,
    grid_offset: Vec2,
    dragging: bool,
    last_mouse_pos: Pos2,
    zoom: f32,

    // UI Controls
    speed_ms: f32,
    grid_size: f32,
    show_grid: bool,
    wrap_edges: bool,

    // Patterns
    selected_pattern: PatternType,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum PatternType {
    Single,
    Glider,
    Blinker,
    Block,
    Toad,
    Beacon,
    Pulsar,
    GliderGun,
}

impl Default for GameOfLifeApp {
    fn default() -> Self {
        Self::new()
    }
}

impl GameOfLifeApp {
    fn new() -> Self {
        Self {
            cells: HashSet::new(),
            is_running: false,
            generation: 0,
            last_update: Instant::now(),
            update_interval: Duration::from_millis(100),
            cell_size: 10.0,
            grid_offset: Vec2::ZERO,
            dragging: false,
            last_mouse_pos: Pos2::ZERO,
            zoom: 1.0,
            speed_ms: 100.0,
            grid_size: 10.0,
            show_grid: true,
            wrap_edges: false,
            selected_pattern: PatternType::Single,
        }
    }

    fn step(&mut self) {
        let mut neighbor_counts: std::collections::HashMap<Cell, u32> =
            std::collections::HashMap::new();

        // Count neighbors for all cells and their neighbors
        for cell in &self.cells {
            for neighbor in cell.neighbors() {
                *neighbor_counts.entry(neighbor).or_insert(0) += 1;
            }
        }

        let mut new_cells = HashSet::new();

        for (cell, count) in neighbor_counts {
            let is_alive = self.cells.contains(&cell);

            match (is_alive, count) {
                // Live cell with 2 or 3 neighbors survives
                (true, 2) | (true, 3) => {
                    new_cells.insert(cell);
                }
                // Dead cell with exactly 3 neighbors becomes alive
                (false, 3) => {
                    new_cells.insert(cell);
                }
                _ => {}
            }
        }

        self.cells = new_cells;
        self.generation += 1;
    }

    fn clear(&mut self) {
        self.cells.clear();
        self.generation = 0;
        self.is_running = false;
    }

    fn randomize(&mut self) {
        self.clear();
        let mut rng = rand::thread_rng();

        for x in -50..50 {
            for y in -30..30 {
                if rng.gen_bool(0.3) {
                    self.cells.insert(Cell::new(x, y));
                }
            }
        }
    }

    fn screen_to_grid(&self, pos: Pos2) -> Cell {
        let adjusted_pos = pos - self.grid_offset;
        let grid_x = (adjusted_pos.x / (self.cell_size * self.zoom)).floor() as i32;
        let grid_y = (adjusted_pos.y / (self.cell_size * self.zoom)).floor() as i32;
        Cell::new(grid_x, grid_y)
    }

    fn grid_to_screen(&self, cell: &Cell) -> Pos2 {
        let x = cell.x as f32 * self.cell_size * self.zoom + self.grid_offset.x;
        let y = cell.y as f32 * self.cell_size * self.zoom + self.grid_offset.y;
        Pos2::new(x, y)
    }

    fn place_pattern(&mut self, center: Cell, pattern: PatternType) {
        let pattern_cells = match pattern {
            PatternType::Single => vec![(0, 0)],
            PatternType::Glider => vec![(0, 0), (1, 0), (2, 0), (2, 1), (1, 2)],
            PatternType::Blinker => vec![(0, 0), (1, 0), (2, 0)],
            PatternType::Block => vec![(0, 0), (0, 1), (1, 0), (1, 1)],
            PatternType::Toad => vec![(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)],
            PatternType::Beacon => vec![(0, 0), (1, 0), (0, 1), (3, 2), (2, 3), (3, 3)],
            PatternType::Pulsar => vec![
                (2, 0),
                (3, 0),
                (4, 0),
                (8, 0),
                (9, 0),
                (10, 0),
                (0, 2),
                (5, 2),
                (7, 2),
                (12, 2),
                (0, 3),
                (5, 3),
                (7, 3),
                (12, 3),
                (0, 4),
                (5, 4),
                (7, 4),
                (12, 4),
                (2, 5),
                (3, 5),
                (4, 5),
                (8, 5),
                (9, 5),
                (10, 5),
                (2, 7),
                (3, 7),
                (4, 7),
                (8, 7),
                (9, 7),
                (10, 7),
                (0, 8),
                (5, 8),
                (7, 8),
                (12, 8),
                (0, 9),
                (5, 9),
                (7, 9),
                (12, 9),
                (0, 10),
                (5, 10),
                (7, 10),
                (12, 10),
                (2, 12),
                (3, 12),
                (4, 12),
                (8, 12),
                (9, 12),
                (10, 12),
            ],
            PatternType::GliderGun => vec![
                (24, 0),
                (22, 1),
                (24, 1),
                (12, 2),
                (13, 2),
                (20, 2),
                (21, 2),
                (34, 2),
                (35, 2),
                (11, 3),
                (15, 3),
                (20, 3),
                (21, 3),
                (34, 3),
                (35, 3),
                (0, 4),
                (1, 4),
                (10, 4),
                (16, 4),
                (20, 4),
                (21, 4),
                (0, 5),
                (1, 5),
                (10, 5),
                (14, 5),
                (16, 5),
                (17, 5),
                (22, 5),
                (24, 5),
                (10, 6),
                (16, 6),
                (24, 6),
                (11, 7),
                (15, 7),
                (12, 8),
                (13, 8),
            ],
        };

        for (dx, dy) in pattern_cells {
            self.cells.insert(Cell::new(center.x + dx, center.y + dy));
        }
    }

    fn draw_grid(&self, painter: &egui::Painter, rect: Rect) {
        if !self.show_grid {
            return;
        }

        let cell_size = self.cell_size * self.zoom;
        if cell_size < 5.0 {
            return; // Don't draw grid when too zoomed out
        }

        let stroke = Stroke::new(0.5, Color32::from_gray(100));

        // Vertical lines
        let start_x = ((rect.min.x - self.grid_offset.x) / cell_size).floor() as i32;
        let end_x = ((rect.max.x - self.grid_offset.x) / cell_size).ceil() as i32;

        for i in start_x..=end_x {
            let x = i as f32 * cell_size + self.grid_offset.x;
            if x >= rect.min.x && x <= rect.max.x {
                painter.line_segment([Pos2::new(x, rect.min.y), Pos2::new(x, rect.max.y)], stroke);
            }
        }

        // Horizontal lines
        let start_y = ((rect.min.y - self.grid_offset.y) / cell_size).floor() as i32;
        let end_y = ((rect.max.y - self.grid_offset.y) / cell_size).ceil() as i32;

        for i in start_y..=end_y {
            let y = i as f32 * cell_size + self.grid_offset.y;
            if y >= rect.min.y && y <= rect.max.y {
                painter.line_segment([Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)], stroke);
            }
        }
    }

    fn draw_cells(&self, painter: &egui::Painter, rect: Rect) {
        let cell_size = self.cell_size * self.zoom;

        for cell in &self.cells {
            let pos = self.grid_to_screen(cell);
            let cell_rect = Rect::from_min_size(pos, Vec2::splat(cell_size));

            // Only draw cells that are visible
            if cell_rect.intersects(rect) {
                painter.rect_filled(cell_rect, 0.0, Color32::from_rgb(0, 150, 255));
            }
        }
    }
}

impl eframe::App for GameOfLifeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update simulation
        if self.is_running && self.last_update.elapsed() >= self.update_interval {
            self.step();
            self.last_update = Instant::now();
        }

        // Update interval from slider
        self.update_interval = Duration::from_millis(self.speed_ms as u64);
        self.cell_size = self.grid_size;

        // Request repaint if running
        if self.is_running {
            ctx.request_repaint();
        }

        // Top panel for controls
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button(if self.is_running { "Pause" } else { "Play" })
                    .clicked()
                {
                    self.is_running = !self.is_running;
                    if self.is_running {
                        self.last_update = Instant::now();
                    }
                }

                if ui.button("Step").clicked() {
                    self.step();
                }

                if ui.button("Clear").clicked() {
                    self.clear();
                }

                if ui.button("Random").clicked() {
                    self.randomize();
                }

                ui.separator();

                ui.label("Speed:");
                ui.add(egui::Slider::new(&mut self.speed_ms, 10.0..=1000.0).text("ms"));

                ui.label("Cell Size:");
                ui.add(egui::Slider::new(&mut self.grid_size, 2.0..=30.0).text("px"));

                ui.checkbox(&mut self.show_grid, "Show Grid");
                ui.checkbox(&mut self.wrap_edges, "Wrap Edges");

                ui.separator();

                ui.label(format!("Generation: {}", self.generation));
                ui.label(format!("Population: {}", self.cells.len()));
            });

            ui.horizontal(|ui| {
                ui.label("Pattern:");
                egui::ComboBox::from_label("")
                    .selected_text(format!("{:?}", self.selected_pattern))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.selected_pattern,
                            PatternType::Single,
                            "Single Cell",
                        );
                        ui.selectable_value(
                            &mut self.selected_pattern,
                            PatternType::Glider,
                            "Glider",
                        );
                        ui.selectable_value(
                            &mut self.selected_pattern,
                            PatternType::Blinker,
                            "Blinker",
                        );
                        ui.selectable_value(
                            &mut self.selected_pattern,
                            PatternType::Block,
                            "Block",
                        );
                        ui.selectable_value(&mut self.selected_pattern, PatternType::Toad, "Toad");
                        ui.selectable_value(
                            &mut self.selected_pattern,
                            PatternType::Beacon,
                            "Beacon",
                        );
                        ui.selectable_value(
                            &mut self.selected_pattern,
                            PatternType::Pulsar,
                            "Pulsar",
                        );
                        ui.selectable_value(
                            &mut self.selected_pattern,
                            PatternType::GliderGun,
                            "Glider Gun",
                        );
                    });

                ui.label("Left click to place pattern, Right click to toggle single cells");
                ui.label("Middle mouse to pan, Scroll to zoom");
            });
        });

        // Main game area
        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), Sense::click_and_drag());

            // Handle mouse input
            if let Some(pos) = response.interact_pointer_pos() {
                // Left click - place pattern
                if response.clicked() {
                    let cell = self.screen_to_grid(pos);
                    self.place_pattern(cell, self.selected_pattern);
                }

                // Right click - toggle single cell
                if response.secondary_clicked() {
                    let cell = self.screen_to_grid(pos);
                    if self.cells.contains(&cell) {
                        self.cells.remove(&cell);
                    } else {
                        self.cells.insert(cell);
                    }
                }

                // Middle mouse drag - pan
                if response.dragged_by(egui::PointerButton::Middle) {
                    let delta = pos - self.last_mouse_pos;
                    self.grid_offset += delta;
                }

                self.last_mouse_pos = pos;
            }

            // Handle zoom
            if let Some(hover_pos) = response.hover_pos() {
                let scroll_delta = ui.input(|i| i.raw_scroll_delta.y);
                if scroll_delta != 0.0 {
                    let zoom_factor = 1.0 + scroll_delta * 0.001;
                    let old_zoom = self.zoom;
                    self.zoom = (self.zoom * zoom_factor).clamp(0.1, 5.0);

                    // Adjust offset to zoom towards mouse position
                    let zoom_change = self.zoom / old_zoom;
                    let mouse_offset = hover_pos - self.grid_offset;
                    self.grid_offset = hover_pos - mouse_offset * zoom_change;
                }
            }

            // Draw grid and cells
            let rect = response.rect;
            self.draw_grid(&painter, rect);
            self.draw_cells(&painter, rect);
        });
    }
}
