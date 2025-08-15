use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2, RichText};
use egui_plot::{Line, Plot, PlotPoints};
use rand::Rng;
use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Conway's Game of Life - Advanced Edition")
            .with_resizable(true),
        ..Default::default()
    };
    
    eframe::run_native(
        "Conway's Game of Life - Advanced Edition",
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

#[derive(Clone)]
struct StatPoint {
    generation: u64,
    population: usize,
    births: usize,
    deaths: usize,
    timestamp: Instant,
}

struct Statistics {
    history: VecDeque<StatPoint>,
    max_history: usize,
    max_population: usize,
    total_generations: u64,
    start_time: Instant,
    last_population: usize,
    population_stable_count: usize,
    extinction_events: usize,
}

impl Statistics {
    fn new() -> Self {
        Self {
            history: VecDeque::new(),
            max_history: 1000,
            max_population: 0,
            total_generations: 0,
            start_time: Instant::now(),
            last_population: 0,
            population_stable_count: 0,
            extinction_events: 0,
        }
    }
    
    fn update(&mut self, generation: u64, current_population: usize, previous_cells: &HashSet<Cell>, current_cells: &HashSet<Cell>) {
        // Calculate births and deaths
        let births = current_cells.difference(previous_cells).count();
        let deaths = previous_cells.difference(current_cells).count();
        
        let stat_point = StatPoint {
            generation,
            population: current_population,
            births,
            deaths,
            timestamp: Instant::now(),
        };
        
        self.history.push_back(stat_point);
        
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
        
        // Update statistics
        self.max_population = self.max_population.max(current_population);
        self.total_generations = generation;
        
        // Track stability and extinction
        if current_population == self.last_population {
            self.population_stable_count += 1;
        } else {
            self.population_stable_count = 0;
        }
        
        if current_population == 0 && self.last_population > 0 {
            self.extinction_events += 1;
        }
        
        self.last_population = current_population;
    }
    
    fn clear(&mut self) {
        self.history.clear();
        self.max_population = 0;
        self.total_generations = 0;
        self.start_time = Instant::now();
        self.last_population = 0;
        self.population_stable_count = 0;
        self.extinction_events = 0;
    }
    
    fn is_stable(&self) -> bool {
        self.population_stable_count > 5
    }
    
    fn average_population(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: usize = self.history.iter().map(|p| p.population).sum();
        sum as f64 / self.history.len() as f64
    }
    
    fn generations_per_second(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_generations as f64 / elapsed
        } else {
            0.0
        }
    }
}

struct GameOfLifeApp {
    cells: HashSet<Cell>,
    previous_cells: HashSet<Cell>,
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
    show_statistics: bool,
    show_charts: bool,
    
    // Patterns
    selected_pattern: PatternType,
    
    // Statistics
    stats: Statistics,
    
    // UI State
    left_panel_open: bool,
    right_panel_open: bool,
    
    // Drawing options
    cell_color: Color32,
    grid_color: Color32,
    background_color: Color32,
    show_birth_death_animation: bool,
    birth_cells: HashSet<Cell>,
    death_cells: HashSet<Cell>,
    animation_timer: Instant,
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
    Spaceship,
    Pentadecathlon,
    Random5x5,
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
            previous_cells: HashSet::new(),
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
            show_statistics: true,
            show_charts: true,
            selected_pattern: PatternType::Single,
            stats: Statistics::new(),
            left_panel_open: true,
            right_panel_open: true,
            cell_color: Color32::from_rgb(0, 150, 255),
            grid_color: Color32::from_gray(100),
            background_color: Color32::from_gray(20),
            show_birth_death_animation: true,
            birth_cells: HashSet::new(),
            death_cells: HashSet::new(),
            animation_timer: Instant::now(),
        }
    }
    
    fn step(&mut self) {
        self.previous_cells = self.cells.clone();
        
        let mut neighbor_counts: std::collections::HashMap<Cell, u32> = std::collections::HashMap::new();
        
        // Count neighbors for all cells and their neighbors
        for cell in &self.cells {
            for neighbor in cell.neighbors() {
                *neighbor_counts.entry(neighbor).or_insert(0) += 1;
            }
        }
        
        let mut new_cells = HashSet::new();
        let mut births = HashSet::new();
        let mut deaths = HashSet::new();
        
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
                    births.insert(cell);
                }
                // Live cell dies
                (true, _) => {
                    deaths.insert(cell);
                }
                _ => {}
            }
        }
        
        // Update animation cells
        if self.show_birth_death_animation {
            self.birth_cells = births;
            self.death_cells = deaths;
            self.animation_timer = Instant::now();
        }
        
        self.cells = new_cells;
        self.generation += 1;
        
        // Update statistics
        self.stats.update(self.generation, self.cells.len(), &self.previous_cells, &self.cells);
    }
    
    fn clear(&mut self) {
        self.cells.clear();
        self.previous_cells.clear();
        self.generation = 0;
        self.is_running = false;
        self.birth_cells.clear();
        self.death_cells.clear();
        self.stats.clear();
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
                (2, 0), (3, 0), (4, 0), (8, 0), (9, 0), (10, 0),
                (0, 2), (5, 2), (7, 2), (12, 2),
                (0, 3), (5, 3), (7, 3), (12, 3),
                (0, 4), (5, 4), (7, 4), (12, 4),
                (2, 5), (3, 5), (4, 5), (8, 5), (9, 5), (10, 5),
                (2, 7), (3, 7), (4, 7), (8, 7), (9, 7), (10, 7),
                (0, 8), (5, 8), (7, 8), (12, 8),
                (0, 9), (5, 9), (7, 9), (12, 9),
                (0, 10), (5, 10), (7, 10), (12, 10),
                (2, 12), (3, 12), (4, 12), (8, 12), (9, 12), (10, 12),
            ],
            PatternType::GliderGun => vec![
                (24, 0),
                (22, 1), (24, 1),
                (12, 2), (13, 2), (20, 2), (21, 2), (34, 2), (35, 2),
                (11, 3), (15, 3), (20, 3), (21, 3), (34, 3), (35, 3),
                (0, 4), (1, 4), (10, 4), (16, 4), (20, 4), (21, 4),
                (0, 5), (1, 5), (10, 5), (14, 5), (16, 5), (17, 5), (22, 5), (24, 5),
                (10, 6), (16, 6), (24, 6),
                (11, 7), (15, 7),
                (12, 8), (13, 8),
            ],
            PatternType::Spaceship => vec![(0, 0), (3, 0), (4, 1), (0, 2), (4, 2), (1, 3), (2, 3), (3, 3), (4, 3)],
            PatternType::Pentadecathlon => vec![
                (0, 0), (1, 0), (3, 0), (4, 0), (5, 0), (6, 0), (8, 0), (9, 0)
            ],
            PatternType::Random5x5 => {
                let mut pattern = Vec::new();
                let mut rng = rand::thread_rng();
                for x in 0..5 {
                    for y in 0..5 {
                        if rng.gen_bool(0.5) {
                            pattern.push((x, y));
                        }
                    }
                }
                pattern
            }
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
        
        let stroke = Stroke::new(0.5, self.grid_color);
        
        // Vertical lines
        let start_x = ((rect.min.x - self.grid_offset.x) / cell_size).floor() as i32;
        let end_x = ((rect.max.x - self.grid_offset.x) / cell_size).ceil() as i32;
        
        for i in start_x..=end_x {
            let x = i as f32 * cell_size + self.grid_offset.x;
            if x >= rect.min.x && x <= rect.max.x {
                painter.line_segment(
                    [Pos2::new(x, rect.min.y), Pos2::new(x, rect.max.y)],
                    stroke,
                );
            }
        }
        
        // Horizontal lines
        let start_y = ((rect.min.y - self.grid_offset.y) / cell_size).floor() as i32;
        let end_y = ((rect.max.y - self.grid_offset.y) / cell_size).ceil() as i32;
        
        for i in start_y..=end_y {
            let y = i as f32 * cell_size + self.grid_offset.y;
            if y >= rect.min.y && y <= rect.max.y {
                painter.line_segment(
                    [Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)],
                    stroke,
                );
            }
        }
    }
    
    fn draw_cells(&self, painter: &egui::Painter, rect: Rect) {
        let cell_size = self.cell_size * self.zoom;
        
        // Draw regular cells
        for cell in &self.cells {
            let pos = self.grid_to_screen(cell);
            let cell_rect = Rect::from_min_size(pos, Vec2::splat(cell_size));
            
            // Only draw cells that are visible
            if cell_rect.intersects(rect) {
                painter.rect_filled(cell_rect, 1.0, self.cell_color);
            }
        }
        
        // Draw birth/death animations
        if self.show_birth_death_animation && self.animation_timer.elapsed() < Duration::from_millis(200) {
            let alpha = (1.0 - self.animation_timer.elapsed().as_secs_f32() / 0.2).max(0.0);
            
            // Draw births in green
            for cell in &self.birth_cells {
                let pos = self.grid_to_screen(cell);
                let cell_rect = Rect::from_min_size(pos, Vec2::splat(cell_size));
                
                if cell_rect.intersects(rect) {
                    let mut color = Color32::from_rgb(0, 255, 100);
                    color[3] = (255.0 * alpha) as u8;
                    painter.rect_filled(cell_rect, 1.0, color);
                }
            }
            
            // Draw deaths in red
            for cell in &self.death_cells {
                let pos = self.grid_to_screen(cell);
                let cell_rect = Rect::from_min_size(pos, Vec2::splat(cell_size));
                
                if cell_rect.intersects(rect) {
                    let mut color = Color32::from_rgb(255, 100, 100);
                    color[3] = (255.0 * alpha) as u8;
                    painter.rect_filled(cell_rect, 1.0, color);
                }
            }
        }
    }
    
    fn draw_statistics_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("statistics_panel")
            .resizable(true)
            .default_width(300.0)
            .show_animated(ctx, self.left_panel_open, |ui| {
                ui.heading("📊 Statistics");
                ui.separator();
                
                ui.horizontal(|ui| {
                    ui.label("Generation:");
                    ui.label(RichText::new(format!("{}", self.generation)).strong());
                });
                
                ui.horizontal(|ui| {
                    ui.label("Population:");
                    ui.label(RichText::new(format!("{}", self.cells.len())).strong());
                });
                
                ui.horizontal(|ui| {
                    ui.label("Max Population:");
                    ui.label(RichText::new(format!("{}", self.stats.max_population)).strong());
                });
                
                ui.horizontal(|ui| {
                    ui.label("Average Population:");
                    ui.label(RichText::new(format!("{:.1}", self.stats.average_population())).strong());
                });
                
                ui.horizontal(|ui| {
                    ui.label("Gen/sec:");
                    ui.label(RichText::new(format!("{:.2}", self.stats.generations_per_second())).strong());
                });
                
                ui.horizontal(|ui| {
                    ui.label("Extinction Events:");
                    ui.label(RichText::new(format!("{}", self.stats.extinction_events)).strong());
                });
                
                if self.stats.is_stable() {
                    ui.label(RichText::new("🔒 Population Stable").color(Color32::YELLOW));
                }
                
                ui.separator();
                
                ui.heading("🎨 Appearance");
                
                ui.horizontal(|ui| {
                    ui.label("Cell Color:");
                    ui.color_edit_button_srgba(&mut self.cell_color);
                });
                
                ui.horizontal(|ui| {
                    ui.label("Grid Color:");
                    ui.color_edit_button_srgba(&mut self.grid_color);
                });
                
                ui.checkbox(&mut self.show_birth_death_animation, "Birth/Death Animation");
                
                ui.separator();
                
                ui.heading("🎯 Patterns");
                egui::ComboBox::from_label("Select Pattern")
                    .selected_text(format!("{:?}", self.selected_pattern))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.selected_pattern, PatternType::Single, "Single Cell");
                        ui.selectable_value(&mut self.selected_pattern, PatternType::Glider, "Glider");
                        ui.selectable_value(&mut self.selected_pattern, PatternType::Blinker, "Blinker");
                        ui.selectable_value(&mut self.selected_pattern, PatternType::Block, "Block");
                        ui.selectable_value(&mut self.selected_pattern, PatternType::Toad, "Toad");
                        ui.selectable_value(&mut self.selected_pattern, PatternType::Beacon, "Beacon");
                        ui.selectable_value(&mut self.selected_pattern, PatternType::Pulsar, "Pulsar");
                        ui.selectable_value(&mut self.selected_pattern, PatternType::GliderGun, "Glider Gun");
                        ui.selectable_value(&mut self.selected_pattern, PatternType::Spaceship, "Spaceship");
                        ui.selectable_value(&mut self.selected_pattern, PatternType::Pentadecathlon, "Pentadecathlon");
                        ui.selectable_value(&mut self.selected_pattern, PatternType::Random5x5, "Random 5x5");
                    });
            });
    }
    
    fn draw_charts_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("charts_panel")
            .resizable(true)
            .default_width(400.0)
            .show_animated(ctx, self.right_panel_open, |ui| {
                ui.heading("📈 Population Chart");
                ui.separator();
                
                if !self.stats.history.is_empty() {
                    // Population over time
                    let population_points: PlotPoints = self.stats.history
                        .iter()
                        .map(|p| [p.generation as f64, p.population as f64])
                        .collect();
                    
                    Plot::new("population_plot")
                        .view_aspect(2.0)
                        .height(200.0)
                        .show(ui, |plot_ui| {
                            plot_ui.line(
                                Line::new(population_points)
                                    .color(Color32::from_rgb(0, 150, 255))
                                    .name("Population")
                            );
                        });
                    
                    ui.separator();
                    
                    // Births and deaths over time
                    ui.heading("📊 Birth/Death Rate");
                    
                    let birth_points: PlotPoints = self.stats.history
                        .iter()
                        .map(|p| [p.generation as f64, p.births as f64])
                        .collect();
                    
                    let death_points: PlotPoints = self.stats.history
                        .iter()
                        .map(|p| [p.generation as f64, p.deaths as f64])
                        .collect();
                    
                    Plot::new("birth_death_plot")
                        .view_aspect(2.0)
                        .height(150.0)
                        .show(ui, |plot_ui| {
                            plot_ui.line(
                                Line::new(birth_points)
                                    .color(Color32::from_rgb(0, 255, 100))
                                    .name("Births")
                            );
                            plot_ui.line(
                                Line::new(death_points)
                                    .color(Color32::from_rgb(255, 100, 100))
                                    .name("Deaths")
                            );
                        });
                    
                    ui.separator();
                    
                    // Recent statistics
                    if let Some(latest) = self.stats.history.back() {
                        ui.heading("📋 Latest Generation");
                        ui.horizontal(|ui| {
                            ui.label("Births:");
                            ui.label(RichText::new(format!("{}", latest.births)).color(Color32::from_rgb(0, 255, 100)));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Deaths:");
                            ui.label(RichText::new(format!("{}", latest.deaths)).color(Color32::from_rgb(255, 100, 100)));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Net Change:");
                            let net = latest.births as i32 - latest.deaths as i32;
                            let color = if net > 0 { Color32::from_rgb(0, 255, 100) } 
                                       else if net < 0 { Color32::from_rgb(255, 100, 100) } 
                                       else { Color32::WHITE };
                            ui.label(RichText::new(format!("{:+}", net)).color(color));
                        });
                    }
                } else {
                    ui.label("No data yet. Start the simulation to see charts!");
                }
            });
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
        
        // Request repaint if running or animating
        if self.is_running || self.animation_timer.elapsed() < Duration::from_millis(200) {
            ctx.request_repaint();
        }
        
        // Top panel for main controls
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Main controls
                if ui.button(if self.is_running { "⏸ Pause" } else { "▶ Play" }).clicked() {
                    self.is_running = !self.is_running;
                    if self.is_running {
                        self.last_update = Instant::now();
                    }
                }
                
                if ui.button("⏭ Step").clicked() {
                    self.step();
                }
                
                if ui.button("🗑 Clear").clicked() {
                    self.clear();
                }
                
                if ui.button("🎲 Random").clicked() {
                    self.randomize();
                }
                
                ui.separator();
                
                // Panel toggles
                ui.toggle_value(&mut self.left_panel_open, "📊 Stats");
                ui.toggle_value(&mut self.right_panel_open, "📈 Charts");
                
                ui.separator();
                
                // Speed and size controls
                ui.label("Speed:");
                ui.add(egui::Slider::new(&mut self.speed_ms, 10.0..=1000.0)
                    .text("ms")
                    .logarithmic(true));
                
                ui.label("Cell Size:");
                ui.add(egui::Slider::new(&mut self.grid_size, 2.0..=30.0).text("px"));
                
                ui.checkbox(&mut self.show_grid, "Grid");
                ui.checkbox(&mut self.wrap_edges, "Wrap");
            });
            
            ui.horizontal(|ui| {
                ui.label("💡 Left click: place pattern | Right click: toggle cell | Middle drag: pan | Scroll: zoom");
            });
        });
        
        // Side panels
        if self.show_statistics {
            self.draw_statistics_panel(ctx);
        }
        
        if self.show_charts {
            self.draw_charts_panel(ctx);
        }
        
        // Main game area
        egui::CentralPanel::default().show(ctx, |ui| {
            // Set background color
            ui.painter().rect_filled(ui.max_rect(), 0.0, self.background_color);
            
            let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
            
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
            
            // Handle keyboard shortcuts
            ui.input(|i| {
                if i.key_pressed(egui::Key::Space) {
                    self.is_running = !self.is_running;
                    if self.is_running {
                        self.last_update = Instant::now();
                    }
                }
                
                if i.key_pressed(egui::Key::S) {
                    self.step();
                }
                
                if i.key_pressed(egui::Key::C) {
                    self.clear();
                }
                
                if i.key_pressed(egui::Key::R) {
                    self.randomize();
                }
                
                if i.key_pressed(egui::Key::G) {
                    self.show_grid = !self.show_grid;
                }
                
                if i.key_pressed(egui::Key::Num1) {
                    self.left_panel_open = !self.left_panel_open;
                }
                
                if i.key_pressed(egui::Key::Num2) {
                    self.right_panel_open = !self.right_panel_open;
                }
            });
            
            // Draw grid and cells
            let rect = response.rect;
            self.draw_grid(&painter, rect);
            self.draw_cells(&painter, rect);
            
            // Show zoom level
            if self.zoom != 1.0 {
                let zoom_text = format!("Zoom: {:.1}x", self.zoom);
                painter.text(
                    rect.min + Vec2::new(10.0, 10.0),
                    egui::Align2::LEFT_TOP,
                    zoom_text,
                    egui::FontId::monospace(12.0),
                    Color32::WHITE,
                );
            }
        });
    }
}