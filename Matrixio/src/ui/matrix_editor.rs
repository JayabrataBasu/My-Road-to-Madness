use eframe::egui;
use crate::matrix::Matrix;
use crate::data::project::Project;
use anyhow::Result;

/// Matrix editor component for creating and editing matrices
pub struct MatrixEditor {
    // Editor state
    editing_matrix: Option<String>,
    matrix_name: String,
    rows: usize,
    cols: usize,
    data: Vec<Vec<String>>, // String for user input
    
    // UI state
    show_generator: bool,
    generator_type: GeneratorType,
    
    // Error handling
    validation_errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
enum GeneratorType {
    Zeros,
    Ones,
    Identity,
    Random,
    Diagonal,
    Hilbert,
    Vandermonde,
}

impl MatrixEditor {
    pub fn new() -> Self {
        Self {
            editing_matrix: None,
            matrix_name: String::new(),
            rows: 3,
            cols: 3,
            data: vec![vec!["0".to_string(); 3]; 3],
            
            show_generator: false,
            generator_type: GeneratorType::Zeros,
            
            validation_errors: Vec::new(),
        }
    }

    pub fn start_new_matrix(&mut self) {
        self.editing_matrix = None;
        self.matrix_name.clear();
        self.rows = 3;
        self.cols = 3;
        self.data = vec![vec!["0".to_string(); 3]; 3];
        self.validation_errors.clear();
    }

    pub fn edit_matrix(&mut self, name: String, matrix: Matrix) {
        self.editing_matrix = Some(name.clone());
        self.matrix_name = name;
        let (rows, cols) = matrix.dimensions();
        self.rows = rows;
        self.cols = cols;
        
        // Convert matrix data to string format for editing
        self.data = Vec::new();
        for i in 0..rows {
            let mut row = Vec::new();
            for j in 0..cols {
                let value = matrix.get(i, j).unwrap_or(0.0);
                row.push(format!("{:.6}", value));
            }
            self.data.push(row);
        }
        
        self.validation_errors.clear();
    }

    pub fn show_generator(&mut self) {
        self.show_generator = true;
    }

    pub fn render(&mut self, ui: &mut egui::Ui, project: &mut Project) {
        ui.heading("Matrix Editor");
        
        // Matrix name input
        ui.horizontal(|ui| {
            ui.label("Name:");
            ui.text_edit_singleline(&mut self.matrix_name);
        });

        // Dimensions input
        ui.horizontal(|ui| {
            ui.label("Dimensions:");
            
            let mut rows_changed = false;
            let mut cols_changed = false;
            
            ui.add(egui::DragValue::new(&mut self.rows).range(1..=1000).prefix("Rows: "));
            if ui.input(|i| i.pointer.any_released()) {
                rows_changed = true;
            }
            
            ui.add(egui::DragValue::new(&mut self.cols).range(1..=1000).prefix("Cols: "));
            if ui.input(|i| i.pointer.any_released()) {
                cols_changed = true;
            }
            
            if rows_changed || cols_changed {
                self.resize_data();
            }
        });

        ui.separator();

        // Generator button
        if ui.button("🎲 Matrix Generator").clicked() {
            self.show_generator = true;
        }

        ui.separator();

        // Matrix data input
        self.render_matrix_input(ui);

        ui.separator();

        // Action buttons
        ui.horizontal(|ui| {
            let save_button_text = if self.editing_matrix.is_some() {
                "Update Matrix"
            } else {
                "Create Matrix"
            };

            if ui.button(save_button_text).clicked() {
                self.save_matrix(project);
            }

            if ui.button("Clear").clicked() {
                self.clear_data();
            }

            if self.editing_matrix.is_some() {
                if ui.button("Cancel").clicked() {
                    self.start_new_matrix();
                }
            }
        });

        // Show validation errors
        if !self.validation_errors.is_empty() {
            ui.separator();
            ui.label(egui::RichText::new("Validation Errors:").color(egui::Color32::RED));
            for error in &self.validation_errors {
                ui.label(egui::RichText::new(format!("• {}", error)).color(egui::Color32::RED));
            }
        }

        // Matrix generator modal
        if self.show_generator {
            self.render_generator_modal(ui.ctx());
        }
    }

    fn render_matrix_input(&mut self, ui: &mut egui::Ui) {
        ui.label("Matrix Data:");
        
        // Determine if we should show the full matrix or a preview
        let max_display_size = 20;
        let show_full = self.rows <= max_display_size && self.cols <= max_display_size;
        
        if show_full {
            self.render_full_matrix_input(ui);
        } else {
            self.render_large_matrix_preview(ui);
        }
    }

    fn render_full_matrix_input(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("matrix_grid")
            .striped(true)
            .show(ui, |ui| {
                for i in 0..self.rows {
                    for j in 0..self.cols {
                        let mut value = self.data[i][j].clone();
                        let response = ui.add(
                            egui::TextEdit::singleline(&mut value)
                                .desired_width(60.0)
                                .font(egui::TextStyle::Monospace)
                        );
                        
                        if response.changed() {
                            self.data[i][j] = value;
                        }
                    }
                    ui.end_row();
                }
            });
    }

    fn render_large_matrix_preview(&mut self, ui: &mut egui::Ui) {
        ui.label(format!("Matrix is too large to display fully ({}×{})", self.rows, self.cols));
        ui.label("Showing top-left 5×5 preview:");
        
        let preview_size = std::cmp::min(5, std::cmp::min(self.rows, self.cols));
        
        egui::Grid::new("matrix_preview_grid")
            .striped(true)
            .show(ui, |ui| {
                for i in 0..preview_size {
                    for j in 0..preview_size {
                        let mut value = self.data[i][j].clone();
                        let response = ui.add(
                            egui::TextEdit::singleline(&mut value)
                                .desired_width(60.0)
                                .font(egui::TextStyle::Monospace)
                        );
                        
                        if response.changed() {
                            self.data[i][j] = value;
                        }
                    }
                    
                    if preview_size < self.cols {
                        ui.label("...");
                    }
                    ui.end_row();
                }
                
                if preview_size < self.rows {
                    for _ in 0..preview_size {
                        ui.label("⋮");
                    }
                    if preview_size < self.cols {
                        ui.label("⋱");
                    }
                    ui.end_row();
                }
            });

        ui.separator();
        
        // For large matrices, provide bulk operations
        ui.label("Bulk Operations:");
        ui.horizontal(|ui| {
            if ui.button("Fill with zeros").clicked() {
                self.fill_with_value(0.0);
            }
            if ui.button("Fill with ones").clicked() {
                self.fill_with_value(1.0);
            }
            if ui.button("Random fill").clicked() {
                self.fill_random();
            }
        });
    }

    fn render_generator_modal(&mut self, ctx: &egui::Context) {
        egui::Window::new("Matrix Generator")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("Generate Matrix");
                
                ui.horizontal(|ui| {
                    ui.label("Type:");
                    egui::ComboBox::from_label("")
                        .selected_text(format!("{:?}", self.generator_type))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.generator_type, GeneratorType::Zeros, "Zeros");
                            ui.selectable_value(&mut self.generator_type, GeneratorType::Ones, "Ones");
                            ui.selectable_value(&mut self.generator_type, GeneratorType::Identity, "Identity");
                            ui.selectable_value(&mut self.generator_type, GeneratorType::Random, "Random");
                            ui.selectable_value(&mut self.generator_type, GeneratorType::Diagonal, "Diagonal");
                            ui.selectable_value(&mut self.generator_type, GeneratorType::Hilbert, "Hilbert");
                            ui.selectable_value(&mut self.generator_type, GeneratorType::Vandermonde, "Vandermonde");
                        });
                });

                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("Generate").clicked() {
                        self.generate_matrix();
                        self.show_generator = false;
                    }

                    if ui.button("Cancel").clicked() {
                        self.show_generator = false;
                    }
                });
            });
    }

    fn resize_data(&mut self) {
        // Resize the data matrix to match new dimensions
        self.data.resize(self.rows, Vec::new());
        for row in &mut self.data {
            row.resize(self.cols, "0".to_string());
        }
    }

    fn clear_data(&mut self) {
        for row in &mut self.data {
            for cell in row {
                *cell = "0".to_string();
            }
        }
    }

    fn fill_with_value(&mut self, value: f64) {
        let value_str = value.to_string();
        for row in &mut self.data {
            for cell in row {
                *cell = value_str.clone();
            }
        }
    }

    fn fill_random(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for row in &mut self.data {
            for cell in row {
                let random_value: f64 = rng.gen_range(-10.0..10.0);
                *cell = format!("{:.6}", random_value);
            }
        }
    }

    fn generate_matrix(&mut self) {
        match self.generator_type {
            GeneratorType::Zeros => self.fill_with_value(0.0),
            GeneratorType::Ones => self.fill_with_value(1.0),
            GeneratorType::Identity => self.generate_identity(),
            GeneratorType::Random => self.fill_random(),
            GeneratorType::Diagonal => self.generate_diagonal(),
            GeneratorType::Hilbert => self.generate_hilbert(),
            GeneratorType::Vandermonde => self.generate_vandermonde(),
        }
    }

    fn generate_identity(&mut self) {
        self.clear_data();
        let size = std::cmp::min(self.rows, self.cols);
        for i in 0..size {
            self.data[i][i] = "1".to_string();
        }
    }

    fn generate_diagonal(&mut self) {
        self.clear_data();
        let size = std::cmp::min(self.rows, self.cols);
        for i in 0..size {
            self.data[i][i] = (i + 1).to_string();
        }
    }

    fn generate_hilbert(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = 1.0 / ((i + j + 1) as f64);
                self.data[i][j] = format!("{:.6}", value);
            }
        }
    }

    fn generate_vandermonde(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let base = (i + 1) as f64;
                let value = base.powi(j as i32);
                self.data[i][j] = format!("{:.6}", value);
            }
        }
    }

    fn save_matrix(&mut self, project: &mut Project) {
        self.validation_errors.clear();

        // Validate name
        if self.matrix_name.trim().is_empty() {
            self.validation_errors.push("Matrix name cannot be empty".to_string());
        }

        // Check for duplicate name (if not editing)
        if self.editing_matrix.is_none() && project.matrices.contains_key(&self.matrix_name) {
            self.validation_errors.push("Matrix name already exists".to_string());
        }

        // Validate and parse data
        let mut parsed_data = Vec::new();
        for i in 0..self.rows {
            let mut row = Vec::new();
            for j in 0..self.cols {
                match self.data[i][j].trim().parse::<f64>() {
                    Ok(value) => row.push(value),
                    Err(_) => {
                        self.validation_errors.push(format!(
                            "Invalid number at position ({}, {}): '{}'", 
                            i + 1, j + 1, self.data[i][j]
                        ));
                        return;
                    }
                }
            }
            parsed_data.push(row);
        }

        if !self.validation_errors.is_empty() {
            return;
        }

        // Create matrix
        match Matrix::from_vec(parsed_data, self.matrix_name.clone()) {
            Ok(matrix) => {
                // Remove old matrix if editing
                if let Some(ref old_name) = self.editing_matrix {
                    project.matrices.remove(old_name);
                }
                
                project.matrices.insert(self.matrix_name.clone(), matrix);
                self.start_new_matrix(); // Reset editor
            }
            Err(e) => {
                self.validation_errors.push(format!("Failed to create matrix: {}", e));
            }
        }
    }
}