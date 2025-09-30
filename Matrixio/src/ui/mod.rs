use crate::data::project::Project;
use crate::matrix::{decomposition::*, Matrix};
use eframe::egui;

pub mod components;
pub mod matrix_editor;
pub mod operations_panel;
pub mod results_viewer;

/// Main application state
pub struct MatrixioApp {
    // Core application state
    project: Project,

    // UI state
    selected_matrix: Option<String>,

    // Panels and editors
    matrix_editor: matrix_editor::MatrixEditor,
    operations_panel: operations_panel::OperationsPanel,

    // Operation results
    operation_results: Vec<(String, OperationResult)>,
    show_results: bool,

    // Custom matrix creation fields
    custom_rows: String,
    custom_cols: String,
    matrix_name_input: String,

    // Matrix element editing state
    matrix_element_texts:
        std::collections::HashMap<String, std::collections::HashMap<(usize, usize), String>>,
    editing_element: Option<(String, usize, usize)>, // matrix_name, row, col

    // Modal states
    show_about: bool,

    // Messages
    success_message: Option<String>,
    error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub enum OperationResult {
    Matrix(Matrix),
    Scalar(f64),
    Vector(Vec<f64>),
    Text(String),
    Decomposition(DecompositionResult),
}

#[derive(Debug, Clone)]
pub enum DecompositionResult {
    LU(LUDecomposition),
    QR(QRDecomposition),
    SVD(SVDDecomposition),
    Eigen(EigenDecomposition),
    Cholesky(CholeskyDecomposition),
}

impl MatrixioApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            project: Project::new(),
            selected_matrix: None,
            matrix_editor: matrix_editor::MatrixEditor::new(),
            operations_panel: operations_panel::OperationsPanel::new(),
            operation_results: Vec::new(),
            show_results: false,
            custom_rows: "3".to_string(),
            custom_cols: "3".to_string(),
            matrix_name_input: String::new(),
            matrix_element_texts: std::collections::HashMap::new(),
            editing_element: None,
            show_about: false,
            success_message: None,
            error_message: None,
        }
    }

    fn show_success(&mut self, message: String) {
        self.success_message = Some(message);
        self.error_message = None;
    }

    fn show_error(&mut self, message: String) {
        self.error_message = Some(message);
        self.success_message = None;
    }
}

impl eframe::App for MatrixioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                // File menu
                ui.menu_button("File", |ui| {
                    if ui.button("🆕 New Project").clicked() {
                        self.project = Project::new();
                        self.selected_matrix = None;
                        ui.close_menu();
                    }

                    if ui.button("🚪 Exit").clicked() {
                        std::process::exit(0);
                    }
                });

                // Help menu
                ui.menu_button("Help", |ui| {
                    if ui.button("❓ About").clicked() {
                        self.show_about = true;
                        ui.close_menu();
                    }
                });
            });
        });

        // Main content
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Left panel: Matrix list and editor
                ui.vertical(|ui| {
                    ui.set_width(300.0);

                    self.render_matrix_list(ui);

                    ui.separator();

                    self.render_matrix_editor(ui);
                });

                ui.separator();

                // Right panel: Matrix display and operations
                ui.vertical(|ui| {
                    if let Some(selected) = &self.selected_matrix {
                        let selected_clone = selected.clone();
                        let has_matrix = self.project.matrices.contains_key(&selected_clone);

                        if has_matrix {
                            ui.heading(format!("Matrix: {}", selected_clone));

                            // Display matrix content (editable) - split borrowing
                            ui.separator();

                            // Get matrix info first to avoid borrowing conflicts
                            let matrix_info =
                                if let Some(matrix) = self.project.matrices.get(&selected_clone) {
                                    Some((matrix.dimensions(), matrix.name.clone()))
                                } else {
                                    None
                                };

                            if let Some(((rows, cols), matrix_name)) = matrix_info {
                                ui.horizontal(|ui| {
                                    ui.label(format!("Size: {}×{}", rows, cols));
                                    ui.label(format!("Name: {}", matrix_name));

                                    // Show editing mode indicator
                                    if rows <= 10 && cols <= 10 {
                                        ui.label("🖊️ All cells editable");
                                    } else {
                                        ui.label("📋 Scroll to edit cells");
                                    }
                                });
                                ui.add_space(10.0);

                                // Render matrix based on size
                                self.render_editable_matrix(ui, &selected_clone);
                            }

                            ui.separator();
                            ui.label("Operations:");

                            // Get operation result
                            let operation_result =
                                self.operations_panel
                                    .show(ui, &self.project, &selected_clone);

                            if let Some(result) = operation_result {
                                // Store the result
                                let result_name =
                                    format!("Result_{}", self.operation_results.len() + 1);
                                self.operation_results.push((result_name.clone(), result));
                                self.success_message =
                                    Some(format!("Operation completed: {}", result_name));
                            }

                            // Results section
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.label("Results:");
                                ui.checkbox(&mut self.show_results, "Show Details");
                            });

                            if self.show_results {
                                self.render_operation_results(ui);
                            } else if !self.operation_results.is_empty() {
                                ui.label(format!(
                                    "{} result(s) available",
                                    self.operation_results.len()
                                ));
                            }
                        } else {
                            ui.label("Selected matrix no longer exists");
                            self.selected_matrix = None;
                        }
                    } else {
                        ui.vertical_centered(|ui| {
                            ui.add_space(50.0);
                            ui.label("Select a matrix to perform operations");
                        });
                    }
                });
            });
        });

        // Show messages
        if let Some(message) = &self.success_message {
            egui::TopBottomPanel::bottom("success_bar").show(ctx, |ui| {
                ui.colored_label(egui::Color32::GREEN, format!("✓ {}", message));
            });
        }

        if let Some(message) = &self.error_message {
            egui::TopBottomPanel::bottom("error_bar").show(ctx, |ui| {
                ui.colored_label(egui::Color32::RED, format!("✗ {}", message));
            });
        }

        // Render modal dialogs
        self.render_modals(ctx);
    }
}

impl MatrixioApp {
    fn render_matrix_list(&mut self, ui: &mut egui::Ui) {
        ui.heading("Matrices");

        // Create a list of matrix names to avoid borrowing issues
        let matrix_names: Vec<String> = self.project.matrices.keys().cloned().collect();

        for name in matrix_names {
            if let Some(matrix) = self.project.matrices.get(&name) {
                let is_selected = self.selected_matrix.as_ref() == Some(&name);

                let response = ui.selectable_label(
                    is_selected,
                    format!(
                        "{} ({}×{})",
                        name,
                        matrix.dimensions().0,
                        matrix.dimensions().1
                    ),
                );

                if response.clicked() {
                    self.selected_matrix = Some(name.clone());
                }
            }
        }

        if self.project.matrices.is_empty() {
            ui.label("No matrices yet. Create one using the editor below.");
        }
    }

    fn render_matrix_editor(&mut self, ui: &mut egui::Ui) {
        ui.heading("Matrix Editor");

        // Quick creation buttons
        ui.label("Quick Create:");
        ui.horizontal(|ui| {
            if ui.button("3x3 Identity").clicked() {
                let matrix = Matrix::identity(3, "temp".to_string());
                let name = format!("identity_3x3_{}", self.project.matrices.len());
                self.project.matrices.insert(name.clone(), matrix);
                self.selected_matrix = Some(name.clone());
                self.show_success(format!("Created {}", name));
            }

            if ui.button("4x4 Identity").clicked() {
                let matrix = Matrix::identity(4, "temp".to_string());
                let name = format!("identity_4x4_{}", self.project.matrices.len());
                self.project.matrices.insert(name.clone(), matrix);
                self.selected_matrix = Some(name.clone());
                self.show_success(format!("Created {}", name));
            }
        });

        ui.horizontal(|ui| {
            if ui.button("3x3 Random").clicked() {
                let matrix = Matrix::random(3, 3, "temp".to_string());
                let name = format!("random_3x3_{}", self.project.matrices.len());
                self.project.matrices.insert(name.clone(), matrix);
                self.selected_matrix = Some(name.clone());
                self.show_success(format!("Created {}", name));
            }

            if ui.button("4x4 Random").clicked() {
                let matrix = Matrix::random(4, 4, "temp".to_string());
                let name = format!("random_4x4_{}", self.project.matrices.len());
                self.project.matrices.insert(name.clone(), matrix);
                self.selected_matrix = Some(name.clone());
                self.show_success(format!("Created {}", name));
            }
        });

        ui.separator();

        // Custom matrix creation
        ui.label("Custom Matrix Creation:");

        ui.horizontal(|ui| {
            ui.label("Rows:");
            ui.add(
                egui::TextEdit::singleline(&mut self.custom_rows)
                    .desired_width(60.0)
                    .hint_text("1-200"),
            );

            ui.label("Columns:");
            ui.add(
                egui::TextEdit::singleline(&mut self.custom_cols)
                    .desired_width(60.0)
                    .hint_text("1-200"),
            );
        });

        ui.horizontal(|ui| {
            ui.label("Name (optional):");
            ui.add(
                egui::TextEdit::singleline(&mut self.matrix_name_input)
                    .desired_width(150.0)
                    .hint_text("Leave empty for auto-name"),
            );
        });

        ui.horizontal(|ui| {
            if ui.button("Create Zero Matrix").clicked() {
                self.create_custom_matrix(false);
            }

            if ui.button("Create Random Matrix").clicked() {
                self.create_custom_matrix(true);
            }
        });

        // Show validation errors
        if let Some(error) = &self.error_message {
            ui.colored_label(egui::Color32::RED, error);
        }
    }

    fn render_operation_results(&mut self, ui: &mut egui::Ui) {
        // Collect actions to perform after rendering to avoid borrowing conflicts
        let mut matrix_to_add: Option<(String, Matrix)> = None;
        let mut should_clear_results = false;

        egui::ScrollArea::vertical()
            .max_height(200.0)
            .show(ui, |ui| {
                for (name, result) in &self.operation_results {
                    ui.collapsing(name, |ui| {
                        match result {
                            OperationResult::Matrix(matrix) => {
                                let (rows, cols) = matrix.dimensions();
                                ui.label(format!("Result Matrix ({}×{})", rows, cols));

                                // Display result matrix (read-only)
                                egui::Grid::new(format!("result_grid_{}", name))
                                    .striped(true)
                                    .show(ui, |ui| {
                                        for i in 0..rows.min(5) {
                                            for j in 0..cols.min(5) {
                                                if let Some(value) = matrix.get(i, j) {
                                                    ui.label(format!("{:.3}", value));
                                                }
                                            }
                                            ui.end_row();
                                        }
                                        if rows > 5 || cols > 5 {
                                            ui.label("...");
                                        }
                                    });

                                if ui.button("Add to Project").clicked() {
                                    let new_name =
                                        format!("result_{}", self.operation_results.len());
                                    let mut new_matrix = matrix.clone();
                                    new_matrix.name = new_name.clone();
                                    matrix_to_add = Some((new_name, new_matrix));
                                }
                            }
                            OperationResult::Scalar(value) => {
                                ui.label(format!("Scalar Result: {:.6}", value));
                            }
                            OperationResult::Vector(vec) => {
                                ui.label(format!("Vector Result (length {})", vec.len()));
                                for (i, value) in vec.iter().enumerate().take(10) {
                                    ui.label(format!("[{}]: {:.3}", i, value));
                                }
                                if vec.len() > 10 {
                                    ui.label("...");
                                }
                            }
                            OperationResult::Text(text) => {
                                ui.label(text);
                            }
                            OperationResult::Decomposition(_) => {
                                ui.label("Decomposition result (details coming soon)");
                            }
                        }
                    });
                }

                if !self.operation_results.is_empty() {
                    if ui.button("Clear All Results").clicked() {
                        should_clear_results = true;
                    }
                }
            });

        // Perform actions after rendering to avoid borrowing conflicts
        if let Some((name, matrix)) = matrix_to_add {
            self.project.matrices.insert(name.clone(), matrix);
            self.success_message = Some(format!("Added {} to project", name));
        }

        if should_clear_results {
            self.operation_results.clear();
            self.success_message = Some("All results cleared".to_string());
        }
    }

    fn create_custom_matrix(&mut self, random: bool) {
        // Clear any previous error messages
        self.error_message = None;

        // Parse and validate input dimensions
        let rows = match self.custom_rows.trim().parse::<usize>() {
            Ok(r) if r > 0 && r <= 200 => r,
            Ok(r) if r == 0 => {
                self.show_error("Rows must be greater than 0".to_string());
                return;
            }
            Ok(_) => {
                self.show_error(
                    "Maximum matrix size is 200×200 for performance reasons".to_string(),
                );
                return;
            }
            Err(_) => {
                self.show_error("Please enter a valid number for rows".to_string());
                return;
            }
        };

        let cols = match self.custom_cols.trim().parse::<usize>() {
            Ok(c) if c > 0 && c <= 200 => c,
            Ok(c) if c == 0 => {
                self.show_error("Columns must be greater than 0".to_string());
                return;
            }
            Ok(_) => {
                self.show_error(
                    "Maximum matrix size is 200×200 for performance reasons".to_string(),
                );
                return;
            }
            Err(_) => {
                self.show_error("Please enter a valid number for columns".to_string());
                return;
            }
        };

        // Check total size for memory safety
        if rows * cols > 40000 {
            self.show_error("Matrix too large (max 40,000 elements for performance)".to_string());
            return;
        }

        // Generate matrix name
        let matrix_name = if self.matrix_name_input.trim().is_empty() {
            if random {
                format!("random_{}x{}_{}", rows, cols, self.project.matrices.len())
            } else {
                format!("zeros_{}x{}_{}", rows, cols, self.project.matrices.len())
            }
        } else {
            let base_name = self.matrix_name_input.trim().to_string();
            // Check if name already exists and add suffix if needed
            if self.project.matrices.contains_key(&base_name) {
                format!("{}_{}", base_name, self.project.matrices.len())
            } else {
                base_name
            }
        };

        // Create the matrix
        let matrix = if random {
            Matrix::random(rows, cols, matrix_name.clone())
        } else {
            // Create zero matrix using Matrix::new (which creates zeros by default)
            Matrix::new(rows, cols, matrix_name.clone())
        };

        // Add to project and select it
        self.project.matrices.insert(matrix_name.clone(), matrix);
        self.selected_matrix = Some(matrix_name.clone());

        // Clear input fields for next use
        self.matrix_name_input.clear();

        // Show success message
        let matrix_type = if random { "random" } else { "zero" };
        self.show_success(format!(
            "Created {} matrix '{}' ({}×{})",
            matrix_type, matrix_name, rows, cols
        ));
    }

    fn render_modals(&mut self, ctx: &egui::Context) {
        // About dialog
        if self.show_about {
            egui::Window::new("About Matrixio")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Matrixio v0.1.0");
                    ui.label("High-performance matrix calculation software");
                    ui.label("Built with Rust and egui");

                    ui.separator();

                    if ui.button("Close").clicked() {
                        self.show_about = false;
                    }
                });
        }
    }

    fn render_editable_matrix(&mut self, ui: &mut egui::Ui, matrix_name: &str) {
        if let Some(matrix) = self.project.matrices.get(matrix_name) {
            let (rows, cols) = matrix.dimensions();

            // Initialize text storage for this matrix if it doesn't exist
            if !self.matrix_element_texts.contains_key(matrix_name) {
                self.matrix_element_texts
                    .insert(matrix_name.to_string(), std::collections::HashMap::new());
            }

            // Choose rendering strategy based on matrix size
            if rows <= 10 && cols <= 10 {
                // Small matrix: render all cells as editable text fields
                self.render_small_matrix_editor(ui, matrix_name, rows, cols);
            } else {
                // Large matrix: render with scrolling
                self.render_large_matrix_editor(ui, matrix_name, rows, cols);
            }
        }
    }

    fn render_small_matrix_editor(
        &mut self,
        ui: &mut egui::Ui,
        matrix_name: &str,
        rows: usize,
        cols: usize,
    ) {
        ui.label("📝 Direct editing mode - click any cell to edit");
        ui.add_space(5.0);

        // Fixed-size grid for small matrices
        egui::Grid::new(format!("small_matrix_grid_{}", matrix_name))
            .striped(true)
            .spacing([4.0, 4.0])
            .show(ui, |ui| {
                // Header row with column indices
                ui.label(""); // Empty corner
                for j in 0..cols {
                    ui.label(format!("Col {}", j));
                }
                ui.end_row();

                // Matrix rows with row indices
                for i in 0..rows {
                    // Row index
                    ui.label(format!("Row {}", i));

                    // Matrix elements
                    for j in 0..cols {
                        self.render_matrix_cell(ui, matrix_name, i, j, true);
                    }
                    ui.end_row();
                }
            });
    }

    fn render_large_matrix_editor(
        &mut self,
        ui: &mut egui::Ui,
        matrix_name: &str,
        rows: usize,
        cols: usize,
    ) {
        ui.label("📋 Scrollable view - scroll to see more cells");
        ui.label(format!(
            "Showing visible portion of {}×{} matrix",
            rows, cols
        ));
        ui.add_space(5.0);

        // Scrollable area for large matrices
        egui::ScrollArea::both()
            .max_height(400.0)
            .max_width(600.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new(format!("large_matrix_grid_{}", matrix_name))
                    .striped(true)
                    .spacing([3.0, 3.0])
                    .min_col_width(70.0)
                    .show(ui, |ui| {
                        // Header row with column indices (show more columns for scrolling)
                        ui.label(""); // Empty corner
                        for j in 0..cols.min(50) {
                            // Limit to 50 columns for performance
                            ui.label(format!("C{}", j));
                        }
                        if cols > 50 {
                            ui.label("...");
                        }
                        ui.end_row();

                        // Matrix rows (show more rows for scrolling)
                        for i in 0..rows.min(100) {
                            // Limit to 100 rows for performance
                            // Row index
                            ui.label(format!("R{}", i));

                            // Matrix elements
                            for j in 0..cols.min(50) {
                                self.render_matrix_cell(ui, matrix_name, i, j, false);
                            }

                            if cols > 50 {
                                ui.label("...");
                            }
                            ui.end_row();
                        }

                        if rows > 100 {
                            ui.label("...");
                            for _ in 0..cols.min(50) {
                                ui.label("...");
                            }
                            ui.end_row();
                        }
                    });
            });
    }

    fn render_matrix_cell(
        &mut self,
        ui: &mut egui::Ui,
        matrix_name: &str,
        row: usize,
        col: usize,
        is_small_matrix: bool,
    ) {
        let cell_key = (row, col);

        // Get current matrix value
        let current_value = if let Some(matrix) = self.project.matrices.get(matrix_name) {
            matrix.get(row, col).unwrap_or(0.0)
        } else {
            return; // Matrix doesn't exist
        };

        // Get or initialize the text for this cell
        let mut text = {
            let text_map = self.matrix_element_texts.get_mut(matrix_name).unwrap();
            text_map
                .get(&cell_key)
                .cloned()
                .unwrap_or_else(|| format!("{:.3}", current_value))
        };

        // Determine cell width based on matrix size
        let cell_width = if is_small_matrix { 80.0 } else { 65.0 };

        // Create the text edit widget
        let text_edit = egui::TextEdit::singleline(&mut text)
            .desired_width(cell_width)
            .font(egui::TextStyle::Monospace)
            .hint_text("0.0");

        let response = ui.add(text_edit);

        // Handle text changes with validation
        let mut update_result: Option<Result<f64, String>> = None;

        if response.changed() {
            // Store the current text (even if invalid) to maintain user input
            if let Some(text_map) = self.matrix_element_texts.get_mut(matrix_name) {
                text_map.insert(cell_key, text.clone());
            }

            // Validate the input
            match text.trim().parse::<f64>() {
                Ok(new_value) => {
                    if new_value.is_finite() {
                        update_result = Some(Ok(new_value));
                    } else {
                        update_result = Some(Err("Please enter a finite number".to_string()));
                    }
                }
                Err(_) => {
                    if !text.trim().is_empty() {
                        update_result = Some(Err("Invalid number format".to_string()));
                    }
                }
            }
        }

        // Apply matrix update and show errors (separate borrowing scope)
        if let Some(result) = update_result {
            match result {
                Ok(new_value) => {
                    if let Some(matrix) = self.project.matrices.get_mut(matrix_name) {
                        if let Err(_) = matrix.set(row, col, new_value) {
                            self.show_error(format!(
                                "Failed to set matrix element at ({}, {})",
                                row, col
                            ));
                        }
                    }
                }
                Err(error_msg) => {
                    self.show_error(error_msg);
                }
            }
        }

        // Update text when losing focus if the value was successfully parsed
        if response.lost_focus() {
            if let Ok(parsed_value) = text.trim().parse::<f64>() {
                if parsed_value.is_finite() {
                    // Update the display text to match the stored value
                    if let Some(matrix) = self.project.matrices.get(matrix_name) {
                        let stored_value = matrix.get(row, col).unwrap_or(0.0);
                        if let Some(text_map) = self.matrix_element_texts.get_mut(matrix_name) {
                            text_map.insert(cell_key, format!("{:.3}", stored_value));
                        }
                    }
                }
            } else if !text.trim().is_empty() {
                // Reset to current matrix value if input was invalid
                if let Some(text_map) = self.matrix_element_texts.get_mut(matrix_name) {
                    text_map.insert(cell_key, format!("{:.3}", current_value));
                }
            }
        }

        // Visual feedback for focused/active cells
        if response.has_focus() {
            self.editing_element = Some((matrix_name.to_string(), row, col));
        }

        // Add tooltip with cell coordinates and current value
        if response.hovered() {
            response.on_hover_text(format!(
                "Cell ({}, {})\nValue: {:.6}\nClick to edit",
                row, col, current_value
            ));
        }
    }
}
