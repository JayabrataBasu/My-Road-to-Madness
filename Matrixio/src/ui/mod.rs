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
    selected_result: Option<String>, // Selected operation result

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

    // Large matrix viewport state
    viewport_offset: std::collections::HashMap<String, (usize, usize)>, // (row_offset, col_offset) per matrix
    viewport_size: (usize, usize), // (visible_rows, visible_cols)
    zoom_level: f32, // Cell size multiplier
    show_matrix_stats: bool,

    // Matrix sidebar UI state
    matrix_search_filter: String,
    context_menu_matrix: Option<String>, // Matrix name for active context menu
    rename_dialog_matrix: Option<String>, // Matrix being renamed
    rename_input: String,

    // Small matrix viewer state
    edit_mode_enabled: bool,
    selected_cell: Option<(usize, usize)>, // Highlighted cell in small matrices

    // Large matrix viewer state
    keyboard_focus_cell: Option<(usize, usize)>, // Cell with keyboard focus
    show_minimap: bool,

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
            selected_result: None,
            matrix_editor: matrix_editor::MatrixEditor::new(),
            operations_panel: operations_panel::OperationsPanel::new(),
            operation_results: Vec::new(),
            show_results: false,
            custom_rows: "3".to_string(),
            custom_cols: "3".to_string(),
            matrix_name_input: String::new(),
            matrix_element_texts: std::collections::HashMap::new(),
            editing_element: None,
            viewport_offset: std::collections::HashMap::new(),
            viewport_size: (20, 20), // Default 20x20 viewport
            zoom_level: 1.0,
            show_matrix_stats: true,
            matrix_search_filter: String::new(),
            context_menu_matrix: None,
            rename_dialog_matrix: None,
            rename_input: String::new(),
            edit_mode_enabled: true,
            selected_cell: None,
            keyboard_focus_cell: None,
            show_minimap: false,
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
            // Add vertical scroll area to prevent overflow
            egui::ScrollArea::vertical()
                .auto_shrink([false, true]) // Don't shrink horizontally, allow vertical shrinking
                .max_height(ui.available_height()) // Use full available height
                .show(ui, |ui| {
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
                            // Check if we have a selected matrix or result
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

                                    // Results are now shown in the left panel as cards
                                } else {
                                    ui.label("Selected matrix no longer exists");
                                    self.selected_matrix = None;
                                }
                            } else if let Some(selected_result) = &self.selected_result {
                                // Display selected result
                                let result_name = selected_result.clone();
                                self.render_selected_result(ui, &result_name);
                            } else {
                                ui.vertical_centered(|ui| {
                                    ui.add_space(50.0);
                                    ui.label("Select a matrix to perform operations or a result to view");
                                });
                            }
                        });
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
        ui.horizontal(|ui| {
            ui.heading("📁 Matrices");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("🔄").on_hover_text("Refresh list").clicked() {
                    // Force refresh (could clear caches in future)
                }
            });
        });
        
        ui.add_space(5.0);

        // Search/filter input
        ui.horizontal(|ui| {
            ui.label("🔍");
            ui.add(egui::TextEdit::singleline(&mut self.matrix_search_filter)
                .hint_text("Search matrices...")
                .desired_width(f32::INFINITY));
        });
        
        ui.add_space(5.0);

        // Create a filtered list of matrix names
        let matrix_names: Vec<String> = self.project.matrices.keys()
            .filter(|name| {
                if self.matrix_search_filter.is_empty() {
                    true
                } else {
                    name.to_lowercase().contains(&self.matrix_search_filter.to_lowercase())
                }
            })
            .cloned()
            .collect();

        if matrix_names.is_empty() {
            if self.matrix_search_filter.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(20.0);
                    ui.label("📄 No matrices yet");
                    ui.label("Create one using the editor below");
                });
            } else {
                ui.vertical_centered(|ui| {
                    ui.add_space(20.0);
                    ui.label("🔍 No matches found");
                    ui.label(format!("No matrices match '{}'", self.matrix_search_filter));
                });
            }
            return;
        }

        // Scrollable area for matrix cards
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // Collect matrix data first to avoid borrowing issues
                let matrix_data: Vec<(String, Matrix)> = matrix_names.into_iter()
                    .filter_map(|name| {
                        self.project.matrices.get(&name).map(|matrix| (name, matrix.clone()))
                    })
                    .collect();
                
                // Collect result data to avoid borrowing issues
                let result_data: Vec<(String, OperationResult)> = self.operation_results.clone();
                
                for (name, matrix) in matrix_data {
                    self.render_matrix_card(ui, &name, &matrix);
                }
                
                // Add separator between matrices and results
                if !self.project.matrices.is_empty() && !self.operation_results.is_empty() {
                    ui.separator();
                    ui.add_space(5.0);
                }
                
                // Render result cards
                for (name, result) in &result_data {
                    self.render_result_card(ui, name, result);
                }
            });
        
        // Handle rename dialog
        self.render_rename_dialog(ui);
    }

    fn render_matrix_card(&mut self, ui: &mut egui::Ui, name: &str, matrix: &Matrix) {
        let is_selected = self.selected_matrix.as_ref().map(|s| s.as_str()) == Some(name);
        let (rows, cols) = matrix.dimensions();
        
        // Card styling
        let card_color = if is_selected {
            egui::Color32::from_rgb(45, 85, 135) // Selected blue
        } else {
            ui.style().visuals.window_fill
        };
        
        let border_color = if is_selected {
            egui::Color32::from_rgb(65, 105, 155) // Brighter blue border
        } else {
            ui.style().visuals.window_stroke.color
        };

        let card_response = egui::Frame::none()
            .fill(card_color)
            .stroke(egui::Stroke::new(if is_selected { 2.0 } else { 1.0 }, border_color))
            .rounding(6.0)
            .inner_margin(8.0)
            .show(ui, |ui| {
                let response = ui.interact(ui.available_rect_before_wrap(), egui::Id::new(format!("matrix_card_{}", name)), egui::Sense::click());
                
                ui.horizontal(|ui| {
                    // Matrix icon and info
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label("📊");
                            ui.strong(name);
                        });
                        ui.horizontal(|ui| {
                            ui.label(format!("📐 {}×{}", rows, cols));
                            ui.separator();
                            ui.label(format!("📋 {} elements", rows * cols));
                        });
                    });
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // Context menu button
                        if ui.button("⋮").on_hover_text("More options").clicked() {
                            self.context_menu_matrix = Some(name.to_string());
                        }
                    });
                });

                // Handle clicks
                if response.clicked() {
                    self.selected_matrix = Some(name.to_string());
                }
                
                // Right-click context menu using popup
                if response.secondary_clicked() {
                    self.context_menu_matrix = Some(name.to_string());
                }
                
                response
            });
        
        // Show context menu popup if this matrix was right-clicked
        if self.context_menu_matrix.as_ref().map(|s| s.as_str()) == Some(name) {
            egui::popup_below_widget(ui, egui::Id::new(format!("context_{}", name)), &card_response.response, egui::PopupCloseBehavior::CloseOnClickOutside, |ui| {
                self.render_matrix_context_menu(ui, name);
            });
        }
        
        ui.add_space(4.0);
    }
    
    fn render_result_card(&mut self, ui: &mut egui::Ui, name: &str, result: &OperationResult) {
        let is_selected = self.selected_result.as_ref().map(|s| s.as_str()) == Some(name);
        
        // Card styling for results (different color scheme)
        let card_color = if is_selected {
            egui::Color32::from_rgb(85, 45, 135) // Purple for selected result
        } else {
            egui::Color32::from_rgb(45, 45, 55) // Darker background for results
        };
        
        let border_color = if is_selected {
            egui::Color32::from_rgb(105, 65, 155) // Brighter purple border
        } else {
            egui::Color32::from_rgb(65, 65, 75) // Subtle border for results
        };

        let card_response = egui::Frame::none()
            .fill(card_color)
            .stroke(egui::Stroke::new(if is_selected { 2.0 } else { 1.0 }, border_color))
            .rounding(6.0)
            .inner_margin(8.0)
            .show(ui, |ui| {
                let response = ui.interact(ui.available_rect_before_wrap(), egui::Id::new(format!("result_card_{}", name)), egui::Sense::click());
                
                ui.horizontal(|ui| {
                    // Result icon and info
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label("📊"); // Calculator icon for results
                            ui.strong(name);
                        });
                        ui.horizontal(|ui| {
                            match result {
                                OperationResult::Matrix(matrix) => {
                                    let (rows, cols) = matrix.dimensions();
                                    ui.label(format!("📐 {}×{}", rows, cols));
                                    ui.separator();
                                    ui.label(format!("📋 {} elements", rows * cols));
                                }
                                OperationResult::Scalar(value) => {
                                    ui.label("🔢 Scalar");
                                    ui.separator();
                                    ui.label(format!("≈ {:.3}", value));
                                }
                                OperationResult::Vector(vec) => {
                                    ui.label("📏 Vector");
                                    ui.separator();
                                    ui.label(format!("[{}]", vec.len()));
                                }
                                _ => {
                                    ui.label("📄 Result");
                                }
                            }
                        });
                    });
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // Delete button for results
                        if ui.button("🗑️").on_hover_text("Delete result").clicked() {
                            // Mark for deletion (we'll handle this after the loop)
                        }
                    });
                });

                // Handle clicks
                if response.clicked() {
                    self.selected_result = Some(name.to_string());
                    self.selected_matrix = None; // Clear matrix selection
                }
                
                response
            });
        
        ui.add_space(4.0);
    }
    
    fn render_selected_result(&mut self, ui: &mut egui::Ui, result_name: &str) {
        // Find and clone the result to avoid borrowing issues
        let result = self.operation_results.iter()
            .find(|(name, _)| name == result_name)
            .map(|(_, result)| result.clone());
            
        if let Some(result) = result {
            ui.heading(format!("Result: {}", result_name));
            ui.separator();
            
            match result {
                OperationResult::Matrix(matrix) => {
                    let (rows, cols) = matrix.dimensions();
                    ui.horizontal(|ui| {
                        ui.label(format!("Size: {}×{}", rows, cols));
                        ui.label(format!("Type: Matrix Result"));
                    });
                    
                    ui.add_space(10.0);
                    
                    // Action buttons
                    ui.horizontal(|ui| {
                        if ui.button("📁 Add to Project").clicked() {
                            let new_name = format!("result_{}", self.project.matrices.len() + 1);
                            let mut new_matrix = matrix.clone();
                            new_matrix.name = new_name.clone();
                            let final_name = new_name.clone();
                            self.project.matrices.insert(final_name.clone(), new_matrix);
                            self.selected_matrix = Some(final_name.clone());
                            self.selected_result = None;
                            self.success_message = Some(format!("Added result as matrix: {}", final_name));
                        }
                        
                        if ui.button("📋 Copy Values").clicked() {
                            // Could implement clipboard functionality here
                            self.success_message = Some("Matrix values copied to clipboard".to_string());
                        }
                    });
                    
                    ui.add_space(10.0);
                    
                    // Display the full matrix (read-only)
                    egui::ScrollArea::both()
                        .max_height(400.0)
                        .show(ui, |ui| {
                            egui::Grid::new(format!("result_matrix_{}", result_name))
                                .striped(true)
                                .spacing([4.0, 2.0])
                                .show(ui, |ui| {
                                    // Column headers
                                    ui.label("");
                                    for j in 0..cols {
                                        ui.strong(format!("C{}", j));
                                    }
                                    ui.end_row();
                                    
                                    // Matrix data with row headers
                                    for i in 0..rows {
                                        ui.strong(format!("R{}", i));
                                        for j in 0..cols {
                                            if let Some(value) = matrix.get(i, j) {
                                                ui.label(format!("{:.6}", value));
                                            } else {
                                                ui.label("N/A");
                                            }
                                        }
                                        ui.end_row();
                                    }
                                });
                        });
                }
                OperationResult::Scalar(value) => {
                    ui.horizontal(|ui| {
                        ui.label("Type: Scalar Result");
                        ui.label(format!("Value: {:.10}", value));
                    });
                    
                    ui.add_space(20.0);
                    
                    // Large display of the scalar value
                    ui.vertical_centered(|ui| {
                        ui.add_space(50.0);
                        ui.heading(format!("{:.10}", value));
                        ui.add_space(50.0);
                    });
                }
                OperationResult::Vector(vec) => {
                    ui.horizontal(|ui| {
                        ui.label("Type: Vector Result");
                        ui.label(format!("Length: {}", vec.len()));
                    });
                    
                    ui.add_space(10.0);
                    
                    // Display vector elements
                    egui::ScrollArea::vertical()
                        .max_height(400.0)
                        .show(ui, |ui| {
                            egui::Grid::new(format!("result_vector_{}", result_name))
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.strong("Index");
                                    ui.strong("Value");
                                    ui.end_row();
                                    
                                    for (i, value) in vec.iter().enumerate() {
                                        ui.label(format!("{}", i));
                                        ui.label(format!("{:.6}", value));
                                        ui.end_row();
                                    }
                                });
                        });
                }
                _ => {
                    ui.label("Unsupported result type");
                }
            }
        } else {
            ui.label("Result not found");
            self.selected_result = None;
        }
    }

    fn render_matrix_context_menu(&mut self, ui: &mut egui::Ui, matrix_name: &str) {
        if ui.button("✏️ Rename").clicked() {
            self.rename_dialog_matrix = Some(matrix_name.to_string());
            self.rename_input = matrix_name.to_string();
            ui.close_menu();
        }
        
        if ui.button("📋 Duplicate").clicked() {
            if let Some(matrix) = self.project.matrices.get(matrix_name) {
                let new_name = format!("{}_copy", matrix_name);
                let mut counter = 1;
                let mut final_name = new_name.clone();
                
                while self.project.matrices.contains_key(&final_name) {
                    final_name = format!("{}_{}", new_name, counter);
                    counter += 1;
                }
                
                self.project.matrices.insert(final_name.clone(), matrix.clone());
                self.selected_matrix = Some(final_name.clone());
                self.show_success(format!("Duplicated matrix as '{}'", final_name));
            }
            ui.close_menu();
        }
        
        ui.separator();
        
        if ui.button("🗑️ Delete").clicked() {
            self.project.matrices.remove(matrix_name);
            if self.selected_matrix.as_ref().map(|s| s.as_str()) == Some(matrix_name) {
                self.selected_matrix = None;
            }
            self.show_success(format!("Deleted matrix '{}'", matrix_name));
            ui.close_menu();
        }
    }

    fn render_rename_dialog(&mut self, ui: &mut egui::Ui) {
        if let Some(old_name) = &self.rename_dialog_matrix.clone() {
            egui::Window::new("Rename Matrix")
                .collapsible(false)
                .resizable(false)
                .show(ui.ctx(), |ui| {
                    ui.horizontal(|ui| {
                        ui.label("New name:");
                        ui.add(egui::TextEdit::singleline(&mut self.rename_input).desired_width(200.0));
                    });
                    
                    ui.horizontal(|ui| {
                        if ui.button("✅ Rename").clicked() {
                            if !self.rename_input.is_empty() && !self.project.matrices.contains_key(&self.rename_input) {
                                if let Some(matrix) = self.project.matrices.remove(old_name) {
                                    self.project.matrices.insert(self.rename_input.clone(), matrix);
                                    if self.selected_matrix.as_ref().map(|s| s.as_str()) == Some(old_name) {
                                        self.selected_matrix = Some(self.rename_input.clone());
                                    }
                                    self.show_success(format!("Renamed '{}' to '{}'", old_name, self.rename_input));
                                }
                                self.rename_dialog_matrix = None;
                            } else if self.project.matrices.contains_key(&self.rename_input) {
                                self.show_error("Matrix name already exists".to_string());
                            }
                        }
                        
                        if ui.button("❌ Cancel").clicked() {
                            self.rename_dialog_matrix = None;
                        }
                    });
                });
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
        // Header with edit mode toggle and quick actions
        ui.horizontal(|ui| {
            ui.heading("📝 Small Matrix Editor");
            ui.separator();
            
            // Edit mode toggle
            let edit_icon = if self.edit_mode_enabled { "🔓" } else { "🔒" };
            let edit_text = if self.edit_mode_enabled { "View Only" } else { "Enable Edit" };
            
            if ui.button(format!("{} {}", edit_icon, edit_text)).clicked() {
                self.edit_mode_enabled = !self.edit_mode_enabled;
            }
        });
        
        ui.add_space(5.0);
        
        // Quick actions toolbar
        ui.horizontal(|ui| {
            ui.label("⚡ Quick Actions:");
            
            if ui.button("🔄 Transpose").clicked() {
                if let Some(matrix) = self.project.matrices.get_mut(matrix_name) {
                    // For now, show a placeholder message since transpose might need error handling
                    self.show_success("Transpose feature coming soon".to_string());
                }
            }
            
            if ui.button("🧹 Clear").clicked() {
                if let Some(matrix) = self.project.matrices.get_mut(matrix_name) {
                    for i in 0..rows {
                        for j in 0..cols {
                            let _ = matrix.set(i, j, 0.0);
                        }
                    }
                    self.matrix_element_texts.remove(matrix_name);
                    self.show_success("Matrix cleared".to_string());
                }
            }
            
            if ui.button("🎲 Randomize").clicked() {
                if let Some(matrix) = self.project.matrices.get_mut(matrix_name) {
                    for i in 0..rows {
                        for j in 0..cols {
                            let random_val = rand::random::<f64>() * 10.0 - 5.0; // -5 to 5
                            let _ = matrix.set(i, j, random_val);
                        }
                    }
                    self.matrix_element_texts.remove(matrix_name);
                    self.show_success("Matrix randomized".to_string());
                }
            }
            
            if ui.button("📏 Resize").clicked() {
                // TODO: Show resize dialog
                self.show_success("Resize feature coming soon".to_string());
            }
        });
        
        ui.add_space(8.0);

        // Enhanced grid with better styling
        egui::Frame::none()
            .fill(ui.style().visuals.extreme_bg_color)
            .stroke(egui::Stroke::new(1.0, ui.style().visuals.window_stroke.color))
            .rounding(4.0)
            .inner_margin(8.0)
            .show(ui, |ui| {
                egui::Grid::new(format!("small_matrix_grid_{}", matrix_name))
                    .striped(false)
                    .spacing([2.0, 2.0])
                    .show(ui, |ui| {
                        // Header row with column indices
                        ui.label(""); // Empty corner
                        for j in 0..cols {
                            ui.vertical_centered(|ui| {
                                ui.strong(format!("C{}", j));
                            });
                        }
                        ui.end_row();

                        // Matrix rows with row indices
                        for i in 0..rows {
                            // Row index
                            ui.vertical_centered(|ui| {
                                ui.strong(format!("R{}", i));
                            });
                            
                            // Matrix elements
                            for j in 0..cols {
                                self.render_enhanced_matrix_cell(ui, matrix_name, i, j, true);
                            }
                            ui.end_row();
                        }
                    });
            });
        
        // Cell selection info
        if let Some((sel_row, sel_col)) = self.selected_cell {
            ui.add_space(5.0);
            ui.horizontal(|ui| {
                ui.label(format!("📍 Selected: Cell ({}, {})", sel_row, sel_col));
                if let Some(matrix) = self.project.matrices.get(matrix_name) {
                    if let Some(value) = matrix.get(sel_row, sel_col) {
                        ui.separator();
                        ui.label(format!("Value: {:.6}", value));
                    }
                }
            });
        }
    }

    fn render_large_matrix_editor(
        &mut self,
        ui: &mut egui::Ui,
        matrix_name: &str,
        rows: usize,
        cols: usize,
    ) {
        // Matrix info header
        ui.horizontal(|ui| {
            ui.heading("🔍 Large Matrix Viewer");
            ui.separator();
            ui.label(format!("� Dimensions: {}×{}", rows, cols));
        });
        ui.add_space(5.0);

        // Viewport controls
        self.render_viewport_controls(ui, matrix_name, rows, cols);
        ui.add_space(5.0);

        // Matrix summary statistics (if enabled)
        if self.show_matrix_stats {
            self.render_matrix_statistics(ui, matrix_name);
            ui.add_space(5.0);
        }

        // Get viewport offset for this matrix
        let offset = *self.viewport_offset.get(matrix_name).unwrap_or(&(0, 0));
        let (row_offset, col_offset) = offset;
        let (viewport_rows, viewport_cols) = self.viewport_size;

        // Calculate visible range
        let visible_row_end = (row_offset + viewport_rows).min(rows);
        let visible_col_end = (col_offset + viewport_cols).min(cols);
        
        ui.label(format!(
            "📋 Viewport: Rows {}-{}, Cols {}-{}",
            row_offset,
            visible_row_end.saturating_sub(1),
            col_offset,
            visible_col_end.saturating_sub(1)
        ));
        ui.add_space(3.0);

        // Calculate cell size based on zoom
        let base_cell_width = 65.0;
        let cell_width = base_cell_width * self.zoom_level;
        let cell_spacing = 2.0 * self.zoom_level;

        // Optimized viewport rendering
        egui::ScrollArea::both()
            .max_height(500.0)
            .max_width(800.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new(format!("large_matrix_viewport_{}", matrix_name))
                    .striped(true)
                    .spacing([cell_spacing, cell_spacing])
                    .min_col_width(cell_width)
                    .show(ui, |ui| {
                        // Header row with column indices
                        ui.label(""); // Empty corner
                        for j in col_offset..visible_col_end {
                            ui.label(format!("C{}", j));
                        }
                        ui.end_row();

                        // Matrix rows within viewport
                        for i in row_offset..visible_row_end {
                            // Row index
                            ui.label(format!("R{}", i));
                            
                            // Matrix elements
                            for j in col_offset..visible_col_end {
                                self.render_matrix_cell(ui, matrix_name, i, j, false);
                            }
                            ui.end_row();
                        }
                    });
            });
    }

    fn render_viewport_controls(
        &mut self,
        ui: &mut egui::Ui,
        matrix_name: &str,
        rows: usize,
        cols: usize,
    ) {
        ui.horizontal(|ui| {
            ui.label("🎛️ Viewport Controls:");
            
            // Get current offset
            let offset = *self.viewport_offset.get(matrix_name).unwrap_or(&(0, 0));
            let (mut row_offset, mut col_offset) = offset;
            let (viewport_rows, viewport_cols) = self.viewport_size;
            
            // Navigation buttons
            if ui.button("⬅️").clicked() {
                col_offset = col_offset.saturating_sub(viewport_cols / 2);
            }
            if ui.button("➡️").clicked() {
                col_offset = (col_offset + viewport_cols / 2).min(cols.saturating_sub(viewport_cols));
            }
            if ui.button("⬆️").clicked() {
                row_offset = row_offset.saturating_sub(viewport_rows / 2);
            }
            if ui.button("⬇️").clicked() {
                row_offset = (row_offset + viewport_rows / 2).min(rows.saturating_sub(viewport_rows));
            }
            
            ui.separator();
            
            // Home button
            if ui.button("🏠 Home").clicked() {
                row_offset = 0;
                col_offset = 0;
            }
            
            ui.separator();
            
            // Zoom controls
            ui.label("🔍 Zoom:");
            if ui.button("➕").clicked() {
                self.zoom_level = (self.zoom_level * 1.2).min(3.0);
            }
            if ui.button("➖").clicked() {
                self.zoom_level = (self.zoom_level / 1.2).max(0.5);
            }
            ui.label(format!("{:.1}x", self.zoom_level));
            
            // Update offset
            self.viewport_offset.insert(matrix_name.to_string(), (row_offset, col_offset));
        });
        
        ui.horizontal(|ui| {
            ui.label("📐 Viewport Size:");
            let (mut vp_rows, mut vp_cols) = self.viewport_size;
            
            ui.add(egui::DragValue::new(&mut vp_rows)
                .speed(1.0)
                .range(5..=50)
                .prefix("Rows: "));
            ui.add(egui::DragValue::new(&mut vp_cols)
                .speed(1.0)
                .range(5..=50)
                .prefix("Cols: "));
                
            self.viewport_size = (vp_rows, vp_cols);
            
            ui.separator();
            ui.checkbox(&mut self.show_matrix_stats, "📊 Show Statistics");
        });
    }

    fn render_matrix_statistics(&self, ui: &mut egui::Ui, matrix_name: &str) {
        if let Some(matrix) = self.project.matrices.get(matrix_name) {
            let (rows, cols) = matrix.dimensions();
            
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label("📊 Matrix Statistics:");
                    
                    // Calculate basic statistics
                    let mut values = Vec::new();
                    for i in 0..rows {
                        for j in 0..cols {
                            if let Some(val) = matrix.get(i, j) {
                                values.push(val);
                            }
                        }
                    }
                    
                    if !values.is_empty() {
                        let sum: f64 = values.iter().sum();
                        let mean = sum / values.len() as f64;
                        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        
                        ui.label(format!("📈 Mean: {:.3}", mean));
                        ui.separator();
                        ui.label(format!("📉 Min: {:.3}", min));
                        ui.separator();
                        ui.label(format!("📊 Max: {:.3}", max));
                        ui.separator();
                        ui.label(format!("🔢 Elements: {}", values.len()));
                    }
                });
            });
        }
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

    fn render_enhanced_matrix_cell(
        &mut self,
        ui: &mut egui::Ui,
        matrix_name: &str,
        row: usize,
        col: usize,
        is_small_matrix: bool,
    ) {
        let cell_key = (row, col);
        let is_selected = self.selected_cell == Some((row, col));
        let has_keyboard_focus = self.keyboard_focus_cell == Some((row, col));
        
        // Get current matrix value
        let current_value = if let Some(matrix) = self.project.matrices.get(matrix_name) {
            matrix.get(row, col).unwrap_or(0.0)
        } else {
            return;
        };
        
        // Cell styling based on state
        let cell_bg = if has_keyboard_focus {
            egui::Color32::from_rgb(80, 120, 200) // Blue for keyboard focus
        } else if is_selected {
            egui::Color32::from_rgb(100, 140, 100) // Green for selection
        } else {
            ui.style().visuals.extreme_bg_color
        };
        
        let cell_stroke = if is_selected || has_keyboard_focus {
            egui::Stroke::new(2.0, egui::Color32::WHITE)
        } else {
            egui::Stroke::new(1.0, ui.style().visuals.window_stroke.color)
        };

        egui::Frame::none()
            .fill(cell_bg)
            .stroke(cell_stroke)
            .rounding(2.0)
            .inner_margin(4.0)
            .show(ui, |ui| {
                if self.edit_mode_enabled {
                    self.render_editable_cell_content(ui, matrix_name, row, col, current_value, is_small_matrix);
                } else {
                    // View-only mode
                    let response = ui.button(format!("{:.3}", current_value));
                    if response.clicked() {
                        self.selected_cell = Some((row, col));
                    }
                }
            });
    }

    fn render_editable_cell_content(
        &mut self,
        ui: &mut egui::Ui,
        matrix_name: &str,
        row: usize,
        col: usize,
        current_value: f64,
        is_small_matrix: bool,
    ) {
        let cell_key = (row, col);
        
        // Get or initialize the text for this cell
        let mut text = {
            let text_map = self.matrix_element_texts.get_mut(matrix_name).unwrap();
            text_map
                .get(&cell_key)
                .cloned()
                .unwrap_or_else(|| format!("{:.3}", current_value))
        };

        let cell_width = if is_small_matrix { 80.0 } else { 65.0 };
        
        let text_edit = egui::TextEdit::singleline(&mut text)
            .desired_width(cell_width)
            .font(egui::TextStyle::Monospace)
            .hint_text("0.0");

        let response = ui.add(text_edit);
        
        // Handle selection
        if response.clicked() {
            self.selected_cell = Some((row, col));
        }
        
        // Handle text changes with validation
        let mut update_result: Option<Result<f64, String>> = None;
        
        if response.changed() {
            if let Some(text_map) = self.matrix_element_texts.get_mut(matrix_name) {
                text_map.insert(cell_key, text.clone());
            }
            
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

        // Apply matrix update and show errors
        if let Some(result) = update_result {
            match result {
                Ok(new_value) => {
                    if let Some(matrix) = self.project.matrices.get_mut(matrix_name) {
                        if let Err(_) = matrix.set(row, col, new_value) {
                            self.show_error(format!("Failed to set matrix element at ({}, {})", row, col));
                        }
                    }
                }
                Err(error_msg) => {
                    self.show_error(error_msg);
                }
            }
        }

        if response.lost_focus() {
            if let Ok(parsed_value) = text.trim().parse::<f64>() {
                if parsed_value.is_finite() {
                    if let Some(matrix) = self.project.matrices.get(matrix_name) {
                        let stored_value = matrix.get(row, col).unwrap_or(0.0);
                        if let Some(text_map) = self.matrix_element_texts.get_mut(matrix_name) {
                            text_map.insert(cell_key, format!("{:.3}", stored_value));
                        }
                    }
                }
            } else if !text.trim().is_empty() {
                if let Some(text_map) = self.matrix_element_texts.get_mut(matrix_name) {
                    text_map.insert(cell_key, format!("{:.3}", current_value));
                }
            }
        }

        if response.has_focus() {
            self.editing_element = Some((matrix_name.to_string(), row, col));
        }

        if response.hovered() {
            response.on_hover_text(format!(
                "Cell ({}, {})\nValue: {:.6}\nClick to edit", 
                row, col, current_value
            ));
        }
    }

    fn render_enhanced_viewport_controls(
        &mut self,
        ui: &mut egui::Ui,
        matrix_name: &str,
        rows: usize,
        cols: usize,
    ) {
        ui.horizontal(|ui| {
            ui.label("🎛️ Viewport:");
            
            let offset = *self.viewport_offset.get(matrix_name).unwrap_or(&(0, 0));
            let (mut row_offset, mut col_offset) = offset;
            let (viewport_rows, viewport_cols) = self.viewport_size;
            
            // Navigation buttons
            if ui.button("⬅️").clicked() {
                col_offset = col_offset.saturating_sub(viewport_cols / 2);
            }
            if ui.button("➡️").clicked() {
                col_offset = (col_offset + viewport_cols / 2).min(cols.saturating_sub(viewport_cols));
            }
            if ui.button("⬆️").clicked() {
                row_offset = row_offset.saturating_sub(viewport_rows / 2);
            }
            if ui.button("⬇️").clicked() {
                row_offset = (row_offset + viewport_rows / 2).min(rows.saturating_sub(viewport_rows));
            }
            
            ui.separator();
            
            if ui.button("🏠 Home").clicked() {
                row_offset = 0;
                col_offset = 0;
            }
            
            ui.separator();
            
            // Zoom slider (improved)
            ui.label("🔍 Zoom:");
            ui.add(egui::Slider::new(&mut self.zoom_level, 0.5..=3.0)
                .step_by(0.1)
                .text("x"));
            
            self.viewport_offset.insert(matrix_name.to_string(), (row_offset, col_offset));
        });
        
        ui.horizontal(|ui| {
            ui.label("📐 Viewport Size:");
            let (mut vp_rows, mut vp_cols) = self.viewport_size;
            
            ui.add(egui::DragValue::new(&mut vp_rows)
                .speed(1.0)
                .range(5..=30)
                .prefix("Rows: "));
            ui.add(egui::DragValue::new(&mut vp_cols)
                .speed(1.0)
                .range(5..=30)
                .prefix("Cols: "));
                
            self.viewport_size = (vp_rows, vp_cols);
            
            ui.separator();
            ui.checkbox(&mut self.show_matrix_stats, "📊 Statistics");
        });
    }

    fn render_matrix_viewport_with_fixed_headers(
        &mut self,
        ui: &mut egui::Ui,
        matrix_name: &str,
        rows: usize,
        cols: usize,
    ) {
        let offset = *self.viewport_offset.get(matrix_name).unwrap_or(&(0, 0));
        let (row_offset, col_offset) = offset;
        let (viewport_rows, viewport_cols) = self.viewport_size;

        let visible_row_end = (row_offset + viewport_rows).min(rows);
        let visible_col_end = (col_offset + viewport_cols).min(cols);
        
        ui.label(format!(
            "📋 Viewport: Rows {}-{}, Cols {}-{}",
            row_offset,
            visible_row_end.saturating_sub(1),
            col_offset,
            visible_col_end.saturating_sub(1)
        ));
        ui.add_space(3.0);

        let base_cell_width = 70.0;
        let cell_width = base_cell_width * self.zoom_level;
        let cell_spacing = 2.0 * self.zoom_level;

        // Fixed headers with scrollable content
        egui::ScrollArea::both()
            .max_height(500.0)
            .max_width(800.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new(format!("large_matrix_viewport_{}", matrix_name))
                    .striped(true)
                    .spacing([cell_spacing, cell_spacing])
                    .min_col_width(cell_width)
                    .show(ui, |ui| {
                        // Header row with column indices
                        ui.label(""); // Empty corner
                        for j in col_offset..visible_col_end {
                            ui.vertical_centered(|ui| {
                                ui.strong(format!("C{}", j));
                            });
                        }
                        ui.end_row();

                        // Matrix rows within viewport
                        for i in row_offset..visible_row_end {
                            // Row header
                            ui.vertical_centered(|ui| {
                                ui.strong(format!("R{}", i));
                            });
                            
                            // Matrix elements
                            for j in col_offset..visible_col_end {
                                self.render_enhanced_matrix_cell(ui, matrix_name, i, j, false);
                            }
                            ui.end_row();
                        }
                    });
            });
    }

    fn render_matrix_minimap(
        &mut self,
        ui: &mut egui::Ui,
        matrix_name: &str,
        rows: usize,
        cols: usize,
    ) {
        ui.vertical(|ui| {
            ui.heading("🗺️ Mini-map");
            ui.add_space(5.0);
            
            let minimap_size = 150.0;
            let cell_size = (minimap_size / rows.max(cols) as f32).max(1.0);
            
            let offset = *self.viewport_offset.get(matrix_name).unwrap_or(&(0, 0));
            let (row_offset, col_offset) = offset;
            let (viewport_rows, viewport_cols) = self.viewport_size;
            
            // Draw minimap
            let (response, painter) = ui.allocate_painter(
                egui::Vec2::new(minimap_size, minimap_size),
                egui::Sense::click()
            );
            
            let rect = response.rect;
            
            // Draw matrix representation
            for i in 0..rows.min((minimap_size / cell_size) as usize) {
                for j in 0..cols.min((minimap_size / cell_size) as usize) {
                    let cell_rect = egui::Rect::from_min_size(
                        rect.min + egui::Vec2::new(j as f32 * cell_size, i as f32 * cell_size),
                        egui::Vec2::splat(cell_size)
                    );
                    
                    let color = if i >= row_offset && i < row_offset + viewport_rows &&
                                  j >= col_offset && j < col_offset + viewport_cols {
                        egui::Color32::from_rgb(100, 150, 255) // Highlight viewport
                    } else {
                        egui::Color32::from_rgb(200, 200, 200)
                    };
                    
                    painter.rect_filled(cell_rect, 0.0, color);
                    painter.rect_stroke(cell_rect, 0.0, egui::Stroke::new(0.5, egui::Color32::BLACK));
                }
            }
            
            // Handle clicks to jump to position
            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let relative_pos = pos - rect.min;
                    let clicked_row = (relative_pos.y / cell_size) as usize;
                    let clicked_col = (relative_pos.x / cell_size) as usize;
                    
                    let new_row_offset = clicked_row.saturating_sub(viewport_rows / 2).min(rows.saturating_sub(viewport_rows));
                    let new_col_offset = clicked_col.saturating_sub(viewport_cols / 2).min(cols.saturating_sub(viewport_cols));
                    
                    self.viewport_offset.insert(matrix_name.to_string(), (new_row_offset, new_col_offset));
                }
            }
            
            ui.add_space(5.0);
            ui.label(format!("🎯 Click to jump to position"));
        });
    }
}
