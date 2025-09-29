use eframe::egui;
use crate::matrix::{Matrix, decomposition::*};
use crate::data::project::Project;

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
                
                // Right panel: Operations and results
                ui.vertical(|ui| {
                    if let Some(selected) = &self.selected_matrix {
                        if let Some(_matrix) = self.project.matrices.get(selected) {
                            ui.label(format!("Operations for: {}", selected));
                            
                            if let Some(_result) = self.operations_panel.show(ui, &self.project, selected) {
                                // Handle result
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
                
                let response = ui.selectable_label(is_selected, format!(
                    "{} ({}×{})", 
                    name, 
                    matrix.dimensions().0, 
                    matrix.dimensions().1
                ));
                
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
        
        // Simple matrix creation interface
        ui.horizontal(|ui| {
            if ui.button("Create 3x3 Identity").clicked() {
                let matrix = Matrix::identity(3, "temp".to_string());
                let name = format!("identity_3x3_{}", self.project.matrices.len());
                self.project.matrices.insert(name.clone(), matrix);
                self.selected_matrix = Some(name);
            }
            
            if ui.button("Create 4x4 Identity").clicked() {
                let matrix = Matrix::identity(4, "temp".to_string());
                let name = format!("identity_4x4_{}", self.project.matrices.len());
                self.project.matrices.insert(name.clone(), matrix);
                self.selected_matrix = Some(name);
            }
        });
        
        ui.horizontal(|ui| {
            if ui.button("Create 3x3 Random").clicked() {
                let matrix = Matrix::random(3, 3, "temp".to_string());
                let name = format!("random_3x3_{}", self.project.matrices.len());
                self.project.matrices.insert(name.clone(), matrix);
                self.selected_matrix = Some(name);
            }
            
            if ui.button("Create 4x4 Random").clicked() {
                let matrix = Matrix::random(4, 4, "temp".to_string());
                let name = format!("random_4x4_{}", self.project.matrices.len());
                self.project.matrices.insert(name.clone(), matrix);
                self.selected_matrix = Some(name);
            }
        });
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
}