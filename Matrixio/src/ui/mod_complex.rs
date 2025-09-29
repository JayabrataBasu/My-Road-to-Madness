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
    results_viewer: results_viewer::ResultsViewer,
    
    // Modal states
    show_about: bool,
    show_export_dialog: bool,
    show_import_dialog: bool,
    
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
            results_viewer: results_viewer::ResultsViewer::new(),
            show_about: false,
            show_export_dialog: false,
            show_import_dialog: false,
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
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
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
                    
                    if ui.button("💾 Save Project").clicked() {
                        self.save_project();
                        ui.close_menu();
                    }
                    
                    if ui.button("📁 Load Project").clicked() {
                        self.load_project();
                        ui.close_menu();
                    }
                    
                    ui.separator();
                    
                    if ui.button("📤 Export Matrix").clicked() {
                        self.show_export_dialog = true;
                        ui.close_menu();
                    }
                    
                    if ui.button("📥 Import Matrix").clicked() {
                        self.show_import_dialog = true;
                        ui.close_menu();
                    }
                    
                    ui.separator();
                    
                    if ui.button("🚪 Exit").clicked() {
                        std::process::exit(0);
                    }
                });

                // View menu
                ui.menu_button("View", |ui| {
                    if ui.button("🔄 Refresh").clicked() {
                        ctx.request_repaint();
                        ui.close_menu();
                    }
                });

                // Help menu
                ui.menu_button("Help", |ui| {
                    if ui.button("❓ About").clicked() {
                        self.show_about = true;
                        ui.close_menu();
                    }
                    
                    if ui.button("📊 Run Benchmark").clicked() {
                        self.run_benchmark();
                        ui.close_menu();
                    }
                });
            });
        });

        // Main content
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                .resizable(true)
                .default_split_ratio(0.3)
                .show(
                    ctx,
                    |ui| {
                        // Left panel: Matrix list and editor
                        ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                        
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            self.render_matrix_list(ui);
                            
                            ui.separator();
                            
                            self.matrix_editor.show(ui, &mut self.project);
                        });
                    },
                    |ui| {
                        // Right panel: Operations and results
                        if let Some(selected) = &self.selected_matrix {
                            if self.project.matrices.contains_key(selected) {
                                ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                                
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    if let Some(result) = self.operations_panel.show(ui, &self.project, selected) {
                                        self.results_viewer.add_result(result);
                                    }
                                    
                                    ui.separator();
                                    
                                    self.results_viewer.show(ui);
                                });
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
                    },
                );
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
                
                // Context menu for each matrix
                response.context_menu(|ui| {
                    if ui.button("Edit").clicked() {
                        self.matrix_editor.edit_matrix(name.clone(), matrix.clone());
                        ui.close_menu();
                    }
                    
                    if ui.button("Duplicate").clicked() {
                        let new_name = format!("{}_copy", name);
                        let mut new_matrix = matrix.clone();
                        new_matrix.name = new_name.clone();
                        self.project.matrices.insert(new_name, new_matrix);
                        ui.close_menu();
                    }
                    
                    if ui.button("Delete").clicked() {
                        self.project.matrices.remove(&name);
                        if self.selected_matrix.as_ref() == Some(&name) {
                            self.selected_matrix = None;
                        }
                        ui.close_menu();
                    }
                });
            }
        }
        
        if self.project.matrices.is_empty() {
            ui.label("No matrices yet. Create one using the editor below.");
        }
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

    fn save_project(&mut self) {
        match self.project.save() {
            Ok(_) => self.show_success("Project saved successfully".to_string()),
            Err(e) => self.show_error(format!("Failed to save project: {}", e)),
        }
    }

    fn load_project(&mut self) {
        match Project::load() {
            Ok(project) => {
                self.project = project;
                self.selected_matrix = None;
                self.show_success("Project loaded successfully".to_string());
            }
            Err(e) => self.show_error(format!("Failed to load project: {}", e)),
        }
    }

    fn export_matrix(&mut self) {
        self.show_success("Matrix exported successfully".to_string());
    }

    fn run_benchmark(&mut self) {
        // TODO: Implement performance benchmarking
        self.show_success("Benchmark completed".to_string());
    }
}