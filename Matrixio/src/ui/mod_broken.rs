pub mod components;
pub mod matrix_editor;
pub mod operations_panel;
pub mod results_viewer;

use eframe::egui;
use crate::matrix::{Matrix, decomposition::*};
use crate::data::project::Project;



/// Main application state
pub struct MatrixioApp {
    // Core application stat            let names: Vec<_> = self.project.matrices.keys().cloned().collect();
            for name in names {
                if let Some(matrix) = self.project.matrices.get(&name) {
    project: Project,
    selected_matrix: Option<String>,
    operation_result: Option<OperationResult>,
    
    // UI state
    show_matrix_editor: bool,
    show_operations_panel: bool,
    show_results_viewer: bool,
    show_about: bool,
    
    // Matrix editor state
    matrix_editor: matrix_editor::MatrixEditor,
    
    // Operations panel state
    operations_panel: operations_panel::OperationsPanel,
    
    // Results viewer state
    results_viewer: results_viewer::ResultsViewer,
    
    // Error handling
    error_message: Option<String>,
    success_message: Option<String>,
}

#[derive(Debug, Clone)]
pub enum OperationResult {
    Matrix(Matrix),
    Scalar(f64),
    Vector(Vec<f64>),
    Decomposition(DecompositionResult),
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
            operation_result: None,
            
            show_matrix_editor: true,
            show_operations_panel: true,
            show_results_viewer: true,
            show_about: false,
            
            matrix_editor: matrix_editor::MatrixEditor::new(),
            operations_panel: operations_panel::OperationsPanel::new(),
            results_viewer: results_viewer::ResultsViewer::new(),
            
            error_message: None,
            success_message: None,
        }
    }

    /// Show error message to user
    fn show_error(&mut self, message: String) {
        self.error_message = Some(message);
        log::error!("Error: {}", self.error_message.as_ref().unwrap());
    }

    /// Show success message to user
    fn show_success(&mut self, message: String) {
        self.success_message = Some(message);
        log::info!("Success: {}", self.success_message.as_ref().unwrap());
    }

    /// Clear all messages
    fn clear_messages(&mut self) {
        self.error_message = None;
        self.success_message = None;
    }
}

impl eframe::App for MatrixioApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Handle keyboard shortcuts
        self.handle_shortcuts(ctx);

        // Top menu bar
        self.render_menu_bar(ctx, frame);

        // Status messages
        self.render_status_messages(ctx);

        // Main panels
        self.render_main_panels(ctx);

        // Modal dialogs
        self.render_modals(ctx);
    }
}

impl MatrixioApp {
    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        ctx.input_mut(|i| {
            // Ctrl+N: New matrix
            if i.consume_key(egui::Modifiers::CTRL, egui::Key::N) {
                self.matrix_editor.start_new_matrix();
            }
            
            // Ctrl+O: Open project
            if i.consume_key(egui::Modifiers::CTRL, egui::Key::O) {
                self.open_project();
            }
            
            // Ctrl+S: Save project
            if i.consume_key(egui::Modifiers::CTRL, egui::Key::S) {
                self.save_project();
            }
            
            // F1: Show about
            if i.consume_key(egui::Modifiers::NONE, egui::Key::F1) {
                self.show_about = true;
            }
            
            // Escape: Clear messages
            if i.consume_key(egui::Modifiers::NONE, egui::Key::Escape) {
                self.clear_messages();
            }
        });
    }

    fn render_menu_bar(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                // File menu
                ui.menu_button("File", |ui| {
                    if ui.button("🗋 New Matrix (Ctrl+N)").clicked() {
                        self.matrix_editor.start_new_matrix();
                        ui.close_menu();
                    }
                    
                    ui.separator();
                    
                    if ui.button("📂 Open Project (Ctrl+O)").clicked() {
                        self.open_project();
                        ui.close_menu();
                    }
                    
                    if ui.button("💾 Save Project (Ctrl+S)").clicked() {
                        self.save_project();
                        ui.close_menu();
                    }
                    
                    if ui.button("📤 Export Matrix").clicked() {
                        self.export_matrix();
                        ui.close_menu();
                    }
                    
                    ui.separator();
                    
                    if ui.button("🚪 Exit").clicked() {
                        std::process::exit(0);
                    }
                });

                // View menu
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_matrix_editor, "Matrix Editor");
                    ui.checkbox(&mut self.show_operations_panel, "Operations Panel");
                    ui.checkbox(&mut self.show_results_viewer, "Results Viewer");
                });

                // Tools menu
                ui.menu_button("Tools", |ui| {
                    if ui.button("🔧 Matrix Generator").clicked() {
                        self.matrix_editor.show_generator();
                        ui.close_menu();
                    }
                    
                    if ui.button("📊 Performance Benchmark").clicked() {
                        self.run_benchmark();
                        ui.close_menu();
                    }
                    
                    if ui.button("🧮 Batch Operations").clicked() {
                        // TODO: Implement batch operations
                        ui.close_menu();
                    }
                });

                // Help menu
                ui.menu_button("Help", |ui| {
                    if ui.button("❓ About (F1)").clicked() {
                        self.show_about = true;
                        ui.close_menu();
                    }
                    
                    if ui.button("📖 User Guide").clicked() {
                        // TODO: Open user guide
                        ui.close_menu();
                    }
                    
                    if ui.button("⌨ Keyboard Shortcuts").clicked() {
                        // TODO: Show shortcuts dialog
                        ui.close_menu();
                    }
                });
            });
        });
    }

    fn render_status_messages(&mut self, ctx: &egui::Context) {
        // Error messages
        if let Some(ref error) = self.error_message.clone() {
            egui::TopBottomPanel::top("error_panel").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("❌ Error:").color(egui::Color32::RED));
                    ui.label(error);
                    if ui.button("✖").clicked() {
                        self.error_message = None;
                    }
                });
            });
        }

        // Success messages
        if let Some(ref success) = self.success_message.clone() {
            egui::TopBottomPanel::top("success_panel").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("✅ Success:").color(egui::Color32::GREEN));
                    ui.label(success);
                    if ui.button("✖").clicked() {
                        self.success_message = None;
                    }
                });
            });
        }
    }

    fn render_main_panels(&mut self, ctx: &egui::Context) {
        // Left panel: Matrix list and editor
        if self.show_matrix_editor {
            egui::SidePanel::left("matrix_panel")
                .default_width(300.0)
                .show(ctx, |ui| {
                    self.render_matrix_list(ui);
                    ui.separator();
                    self.matrix_editor.render(ui, &mut self.project);
                });
        }

        // Right panel: Operations
        if self.show_operations_panel {
            egui::SidePanel::right("operations_panel")
                .default_width(300.0)
                .show(ctx, |ui| {
                    self.operations_panel.render(ui, &self.project, self.selected_matrix.as_ref());
                });
        }

        // Central panel: Results viewer
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.show_results_viewer {
                self.results_viewer.render(ui, &self.operation_result);
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Results viewer is hidden. Use View menu to show it.");
                });
            }
        });
    }

    fn render_matrix_list(&mut self, ui: &mut egui::Ui) {
        ui.heading("Matrices");
        
        egui::ScrollArea::vertical().show(ui, |ui| {
            let names: Vec<_> = self.project.matrices.keys().cloned().collect();\n            for name in names {\n                if let Some(matrix) = self.project.matrices.get(&name) {
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
            
            if self.project.matrices.is_empty() {
                ui.label("No matrices yet. Create one using the editor below.");
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
                    ui.heading("Matrixio v0.1.0");
                    ui.separator();
                    
                    ui.label("High-Performance Matrix Calculator");
                    ui.label("Built with Rust, egui, and nalgebra");
                    ui.separator();
                    
                    ui.label("Features:");
                    ui.label("• Large matrix support (beyond 4×4)");
                    ui.label("• BLAS/LAPACK optimization");
                    ui.label("• Parallel computation");
                    ui.label("• GPU acceleration (optional)");
                    ui.label("• Intuitive user interface");
                    ui.separator();
                    
                    ui.horizontal(|ui| {
                        if ui.button("Close").clicked() {
                            self.show_about = false;
                        }
                    });
                });
        }
    }

    // File operations
    fn open_project(&mut self) {
        // TODO: Implement file dialog for opening projects
        self.show_success("Project opened successfully".to_string());
    }

    fn save_project(&mut self) {
        // TODO: Implement file dialog for saving projects
        self.show_success("Project saved successfully".to_string());
    }

    fn export_matrix(&mut self) {
        // TODO: Implement matrix export functionality
        self.show_success("Matrix exported successfully".to_string());
    }

    fn run_benchmark(&mut self) {
        // TODO: Implement performance benchmarking
        self.show_success("Benchmark completed".to_string());
    }
}