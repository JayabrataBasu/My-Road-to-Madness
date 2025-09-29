use eframe::egui;
use crate::ui::{OperationResult, DecompositionResult};
use crate::matrix::Matrix;

/// Results viewer for displaying operation results
pub struct ResultsViewer {
    // Display options
    show_full_matrix: bool,
    precision: usize,
    scientific_notation: bool,
    
    // Navigation for large matrices
    view_start_row: usize,
    view_start_col: usize,
    view_rows: usize,
    view_cols: usize,
}

impl ResultsViewer {
    pub fn new() -> Self {
        Self {
            show_full_matrix: true,
            precision: 6,
            scientific_notation: false,
            view_start_row: 0,
            view_start_col: 0,
            view_rows: 20,
            view_cols: 10,
        }
    }

    pub fn render(&mut self, ui: &mut egui::Ui, result: &Option<OperationResult>) {
        ui.heading("Results");

        if let Some(result) = result {
            // Display options
            self.render_display_options(ui);
            ui.separator();

            // Render the actual result
            match result {
                OperationResult::Matrix(matrix) => self.render_matrix_result(ui, matrix),
                OperationResult::Scalar(value) => self.render_scalar_result(ui, *value),
                OperationResult::Vector(vector) => self.render_vector_result(ui, vector),
                OperationResult::Decomposition(decomp) => self.render_decomposition_result(ui, decomp),
                OperationResult::Text(text) => self.render_text_result(ui, text),
            }
        } else {
            ui.centered_and_justified(|ui| {
                ui.label("No results to display. Perform an operation to see results here.");
            });
        }
    }

    fn render_display_options(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Display:");
            ui.checkbox(&mut self.show_full_matrix, "Full matrix");
            
            ui.separator();
            
            ui.label("Precision:");
            ui.add(egui::DragValue::new(&mut self.precision).range(0..=15));
            
            ui.separator();
            
            ui.checkbox(&mut self.scientific_notation, "Scientific notation");
        });
    }

    fn render_matrix_result(&mut self, ui: &mut egui::Ui, matrix: &Matrix) {
        let (rows, cols) = matrix.dimensions();
        
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(format!("Matrix: {}", matrix.name));
                ui.separator();
                ui.label(format!("Dimensions: {}×{}", rows, cols));
                ui.separator();
                ui.label(format!("Elements: {}", rows * cols));
            });
        });

        ui.separator();

        // For large matrices, show navigation controls
        if rows > 20 || cols > 10 {
            self.render_navigation_controls(ui, rows, cols);
            ui.separator();
        }

        // Matrix visualization
        if self.should_show_heatmap(rows, cols) {
            self.render_matrix_heatmap(ui, matrix);
        }

        // Matrix data display
        if rows <= 50 && cols <= 20 || self.show_full_matrix {
            self.render_matrix_data(ui, matrix);
        } else {
            self.render_matrix_preview(ui, matrix);
        }

        // Matrix statistics
        ui.separator();
        self.render_matrix_statistics(ui, matrix);
    }

    fn render_scalar_result(&self, ui: &mut egui::Ui, value: f64) {
        ui.group(|ui| {
            ui.label("Scalar Result:");
            ui.separator();
            
            let formatted_value = if self.scientific_notation {
                format!("{:.precision$e}", value, precision = self.precision)
            } else {
                format!("{:.precision$}", value, precision = self.precision)
            };
            
            ui.label(egui::RichText::new(formatted_value)
                .font(egui::FontId::monospace(20.0))
                .color(egui::Color32::LIGHT_BLUE));
        });
    }

    fn render_vector_result(&self, ui: &mut egui::Ui, vector: &[f64]) {
        ui.group(|ui| {
            ui.label(format!("Vector Result (length: {})", vector.len()));
        });

        ui.separator();

        if vector.len() <= 50 {
            // Show full vector
            egui::Grid::new("vector_grid")
                .striped(true)
                .show(ui, |ui| {
                    for (i, &value) in vector.iter().enumerate() {
                        ui.label(format!("[{}]", i));
                        
                        let formatted_value = if self.scientific_notation {
                            format!("{:.precision$e}", value, precision = self.precision)
                        } else {
                            format!("{:.precision$}", value, precision = self.precision)
                        };
                        
                        ui.label(egui::RichText::new(formatted_value).font(egui::FontId::monospace(12.0)));
                        ui.end_row();
                    }
                });
        } else {
            // Show preview for large vectors
            ui.label("Vector preview (first 10 elements):");
            egui::Grid::new("vector_preview_grid")
                .striped(true)
                .show(ui, |ui| {
                    for (i, &value) in vector.iter().take(10).enumerate() {
                        ui.label(format!("[{}]", i));
                        
                        let formatted_value = if self.scientific_notation {
                            format!("{:.precision$e}", value, precision = self.precision)
                        } else {
                            format!("{:.precision$}", value, precision = self.precision)
                        };
                        
                        ui.label(egui::RichText::new(formatted_value).font(egui::FontId::monospace(12.0)));
                        ui.end_row();
                    }
                    
                    ui.label("...");
                    ui.label(format!("({} more elements)", vector.len() - 10));
                    ui.end_row();
                });
        }
    }

    fn render_decomposition_result(&mut self, ui: &mut egui::Ui, decomp: &DecompositionResult) {
        match decomp {
            DecompositionResult::LU(lu) => {
                ui.label("LU Decomposition Result:");
                ui.separator();
                
                ui.collapsing("L Matrix (Lower Triangular)", |ui| {
                    self.render_matrix_data(ui, &lu.l);
                });
                
                ui.collapsing("U Matrix (Upper Triangular)", |ui| {
                    self.render_matrix_data(ui, &lu.u);
                });
                
                ui.collapsing("Permutation Vector", |ui| {
                    ui.label(format!("P = {:?}", lu.p));
                });
            }
            
            DecompositionResult::QR(qr) => {
                ui.label("QR Decomposition Result:");
                ui.separator();
                
                ui.collapsing("Q Matrix (Orthogonal)", |ui| {
                    self.render_matrix_data(ui, &qr.q);
                });
                
                ui.collapsing("R Matrix (Upper Triangular)", |ui| {
                    self.render_matrix_data(ui, &qr.r);
                });
            }
            
            DecompositionResult::SVD(svd) => {
                ui.label("SVD Decomposition Result:");
                ui.separator();
                
                if let Some(ref u) = svd.u {
                    ui.collapsing("U Matrix", |ui| {
                        self.render_matrix_data(ui, u);
                    });
                }
                
                ui.collapsing("Singular Values", |ui| {
                    self.render_matrix_data(ui, &svd.s);
                });
                
                if let Some(ref v_t) = svd.v_t {
                    ui.collapsing("V^T Matrix", |ui| {
                        self.render_matrix_data(ui, v_t);
                    });
                }
            }
            
            DecompositionResult::Eigen(eigen) => {
                ui.label("Eigenvalue Decomposition Result:");
                ui.separator();
                
                ui.collapsing("Eigenvalues", |ui| {
                    self.render_vector_result(ui, &eigen.values);
                });
                
                if let Some(ref eigenvectors) = eigen.vectors {
                    ui.collapsing("Eigenvectors", |ui| {
                        self.render_matrix_data(ui, eigenvectors);
                    });
                }
            }
            
            DecompositionResult::Cholesky(chol) => {
                ui.label("Cholesky Decomposition Result:");
                ui.separator();
                
                ui.collapsing("L Matrix (Lower Triangular)", |ui| {
                    self.render_matrix_data(ui, &chol.l);
                });
            }
        }
    }

    fn render_text_result(&self, ui: &mut egui::Ui, text: &str) {
        ui.group(|ui| {
            ui.label("Text Result:");
            ui.separator();
            ui.label(text);
        });
    }

    fn render_navigation_controls(&mut self, ui: &mut egui::Ui, total_rows: usize, total_cols: usize) {
        ui.horizontal(|ui| {
            ui.label("View navigation:");
            
            ui.separator();
            ui.label("Start row:");
            ui.add(egui::DragValue::new(&mut self.view_start_row).range(0..=total_rows.saturating_sub(1)));
            
            ui.label("Start col:");
            ui.add(egui::DragValue::new(&mut self.view_start_col).range(0..=total_cols.saturating_sub(1)));
            
            ui.separator();
            ui.label("View size:");
            ui.add(egui::DragValue::new(&mut self.view_rows).range(1..=50));
            ui.label("×");
            ui.add(egui::DragValue::new(&mut self.view_cols).range(1..=20));
        });
    }

    fn should_show_heatmap(&self, rows: usize, cols: usize) -> bool {
        rows >= 5 && cols >= 5 && rows <= 100 && cols <= 100
    }

    fn render_matrix_heatmap(&self, ui: &mut egui::Ui, matrix: &Matrix) {
        // TODO: Implement matrix heatmap visualization
        // This would show a color-coded representation of matrix values
        ui.label("🔥 Heatmap visualization (TODO: Implement)");
    }

    fn render_matrix_data(&self, ui: &mut egui::Ui, matrix: &Matrix) {
        let (rows, cols) = matrix.dimensions();
        
        if rows == 0 || cols == 0 {
            ui.label("Empty matrix");
            return;
        }

        let max_display_rows = if self.show_full_matrix { usize::MAX } else { 20 };
        let max_display_cols = if self.show_full_matrix { usize::MAX } else { 10 };
        
        let display_rows = std::cmp::min(rows, max_display_rows);
        let display_cols = std::cmp::min(cols, max_display_cols);

        egui::ScrollArea::both()
            .max_height(400.0)
            .show(ui, |ui| {
                egui::Grid::new("matrix_data_grid")
                    .striped(true)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        // Header row (column indices)
                        ui.label(""); // Empty corner
                        for j in 0..display_cols {
                            ui.label(egui::RichText::new(format!("[{}]", j))
                                .color(egui::Color32::GRAY)
                                .font(egui::FontId::monospace(10.0)));
                        }
                        if display_cols < cols {
                            ui.label("...");
                        }
                        ui.end_row();

                        // Data rows
                        for i in 0..display_rows {
                            // Row index
                            ui.label(egui::RichText::new(format!("[{}]", i))
                                .color(egui::Color32::GRAY)
                                .font(egui::FontId::monospace(10.0)));
                            
                            // Row data
                            for j in 0..display_cols {
                                if let Some(value) = matrix.get(i, j) {
                                    let formatted_value = if self.scientific_notation {
                                        format!("{:.precision$e}", value, precision = self.precision)
                                    } else {
                                        format!("{:.precision$}", value, precision = self.precision)
                                    };
                                    
                                    ui.label(egui::RichText::new(formatted_value)
                                        .font(egui::FontId::monospace(10.0)));
                                } else {
                                    ui.label("N/A");
                                }
                            }
                            
                            if display_cols < cols {
                                ui.label("...");
                            }
                            ui.end_row();
                        }

                        // Show "..." row if truncated
                        if display_rows < rows {
                            ui.label("⋮");
                            for _ in 0..display_cols {
                                ui.label("⋮");
                            }
                            if display_cols < cols {
                                ui.label("⋱");
                            }
                            ui.end_row();
                        }
                    });
            });
    }

    fn render_matrix_preview(&self, ui: &mut egui::Ui, matrix: &Matrix) {
        ui.label("Matrix preview (showing top-left corner):");
        
        // Create a temporary "preview" matrix for display
        let preview_size = 5;
        let (rows, cols) = matrix.dimensions();
        let preview_rows = std::cmp::min(rows, preview_size);
        let preview_cols = std::cmp::min(cols, preview_size);

        egui::Grid::new("matrix_preview_grid")
            .striped(true)
            .show(ui, |ui| {
                for i in 0..preview_rows {
                    for j in 0..preview_cols {
                        if let Some(value) = matrix.get(i, j) {
                            let formatted_value = format!("{:.3}", value);
                            ui.label(egui::RichText::new(formatted_value)
                                .font(egui::FontId::monospace(10.0)));
                        }
                    }
                    
                    if cols > preview_size {
                        ui.label("...");
                    }
                    ui.end_row();
                }
                
                if rows > preview_size {
                    for _ in 0..preview_cols {
                        ui.label("⋮");
                    }
                    if cols > preview_size {
                        ui.label("⋱");
                    }
                    ui.end_row();
                }
            });
    }

    fn render_matrix_statistics(&self, ui: &mut egui::Ui, matrix: &Matrix) {
        ui.collapsing("Matrix Statistics", |ui| {
            let (rows, cols) = matrix.dimensions();
            
            ui.horizontal(|ui| {
                ui.label(format!("Dimensions: {}×{}", rows, cols));
                ui.separator();
                ui.label(format!("Elements: {}", rows * cols));
                ui.separator();
                ui.label(format!("Type: {}", if matrix.is_square() { "Square" } else { "Rectangular" }));
            });

            if matrix.is_square() {
                ui.horizontal(|ui| {
                    ui.label(format!("Symmetric: {}", if matrix.is_symmetric() { "Yes" } else { "No" }));
                    ui.separator();
                    
                    if let Ok(trace) = matrix.trace() {
                        ui.label(format!("Trace: {:.6}", trace));
                    }
                    ui.separator();
                    
                    if let Ok(det) = matrix.determinant() {
                        ui.label(format!("Determinant: {:.6}", det));
                    }
                });
            }

            ui.horizontal(|ui| {
                ui.label(format!("Frobenius Norm: {:.6}", matrix.frobenius_norm()));
                ui.separator();
                ui.label(format!("Rank: {}", matrix.rank()));
                ui.separator();
                ui.label(format!("Condition Number: {:.6}", matrix.condition_number()));
            });
        });
    }
}