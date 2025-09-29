use eframe::egui;
use crate::matrix::Matrix;
use crate::data::project::Project;
use crate::ui::{OperationResult, DecompositionResult};

/// Operations panel for performing matrix calculations
pub struct OperationsPanel {
    // Basic operations
    selecte        egui::ComboBox::from_id_salt(\"matrix_selector\")_operation: BasicOperation,
    
    // Binary operations
    second_matrix: Option<String>,
    
    // Scalar operations
    scalar_value: f64,
    
    // Advanced operations
    selected_decomposition: DecompositionType,
    compute_u: bool,
    compute_v: bool,
    
    // System solving
    rhs_matrix: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
enum BasicOperation {
    // Unary operations
    Transpose,
    Inverse,
    Determinant,
    Trace,
    Rank,
    ConditionNumber,
    FrobeniusNorm,
    
    // Binary operations
    Add,
    Subtract,
    Multiply,
    HadamardProduct,
    
    // Scalar operations
    ScalarMultiply,
    ScalarDivide,
    MatrixPower,
    
    // Element-wise functions
    ElementExp,
    ElementLn,
    ElementSin,
    ElementCos,
    ElementTan,
    ElementSqrt,
    ElementAbs,
    
    // System solving
    SolveLinearSystem,
    PseudoInverse,
}

#[derive(Debug, Clone, PartialEq)]
enum DecompositionType {
    LU,
    QR,
    SVD,
    Eigen,
    Cholesky,
}

impl OperationsPanel {
    pub fn new() -> Self {
        Self {
            selected_operation: BasicOperation::Transpose,
            second_matrix: None,
            scalar_value: 1.0,
            selected_decomposition: DecompositionType::LU,
            compute_u: true,
            compute_v: true,
            rhs_matrix: None,
        }
    }

    pub fn render(&mut self, ui: &mut egui::Ui, project: &Project, selected_matrix: Option<&String>) {
        ui.heading("Operations");

        if selected_matrix.is_none() {
            ui.label("Select a matrix to perform operations");
            return;
        }

        let matrix_name = selected_matrix.unwrap();
        let matrix = project.get_matrix(matrix_name);

        if matrix.is_none() {
            ui.label("Selected matrix not found");
            return;
        }

        let matrix = matrix.unwrap();

        // Show selected matrix info
        ui.group(|ui| {
            ui.label(format!("Selected: {}", matrix_name));
            ui.label(format!("Dimensions: {}×{}", matrix.dimensions().0, matrix.dimensions().1));
            ui.label(format!("Type: {}", if matrix.is_square() { "Square" } else { "Rectangular" }));
        });

        ui.separator();

        // Operation selection
        self.render_operation_selection(ui);

        ui.separator();

        // Operation-specific parameters
        self.render_operation_parameters(ui, project, matrix);

        ui.separator();

        // Execute button
        if ui.button("🔍 Execute Operation").clicked() {
            // TODO: Execute the selected operation
            // This would integrate with the main app to show results
        }

        ui.separator();

        // Matrix decompositions section
        self.render_decomposition_section(ui, matrix);
    }

    fn render_operation_selection(&mut self, ui: &mut egui::Ui) {
        ui.label("Basic Operations:");
        
        egui::ComboBox::from_label("Operation")
            .selected_text(format!("{:?}", self.selected_operation))
            .show_ui(ui, |ui| {
                // Unary operations
                ui.label("Unary Operations:");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::Transpose, "Transpose");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::Inverse, "Inverse");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::Determinant, "Determinant");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::Trace, "Trace");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::Rank, "Rank");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::ConditionNumber, "Condition Number");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::FrobeniusNorm, "Frobenius Norm");

                ui.separator();

                // Binary operations
                ui.label("Binary Operations:");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::Add, "Add");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::Subtract, "Subtract");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::Multiply, "Multiply");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::HadamardProduct, "Hadamard Product");

                ui.separator();

                // Scalar operations
                ui.label("Scalar Operations:");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::ScalarMultiply, "Scalar Multiply");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::ScalarDivide, "Scalar Divide");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::MatrixPower, "Matrix Power");

                ui.separator();

                // Element-wise operations
                ui.label("Element-wise Functions:");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::ElementExp, "Exponential");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::ElementLn, "Natural Log");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::ElementSin, "Sine");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::ElementCos, "Cosine");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::ElementTan, "Tangent");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::ElementSqrt, "Square Root");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::ElementAbs, "Absolute Value");

                ui.separator();

                // System operations
                ui.label("System Operations:");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::SolveLinearSystem, "Solve Linear System");
                ui.selectable_value(&mut self.selected_operation, BasicOperation::PseudoInverse, "Pseudo-Inverse");
            });
    }

    fn render_operation_parameters(&mut self, ui: &mut egui::Ui, project: &Project, matrix: &Matrix) {
        match self.selected_operation {
            // Binary operations need second matrix
            BasicOperation::Add | BasicOperation::Subtract | 
            BasicOperation::Multiply | BasicOperation::HadamardProduct => {
                ui.label("Second Matrix:");
                self.render_matrix_selector(ui, project, &mut self.second_matrix);
                
                if let Some(second_name) = &self.second_matrix {
                    if let Some(second_matrix) = project.get_matrix(second_name) {
                        self.validate_binary_operation(ui, matrix, second_matrix);
                    }
                }
            }

            // Scalar operations need scalar value
            BasicOperation::ScalarMultiply | BasicOperation::ScalarDivide => {
                ui.horizontal(|ui| {
                    ui.label("Scalar value:");
                    ui.add(egui::DragValue::new(&mut self.scalar_value).speed(0.1));
                });
            }

            // Matrix power needs integer exponent
            BasicOperation::MatrixPower => {
                if !matrix.is_square() {
                    ui.colored_label(egui::Color32::RED, "⚠ Matrix power requires a square matrix");
                }
                
                ui.horizontal(|ui| {
                    ui.label("Exponent:");
                    let mut exp = self.scalar_value as i32;
                    ui.add(egui::DragValue::new(&mut exp).range(-10..=10));
                    self.scalar_value = exp as f64;
                });
            }

            // System solving needs RHS matrix
            BasicOperation::SolveLinearSystem => {
                if !matrix.is_square() {
                    ui.colored_label(egui::Color32::RED, "⚠ System solving requires a square coefficient matrix");
                }
                
                ui.label("Right-hand side matrix (b):");
                self.render_matrix_selector(ui, project, &mut self.rhs_matrix);
                
                if let Some(rhs_name) = &self.rhs_matrix {
                    if let Some(rhs_matrix) = project.get_matrix(rhs_name) {
                        if matrix.dimensions().0 != rhs_matrix.dimensions().0 {
                            ui.colored_label(egui::Color32::RED, 
                                format!("⚠ Dimension mismatch: A has {} rows, b has {} rows", 
                                    matrix.dimensions().0, rhs_matrix.dimensions().0));
                        }
                    }
                }
            }

            // Operations that require square matrices
            BasicOperation::Inverse | BasicOperation::Determinant | 
            BasicOperation::Trace => {
                if !matrix.is_square() {
                    ui.colored_label(egui::Color32::RED, 
                        format!("⚠ {} operation requires a square matrix", 
                            format!("{:?}", self.selected_operation)));
                }
            }

            _ => {
                // No additional parameters needed
            }
        }
    }

    fn render_matrix_selector(&self, ui: &mut egui::Ui, project: &Project, selected: &mut Option<String>) {
        let current_text = selected.as_ref()
            .map(|s| s.as_str())
            .unwrap_or("Select matrix...");

        egui::ComboBox::from_id_source("matrix_selector")
            .selected_text(current_text)
            .show_ui(ui, |ui| {
                if ui.selectable_value(selected, None, "None").clicked() {
                    *selected = None;
                }
                
                for name in project.matrix_names() {
                    ui.selectable_value(selected, Some(name.clone()), &name);
                }
            });
    }

    fn validate_binary_operation(&self, ui: &mut egui::Ui, matrix1: &Matrix, matrix2: &Matrix) {
        let (rows1, cols1) = matrix1.dimensions();
        let (rows2, cols2) = matrix2.dimensions();

        match self.selected_operation {
            BasicOperation::Add | BasicOperation::Subtract | BasicOperation::HadamardProduct => {
                if (rows1, cols1) != (rows2, cols2) {
                    ui.colored_label(egui::Color32::RED, 
                        format!("⚠ Dimension mismatch: {}×{} vs {}×{}", rows1, cols1, rows2, cols2));
                } else {
                    ui.colored_label(egui::Color32::GREEN, "✓ Dimensions match");
                }
            }
            
            BasicOperation::Multiply => {
                if cols1 != rows2 {
                    ui.colored_label(egui::Color32::RED, 
                        format!("⚠ Cannot multiply: {} cols ≠ {} rows", cols1, rows2));
                } else {
                    ui.colored_label(egui::Color32::GREEN, 
                        format!("✓ Result will be {}×{}", rows1, cols2));
                }
            }
            
            _ => {}
        }
    }

    fn render_decomposition_section(&mut self, ui: &mut egui::Ui, matrix: &Matrix) {
        ui.heading("Matrix Decompositions");

        egui::ComboBox::from_label("Decomposition Type")
            .selected_text(format!("{:?}", self.selected_decomposition))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.selected_decomposition, DecompositionType::LU, "LU Decomposition");
                ui.selectable_value(&mut self.selected_decomposition, DecompositionType::QR, "QR Decomposition");
                ui.selectable_value(&mut self.selected_decomposition, DecompositionType::SVD, "SVD Decomposition");
                ui.selectable_value(&mut self.selected_decomposition, DecompositionType::Eigen, "Eigenvalue Decomposition");
                ui.selectable_value(&mut self.selected_decomposition, DecompositionType::Cholesky, "Cholesky Decomposition");
            });

        // Decomposition-specific options
        match self.selected_decomposition {
            DecompositionType::LU => {
                if !matrix.is_square() {
                    ui.colored_label(egui::Color32::RED, "⚠ LU decomposition requires a square matrix");
                }
            }
            
            DecompositionType::SVD => {
                ui.checkbox(&mut self.compute_u, "Compute U matrix");
                ui.checkbox(&mut self.compute_v, "Compute V matrix");
            }
            
            DecompositionType::Eigen => {
                if !matrix.is_square() {
                    ui.colored_label(egui::Color32::RED, "⚠ Eigenvalue decomposition requires a square matrix");
                } else if matrix.is_symmetric() {
                    ui.colored_label(egui::Color32::GREEN, "✓ Matrix is symmetric (real eigenvalues)");
                } else {
                    ui.colored_label(egui::Color32::YELLOW, "⚠ Matrix is not symmetric (may have complex eigenvalues)");
                }
            }
            
            DecompositionType::Cholesky => {
                if !matrix.is_square() {
                    ui.colored_label(egui::Color32::RED, "⚠ Cholesky decomposition requires a square matrix");
                } else {
                    ui.colored_label(egui::Color32::YELLOW, "⚠ Matrix must be positive definite");
                }
            }
            
            _ => {}
        }

        if ui.button("🔍 Compute Decomposition").clicked() {
            // TODO: Execute decomposition
        }
    }
}