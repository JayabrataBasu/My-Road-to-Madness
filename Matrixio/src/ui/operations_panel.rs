use crate::data::project::Project;
use crate::matrix::Matrix;
use crate::ui::{OperationResult, DecompositionResult};
use egui;

#[derive(Debug, Clone, PartialEq)]
pub enum BasicOperation {
    Add,
    Subtract,
    Multiply,
    HadamardProduct,
    Transpose,
    Inverse,
    Determinant,
    Rank,
    ConditionNumber,
    SolveLinearSystem,
}

impl BasicOperation {
    pub fn name(&self) -> &'static str {
        match self {
            BasicOperation::Add => "Addition",
            BasicOperation::Subtract => "Subtraction", 
            BasicOperation::Multiply => "Matrix Multiplication",
            BasicOperation::HadamardProduct => "Hadamard Product",
            BasicOperation::Transpose => "Transpose",
            BasicOperation::Inverse => "Inverse",
            BasicOperation::Determinant => "Determinant",
            BasicOperation::Rank => "Rank",
            BasicOperation::ConditionNumber => "Condition Number",
            BasicOperation::SolveLinearSystem => "Solve Linear System",
        }
    }
    
    pub fn description(&self) -> &'static str {
        match self {
            BasicOperation::Add => "Element-wise addition of two matrices",
            BasicOperation::Subtract => "Element-wise subtraction of two matrices",
            BasicOperation::Multiply => "Matrix multiplication (A × B)",
            BasicOperation::HadamardProduct => "Element-wise multiplication",
            BasicOperation::Transpose => "Transpose matrix (A^T)",
            BasicOperation::Inverse => "Matrix inverse (A^-1)",
            BasicOperation::Determinant => "Calculate determinant",
            BasicOperation::Rank => "Calculate matrix rank",
            BasicOperation::ConditionNumber => "Calculate condition number",
            BasicOperation::SolveLinearSystem => "Solve Ax = b",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecompositionOperation {
    LU,
    QR,
    SVD,
    Eigenvalue,
    Cholesky,
}

impl DecompositionOperation {
    pub fn name(&self) -> &'static str {
        match self {
            DecompositionOperation::LU => "LU Decomposition",
            DecompositionOperation::QR => "QR Decomposition",
            DecompositionOperation::SVD => "Singular Value Decomposition",
            DecompositionOperation::Eigenvalue => "Eigenvalue Decomposition",
            DecompositionOperation::Cholesky => "Cholesky Decomposition",
        }
    }
}

#[derive(Default)]
pub struct OperationsPanel {
    pub selected_operation: BasicOperation,
    pub selected_decomposition: DecompositionOperation,
    pub second_matrix: Option<String>,
    pub rhs_matrix: Option<String>,
    pub show_operations: bool,
    pub show_decompositions: bool,
}

impl Default for BasicOperation {
    fn default() -> Self {
        BasicOperation::Add
    }
}

impl Default for DecompositionOperation {
    fn default() -> Self {
        DecompositionOperation::LU
    }
}

impl OperationsPanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn show(&mut self, ui: &mut egui::Ui, project: &Project, selected_matrix: &str) -> Option<OperationResult> {
        let matrix = match project.get_matrix(selected_matrix) {
            Some(m) => m,
            None => {
                ui.label("No matrix selected");
                return None;
            }
        };

        ui.heading("Operations");
        
        let mut result = None;

        // Basic Operations
        ui.collapsing("Basic Operations", |ui| {
            // Operation selector
            let current_op = self.selected_operation.name();
            egui::ComboBox::from_id_salt("basic_operations")
                .selected_text(current_op)
                .show_ui(ui, |ui| {
                    for op in [
                        BasicOperation::Add,
                        BasicOperation::Subtract,
                        BasicOperation::Multiply,
                        BasicOperation::Transpose,
                        BasicOperation::Inverse,
                        BasicOperation::Determinant,
                    ] {
                        ui.selectable_value(&mut self.selected_operation, op.clone(), op.name());
                    }
                });

            ui.label(self.selected_operation.description());

            // Show second matrix selector for binary operations
            match &self.selected_operation {
                BasicOperation::Add | BasicOperation::Subtract | BasicOperation::Multiply => {
                    ui.label("Second Matrix:");
                    self.render_simple_matrix_selector(ui, project);
                }
                _ => {}
            }

            if ui.button("Execute").clicked() {
                result = self.execute_operation(matrix, project);
            }
        });

        result
    }

    fn render_simple_matrix_selector(&mut self, ui: &mut egui::Ui, project: &Project) {
        let current_text = self.second_matrix.as_ref()
            .map(|s| s.as_str())
            .unwrap_or("Select matrix...");

        egui::ComboBox::from_id_salt("second_matrix_selector")
            .selected_text(current_text)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.second_matrix, None, "None");
                for name in project.matrices.keys() {
                    ui.selectable_value(&mut self.second_matrix, Some(name.clone()), name);
                }
            });
    }

    fn execute_operation(&self, matrix: &Matrix, project: &Project) -> Option<OperationResult> {
        match &self.selected_operation {
            BasicOperation::Transpose => {
                Some(OperationResult::Matrix(matrix.transpose()))
            }
            BasicOperation::Determinant => {
                match matrix.determinant() {
                    Ok(det) => Some(OperationResult::Scalar(det)),
                    Err(e) => {
                        eprintln!("Error calculating determinant: {}", e);
                        None
                    }
                }
            }
            BasicOperation::Add => {
                if let Some(second_name) = &self.second_matrix {
                    if let Some(second_matrix) = project.get_matrix(second_name) {
                        match matrix.add(second_matrix) {
                            Ok(result) => Some(OperationResult::Matrix(result)),
                            Err(e) => {
                                eprintln!("Error in addition: {}", e);
                                None
                            }
                        }
                    } else { None }
                } else { None }
            }
            BasicOperation::Subtract => {
                if let Some(second_name) = &self.second_matrix {
                    if let Some(second_matrix) = project.get_matrix(second_name) {
                        match matrix.subtract(second_matrix) {
                            Ok(result) => Some(OperationResult::Matrix(result)),
                            Err(e) => {
                                eprintln!("Error in subtraction: {}", e);
                                None
                            }
                        }
                    } else { None }
                } else { None }
            }
            BasicOperation::Multiply => {
                if let Some(second_name) = &self.second_matrix {
                    if let Some(second_matrix) = project.get_matrix(second_name) {
                        match matrix.multiply(second_matrix) {
                            Ok(result) => Some(OperationResult::Matrix(result)),
                            Err(e) => {
                                eprintln!("Error in multiplication: {}", e);
                                None
                            }
                        }
                    } else { None }
                } else { None }
            }
            _ => None
        }
    }
}