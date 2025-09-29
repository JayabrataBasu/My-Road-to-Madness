use super::{MatrixCollection, ProjectMetadata};
use crate::matrix::Matrix;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use anyhow::{Result, Context};

/// Project management for matrix collections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub matrices: HashMap<String, Matrix>,
    pub metadata: ProjectMetadata,
}

impl Project {
    pub fn new() -> Self {
        Self {
            matrices: HashMap::new(),
            metadata: ProjectMetadata::default(),
        }
    }

    pub fn with_name(name: String) -> Self {
        let mut project = Self::new();
        project.metadata.name = name;
        project
    }

    /// Add a matrix to the project
    pub fn add_matrix(&mut self, name: String, matrix: Matrix) -> Result<()> {
        if self.matrices.contains_key(&name) {
            return Err(anyhow::anyhow!("Matrix '{}' already exists", name));
        }
        
        self.matrices.insert(name, matrix);
        self.update_modified_time();
        Ok(())
    }

    /// Remove a matrix from the project
    pub fn remove_matrix(&mut self, name: &str) -> Option<Matrix> {
        let result = self.matrices.remove(name);
        if result.is_some() {
            self.update_modified_time();
        }
        result
    }

    /// Get a matrix by name
    pub fn get_matrix(&self, name: &str) -> Option<&Matrix> {
        self.matrices.get(name)
    }

    /// Get a mutable reference to a matrix
    pub fn get_matrix_mut(&mut self, name: &str) -> Option<&mut Matrix> {
        self.matrices.get_mut(name)
    }

    /// List all matrix names
    pub fn matrix_names(&self) -> Vec<String> {
        self.matrices.keys().cloned().collect()
    }

    /// Get project statistics
    pub fn statistics(&self) -> ProjectStatistics {
        let total_matrices = self.matrices.len();
        let mut total_elements = 0;
        let mut largest_matrix = (0, 0);
        let mut memory_usage = 0;

        for matrix in self.matrices.values() {
            let (rows, cols) = matrix.dimensions();
            total_elements += rows * cols;
            
            if rows * cols > largest_matrix.0 * largest_matrix.1 {
                largest_matrix = (rows, cols);
            }
            
            // Estimate memory usage (8 bytes per f64 element)
            memory_usage += rows * cols * 8;
        }

        ProjectStatistics {
            total_matrices,
            total_elements,
            largest_matrix,
            estimated_memory_bytes: memory_usage,
        }
    }

    /// Save project to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .context("Failed to serialize project")?;
        
        std::fs::write(path, json)
            .context("Failed to write project file")?;
        
        Ok(())
    }

    /// Load project from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read project file")?;
        
        let project: Project = serde_json::from_str(&content)
            .context("Failed to deserialize project")?;
        
        Ok(project)
    }

    /// Export matrices to CSV format
    pub fn export_matrices_csv<P: AsRef<Path>>(&self, directory: P) -> Result<()> {
        let dir = directory.as_ref();
        
        if !dir.exists() {
            std::fs::create_dir_all(dir)
                .context("Failed to create export directory")?;
        }

        for (name, matrix) in &self.matrices {
            let filename = format!("{}.csv", name);
            let filepath = dir.join(filename);
            
            self.export_matrix_csv(matrix, filepath)?;
        }

        Ok(())
    }

    /// Export a single matrix to CSV
    fn export_matrix_csv<P: AsRef<Path>>(&self, matrix: &Matrix, path: P) -> Result<()> {
        use std::io::Write;
        
        let mut file = std::fs::File::create(path)
            .context("Failed to create CSV file")?;
        
        let (rows, cols) = matrix.dimensions();
        
        for i in 0..rows {
            let mut row_data = Vec::new();
            for j in 0..cols {
                let value = matrix.get(i, j).unwrap_or(0.0);
                row_data.push(value.to_string());
            }
            
            writeln!(file, "{}", row_data.join(","))
                .context("Failed to write CSV row")?;
        }

        Ok(())
    }

    /// Import matrix from CSV
    pub fn import_matrix_csv<P: AsRef<Path>>(&mut self, path: P, name: String) -> Result<()> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read CSV file")?;
        
        let mut data = Vec::new();
        
        for line in content.lines() {
            let row: Result<Vec<f64>, _> = line
                .split(',')
                .map(|s| s.trim().parse::<f64>())
                .collect();
            
            match row {
                Ok(row_data) => data.push(row_data),
                Err(_) => return Err(anyhow::anyhow!("Invalid number in CSV file")),
            }
        }
        
        if data.is_empty() {
            return Err(anyhow::anyhow!("CSV file contains no data"));
        }
        
        let matrix = Matrix::from_vec(data, name.clone())?;
        self.add_matrix(name, matrix)?;
        
        Ok(())
    }

    /// Clear all matrices
    pub fn clear(&mut self) {
        self.matrices.clear();
        self.update_modified_time();
    }

    /// Update the modified timestamp
    fn update_modified_time(&mut self) {
        self.metadata.modified_at = chrono::Utc::now();
    }

    /// Validate project integrity
    pub fn validate(&self) -> Result<()> {
        // Check for duplicate matrix names (should not happen with HashMap)
        if self.matrices.len() != self.matrices.keys().collect::<std::collections::HashSet<_>>().len() {
            return Err(anyhow::anyhow!("Duplicate matrix names detected"));
        }

        // Validate each matrix
        for (name, matrix) in &self.matrices {
            if matrix.name != *name {
                return Err(anyhow::anyhow!(
                    "Matrix name mismatch: key '{}' vs matrix name '{}'", 
                    name, matrix.name
                ));
            }
            
            let (rows, cols) = matrix.dimensions();
            if rows == 0 || cols == 0 {
                return Err(anyhow::anyhow!(
                    "Matrix '{}' has invalid dimensions: {}x{}", 
                    name, rows, cols
                ));
            }
        }

        Ok(())
    }
}

impl Default for Project {
    fn default() -> Self {
        Self::new()
    }
}

/// Project statistics
#[derive(Debug, Clone)]
pub struct ProjectStatistics {
    pub total_matrices: usize,
    pub total_elements: usize,
    pub largest_matrix: (usize, usize),
    pub estimated_memory_bytes: usize,
}

impl ProjectStatistics {
    /// Get memory usage in human-readable format
    pub fn memory_usage_string(&self) -> String {
        let bytes = self.estimated_memory_bytes;
        
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_project_creation() {
        let project = Project::new();
        assert_eq!(project.matrices.len(), 0);
        assert_eq!(project.metadata.name, "Untitled Project");
    }

    #[test]
    fn test_add_matrix() {
        let mut project = Project::new();
        let matrix = Matrix::new(2, 2, "test".to_string());
        
        project.add_matrix("test".to_string(), matrix).unwrap();
        assert_eq!(project.matrices.len(), 1);
        assert!(project.get_matrix("test").is_some());
    }

    #[test]
    fn test_duplicate_matrix_name() {
        let mut project = Project::new();
        let matrix1 = Matrix::new(2, 2, "test".to_string());
        let matrix2 = Matrix::new(3, 3, "test".to_string());
        
        project.add_matrix("test".to_string(), matrix1).unwrap();
        assert!(project.add_matrix("test".to_string(), matrix2).is_err());
    }

    #[test]
    fn test_save_load_project() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_project.json");
        
        let mut project = Project::with_name("Test Project".to_string());
        let matrix = Matrix::from_vec(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]], 
            "test_matrix".to_string()
        ).unwrap();
        
        project.add_matrix("test_matrix".to_string(), matrix).unwrap();
        
        // Save project
        project.save_to_file(&file_path).unwrap();
        
        // Load project
        let loaded_project = Project::load_from_file(&file_path).unwrap();
        
        assert_eq!(loaded_project.metadata.name, "Test Project");
        assert_eq!(loaded_project.matrices.len(), 1);
        assert!(loaded_project.get_matrix("test_matrix").is_some());
    }

    #[test]
    fn test_project_statistics() {
        let mut project = Project::new();
        
        let matrix1 = Matrix::new(2, 3, "matrix1".to_string());
        let matrix2 = Matrix::new(4, 4, "matrix2".to_string());
        
        project.add_matrix("matrix1".to_string(), matrix1).unwrap();
        project.add_matrix("matrix2".to_string(), matrix2).unwrap();
        
        let stats = project.statistics();
        
        assert_eq!(stats.total_matrices, 2);
        assert_eq!(stats.total_elements, 6 + 16); // 2*3 + 4*4
        assert_eq!(stats.largest_matrix, (4, 4));
    }
}