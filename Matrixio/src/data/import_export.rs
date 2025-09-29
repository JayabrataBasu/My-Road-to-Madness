use crate::matrix::Matrix;
use anyhow::{Result, Context};
use std::path::Path;
use std::io::{Write, BufRead, BufReader};

/// Matrix import/export utilities
pub struct ImportExport;

impl ImportExport {
    /// Export matrix to CSV format
    pub fn export_csv<P: AsRef<Path>>(matrix: &Matrix, path: P) -> Result<()> {
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

    /// Import matrix from CSV format
    pub fn import_csv<P: AsRef<Path>>(path: P, name: String) -> Result<Matrix> {
        let file = std::fs::File::open(path)
            .context("Failed to open CSV file")?;
        
        let reader = BufReader::new(file);
        let mut data = Vec::new();
        
        for line in reader.lines() {
            let line = line.context("Failed to read CSV line")?;
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
        
        Matrix::from_vec(data, name)
    }

    /// Export matrix to JSON format
    pub fn export_json<P: AsRef<Path>>(matrix: &Matrix, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(matrix)
            .context("Failed to serialize matrix to JSON")?;
        
        std::fs::write(path, json)
            .context("Failed to write JSON file")?;
        
        Ok(())
    }

    /// Import matrix from JSON format
    pub fn import_json<P: AsRef<Path>>(path: P) -> Result<Matrix> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read JSON file")?;
        
        let matrix: Matrix = serde_json::from_str(&content)
            .context("Failed to deserialize matrix from JSON")?;
        
        Ok(matrix)
    }

    /// Export matrix to MATLAB .mat format (simplified)
    pub fn export_matlab<P: AsRef<Path>>(matrix: &Matrix, path: P) -> Result<()> {
        // Simplified MATLAB export - just write as text format
        let mut file = std::fs::File::create(path)
            .context("Failed to create MATLAB file")?;
        
        let (rows, cols) = matrix.dimensions();
        
        writeln!(file, "% Matrix: {}", matrix.name)?;
        writeln!(file, "% Dimensions: {}x{}", rows, cols)?;
        writeln!(file, "{} = [", matrix.name)?;
        
        for i in 0..rows {
            write!(file, "    ")?;
            for j in 0..cols {
                let value = matrix.get(i, j).unwrap_or(0.0);
                if j == cols - 1 {
                    write!(file, "{:.6}", value)?;
                } else {
                    write!(file, "{:.6}, ", value)?;
                }
            }
            if i == rows - 1 {
                writeln!(file)?;
            } else {
                writeln!(file, ";")?;
            }
        }
        
        writeln!(file, "];")?;
        
        Ok(())
    }

    /// Export matrix to NumPy .npy format (simplified text version)
    pub fn export_numpy<P: AsRef<Path>>(matrix: &Matrix, path: P) -> Result<()> {
        let mut file = std::fs::File::create(path)
            .context("Failed to create NumPy file")?;
        
        let (rows, cols) = matrix.dimensions();
        
        writeln!(file, "# NumPy array export")?;
        writeln!(file, "# Shape: ({}, {})", rows, cols)?;
        writeln!(file, "import numpy as np")?;
        writeln!(file, "{} = np.array([", matrix.name)?;
        
        for i in 0..rows {
            write!(file, "    [")?;
            for j in 0..cols {
                let value = matrix.get(i, j).unwrap_or(0.0);
                if j == cols - 1 {
                    write!(file, "{:.6}", value)?;
                } else {
                    write!(file, "{:.6}, ", value)?;
                }
            }
            if i == rows - 1 {
                writeln!(file, "]")?;
            } else {
                writeln!(file, "],")?;
            }
        }
        
        writeln!(file, "])")?;
        
        Ok(())
    }

    /// Determine file format from extension
    pub fn detect_format<P: AsRef<Path>>(path: P) -> Option<FileFormat> {
        path.as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "csv" => Some(FileFormat::Csv),
                "json" => Some(FileFormat::Json),
                "mat" | "m" => Some(FileFormat::Matlab),
                "npy" => Some(FileFormat::Numpy),
                "txt" => Some(FileFormat::Text),
                _ => None,
            })
    }

    /// Export matrix in detected format
    pub fn export_auto<P: AsRef<Path>>(matrix: &Matrix, path: P) -> Result<()> {
        match Self::detect_format(&path) {
            Some(FileFormat::Csv) => Self::export_csv(matrix, path),
            Some(FileFormat::Json) => Self::export_json(matrix, path),
            Some(FileFormat::Matlab) => Self::export_matlab(matrix, path),
            Some(FileFormat::Numpy) => Self::export_numpy(matrix, path),
            Some(FileFormat::Text) => Self::export_csv(matrix, path), // Default to CSV for text
            None => Err(anyhow::anyhow!("Unsupported file format")),
        }
    }

    /// Import matrix in detected format
    pub fn import_auto<P: AsRef<Path>>(path: P, name: String) -> Result<Matrix> {
        match Self::detect_format(&path) {
            Some(FileFormat::Csv) => Self::import_csv(path, name),
            Some(FileFormat::Json) => Self::import_json(path),
            Some(FileFormat::Text) => Self::import_csv(path, name), // Default to CSV for text
            _ => Err(anyhow::anyhow!("Unsupported file format for import")),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FileFormat {
    Csv,
    Json,
    Matlab,
    Numpy,
    Text,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_csv_export_import() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_matrix.csv");
        
        let original = Matrix::from_vec(
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            "test".to_string()
        ).unwrap();
        
        // Export
        ImportExport::export_csv(&original, &file_path).unwrap();
        
        // Import
        let imported = ImportExport::import_csv(&file_path, "test_imported".to_string()).unwrap();
        
        assert_eq!(imported.dimensions(), (2, 3));
        assert_eq!(imported.get(0, 0), Some(1.0));
        assert_eq!(imported.get(1, 2), Some(6.0));
    }

    #[test]
    fn test_json_export_import() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_matrix.json");
        
        let original = Matrix::from_vec(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            "test".to_string()
        ).unwrap();
        
        // Export
        ImportExport::export_json(&original, &file_path).unwrap();
        
        // Import
        let imported = ImportExport::import_json(&file_path).unwrap();
        
        assert_eq!(imported.dimensions(), (2, 2));
        assert_eq!(imported.name, "test");
        assert_eq!(imported.get(1, 1), Some(4.0));
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(ImportExport::detect_format("test.csv"), Some(FileFormat::Csv));
        assert_eq!(ImportExport::detect_format("test.json"), Some(FileFormat::Json));
        assert_eq!(ImportExport::detect_format("test.mat"), Some(FileFormat::Matlab));
        assert_eq!(ImportExport::detect_format("test.unknown"), None);
    }
}