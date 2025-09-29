pub mod operations;
pub mod storage;
pub mod decomposition;
pub mod parallel;

use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Array1};
use serde::{Deserialize, Serialize};
use std::fmt;
use anyhow::{Result, Context};
use num_traits::{Float, Zero, One};

/// High-performance matrix structure optimized for large-scale operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Matrix {
    pub data: DMatrix<f64>,
    pub name: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Matrix {
    /// Create a new matrix with given dimensions
    pub fn new(rows: usize, cols: usize, name: String) -> Self {
        Self {
            data: DMatrix::zeros(rows, cols),
            name,
            created_at: chrono::Utc::now(),
        }
    }

    /// Create a matrix from a 2D vector
    pub fn from_vec(data: Vec<Vec<f64>>, name: String) -> Result<Self> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Cannot create matrix from empty data"));
        }
        
        let rows = data.len();
        let cols = data[0].len();
        
        // Validate that all rows have the same length
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(anyhow::anyhow!(
                    "Row {} has length {}, expected {}", i, row.len(), cols
                ));
            }
        }
        
        let flat_data: Vec<f64> = data.into_iter().flatten().collect();
        let matrix_data = DMatrix::from_row_slice(rows, cols, &flat_data);
        
        Ok(Self {
            data: matrix_data,
            name,
            created_at: chrono::Utc::now(),
        })
    }

    /// Create an identity matrix
    pub fn identity(size: usize, name: String) -> Self {
        Self {
            data: DMatrix::identity(size, size),
            name,
            created_at: chrono::Utc::now(),
        }
    }

    /// Create a random matrix (useful for testing)
    pub fn random(rows: usize, cols: usize, name: String) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-10.0..10.0));
        
        Self {
            data,
            name,
            created_at: chrono::Utc::now(),
        }
    }

    /// Get matrix dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.data.nrows(), self.data.ncols())
    }

    /// Check if matrix is square
    pub fn is_square(&self) -> bool {
        self.data.nrows() == self.data.ncols()
    }

    /// Get element at position (i, j)
    pub fn get(&self, i: usize, j: usize) -> Option<f64> {
        self.data.get((i, j)).copied()
    }

    /// Set element at position (i, j)
    pub fn set(&mut self, i: usize, j: usize, value: f64) -> Result<()> {
        if i >= self.data.nrows() || j >= self.data.ncols() {
            return Err(anyhow::anyhow!(
                "Index ({}, {}) out of bounds for matrix of size ({}, {})",
                i, j, self.data.nrows(), self.data.ncols()
            ));
        }
        
        self.data[(i, j)] = value;
        Ok(())
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> Self {
        Self {
            data: self.data.transpose(),
            name: format!("{}_T", self.name),
            created_at: chrono::Utc::now(),
        }
    }

    /// Calculate the trace (sum of diagonal elements)
    pub fn trace(&self) -> Result<f64> {
        if !self.is_square() {
            return Err(anyhow::anyhow!("Trace is only defined for square matrices"));
        }
        
        Ok(self.data.diagonal().sum())
    }

    /// Calculate the Frobenius norm
    pub fn frobenius_norm(&self) -> f64 {
        self.data.norm()
    }

    /// Get a sub-matrix
    pub fn submatrix(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Result<Self> {
        if row_end <= row_start || col_end <= col_start {
            return Err(anyhow::anyhow!("Invalid submatrix bounds"));
        }
        
        if row_end > self.data.nrows() || col_end > self.data.ncols() {
            return Err(anyhow::anyhow!("Submatrix bounds exceed matrix dimensions"));
        }
        
        let sub_data = self.data.view((row_start, col_start), (row_end - row_start, col_end - col_start));
        
        Ok(Self {
            data: sub_data.into(),
            name: format!("{}_sub", self.name),
            created_at: chrono::Utc::now(),
        })
    }

    /// Convert to ndarray for interoperability
    pub fn to_ndarray(&self) -> Array2<f64> {
        let (rows, cols) = self.dimensions();
        Array2::from_shape_vec((rows, cols), self.data.iter().cloned().collect())
            .expect("Failed to convert to ndarray")
    }

    /// Create from ndarray
    pub fn from_ndarray(array: Array2<f64>, name: String) -> Self {
        let (rows, cols) = array.dim();
        let data = DMatrix::from_iterator(rows, cols, array.iter().cloned());
        
        Self {
            data,
            name,
            created_at: chrono::Utc::now(),
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix '{}' ({}x{})", self.name, self.data.nrows(), self.data.ncols())?;
        
        // For small matrices, show the actual data
        if self.data.nrows() <= 10 && self.data.ncols() <= 10 {
            write!(f, ":\n{}", self.data)?;
        } else {
            write!(f, " [Large matrix - use preview for display]")?;
        }
        
        Ok(())
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m = Matrix::new(3, 3, "test".to_string());
        assert_eq!(m.dimensions(), (3, 3));
        assert!(m.is_square());
    }

    #[test]
    fn test_matrix_from_vec() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let m = Matrix::from_vec(data, "test".to_string()).unwrap();
        assert_eq!(m.dimensions(), (2, 3));
        assert_eq!(m.get(0, 0), Some(1.0));
        assert_eq!(m.get(1, 2), Some(6.0));
    }

    #[test]
    fn test_identity_matrix() {
        let m = Matrix::identity(3, "identity".to_string());
        assert_eq!(m.get(0, 0), Some(1.0));
        assert_eq!(m.get(0, 1), Some(0.0));
        assert_eq!(m.get(1, 1), Some(1.0));
    }

    #[test]
    fn test_matrix_operations() {
        let mut m = Matrix::new(2, 2, "test".to_string());
        m.set(0, 0, 1.0).unwrap();
        m.set(1, 1, 1.0).unwrap();
        
        assert_eq!(m.trace().unwrap(), 2.0);
        
        let t = m.transpose();
        assert_eq!(t.get(0, 0), Some(1.0));
        assert_eq!(t.get(1, 1), Some(1.0));
    }
}