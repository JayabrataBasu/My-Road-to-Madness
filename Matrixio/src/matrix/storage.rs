// Matrix storage and memory management utilities
use super::Matrix;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Memory-efficient matrix storage manager
pub struct MatrixStorage {
    matrices: HashMap<String, StoredMatrix>,
    memory_limit: usize, // bytes
    current_memory: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMatrix {
    pub matrix: Matrix,
    pub access_count: usize,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    pub memory_size: usize,
}

impl MatrixStorage {
    pub fn new(memory_limit: usize) -> Self {
        Self {
            matrices: HashMap::new(),
            memory_limit,
            current_memory: 0,
        }
    }

    /// Store a matrix with automatic memory management
    pub fn store(&mut self, name: String, matrix: Matrix) -> anyhow::Result<()> {
        let memory_size = self.estimate_memory_size(&matrix);
        
        // Check if we need to free memory
        while self.current_memory + memory_size > self.memory_limit {
            if !self.evict_least_recently_used()? {
                return Err(anyhow::anyhow!("Cannot fit matrix in memory limit"));
            }
        }

        let stored_matrix = StoredMatrix {
            matrix,
            access_count: 1,
            last_accessed: chrono::Utc::now(),
            memory_size,
        };

        self.matrices.insert(name, stored_matrix);
        self.current_memory += memory_size;
        
        Ok(())
    }

    /// Retrieve a matrix and update access statistics
    pub fn get(&mut self, name: &str) -> Option<&Matrix> {
        if let Some(stored) = self.matrices.get_mut(name) {
            stored.access_count += 1;
            stored.last_accessed = chrono::Utc::now();
            Some(&stored.matrix)
        } else {
            None
        }
    }

    /// Remove a matrix from storage
    pub fn remove(&mut self, name: &str) -> Option<Matrix> {
        if let Some(stored) = self.matrices.remove(name) {
            self.current_memory -= stored.memory_size;
            Some(stored.matrix)
        } else {
            None
        }
    }

    /// Get storage statistics
    pub fn statistics(&self) -> StorageStatistics {
        let total_matrices = self.matrices.len();
        let total_accesses = self.matrices.values().map(|s| s.access_count).sum();
        let most_accessed = self.matrices.values()
            .max_by_key(|s| s.access_count)
            .map(|s| s.matrix.name.clone());

        StorageStatistics {
            total_matrices,
            memory_used: self.current_memory,
            memory_limit: self.memory_limit,
            memory_usage_percent: (self.current_memory as f64 / self.memory_limit as f64) * 100.0,
            total_accesses,
            most_accessed_matrix: most_accessed,
        }
    }

    /// Clear all stored matrices
    pub fn clear(&mut self) {
        self.matrices.clear();
        self.current_memory = 0;
    }

    /// Estimate memory usage of a matrix
    fn estimate_memory_size(&self, matrix: &Matrix) -> usize {
        let (rows, cols) = matrix.dimensions();
        rows * cols * std::mem::size_of::<f64>() + 
        matrix.name.len() + 
        std::mem::size_of::<chrono::DateTime<chrono::Utc>>()
    }

    /// Evict the least recently used matrix
    fn evict_least_recently_used(&mut self) -> anyhow::Result<bool> {
        if self.matrices.is_empty() {
            return Ok(false);
        }

        let lru_name = self.matrices
            .iter()
            .min_by_key(|(_, stored)| stored.last_accessed)
            .map(|(name, _)| name.clone())
            .unwrap();

        self.remove(&lru_name);
        Ok(true)
    }

    /// Get list of all stored matrix names
    pub fn list_matrices(&self) -> Vec<String> {
        self.matrices.keys().cloned().collect()
    }

    /// Check if storage contains a matrix
    pub fn contains(&self, name: &str) -> bool {
        self.matrices.contains_key(name)
    }

    /// Get matrix info without accessing it
    pub fn get_info(&self, name: &str) -> Option<MatrixInfo> {
        self.matrices.get(name).map(|stored| {
            let (rows, cols) = stored.matrix.dimensions();
            MatrixInfo {
                name: stored.matrix.name.clone(),
                dimensions: (rows, cols),
                memory_size: stored.memory_size,
                access_count: stored.access_count,
                last_accessed: stored.last_accessed,
                is_square: stored.matrix.is_square(),
            }
        })
    }
}

#[derive(Debug, Clone)]
pub struct StorageStatistics {
    pub total_matrices: usize,
    pub memory_used: usize,
    pub memory_limit: usize,
    pub memory_usage_percent: f64,
    pub total_accesses: usize,
    pub most_accessed_matrix: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MatrixInfo {
    pub name: String,
    pub dimensions: (usize, usize),
    pub memory_size: usize,
    pub access_count: usize,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    pub is_square: bool,
}

/// Matrix compression utilities for large matrices
pub struct MatrixCompression;

impl MatrixCompression {
    /// Compress matrix using sparse representation for matrices with many zeros
    pub fn compress_sparse(matrix: &Matrix) -> CompressedMatrix {
        let (rows, cols) = matrix.dimensions();
        let mut non_zero_elements = Vec::new();
        let mut zero_count = 0;

        for i in 0..rows {
            for j in 0..cols {
                if let Some(value) = matrix.get(i, j) {
                    if value.abs() > f64::EPSILON {
                        non_zero_elements.push(SparseElement { row: i, col: j, value });
                    } else {
                        zero_count += 1;
                    }
                }
            }
        }

        let sparsity = zero_count as f64 / (rows * cols) as f64;
        let compression_ratio = if non_zero_elements.is_empty() {
            1.0
        } else {
            (rows * cols) as f64 / non_zero_elements.len() as f64
        };

        CompressedMatrix {
            original_dimensions: (rows, cols),
            non_zero_elements,
            sparsity,
            compression_ratio,
            original_name: matrix.name.clone(),
        }
    }

    /// Decompress sparse matrix back to full matrix
    pub fn decompress_sparse(compressed: &CompressedMatrix) -> anyhow::Result<Matrix> {
        let (rows, cols) = compressed.original_dimensions;
        let mut matrix = Matrix::new(rows, cols, compressed.original_name.clone());

        for element in &compressed.non_zero_elements {
            matrix.set(element.row, element.col, element.value)?;
        }

        Ok(matrix)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedMatrix {
    pub original_dimensions: (usize, usize),
    pub non_zero_elements: Vec<SparseElement>,
    pub sparsity: f64, // Percentage of zero elements
    pub compression_ratio: f64,
    pub original_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseElement {
    pub row: usize,
    pub col: usize,
    pub value: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_storage() {
        let mut storage = MatrixStorage::new(1024 * 1024); // 1MB limit
        
        let matrix = Matrix::new(10, 10, "test".to_string());
        storage.store("test".to_string(), matrix).unwrap();
        
        assert!(storage.contains("test"));
        assert_eq!(storage.list_matrices().len(), 1);
        
        let retrieved = storage.get("test");
        assert!(retrieved.is_some());
        
        let stats = storage.statistics();
        assert_eq!(stats.total_matrices, 1);
        assert!(stats.memory_used > 0);
    }

    #[test]
    fn test_sparse_compression() {
        // Create a sparse matrix (mostly zeros)
        let mut data = vec![vec![0.0; 10]; 10];
        data[0][0] = 1.0;
        data[5][5] = 2.0;
        data[9][9] = 3.0;
        
        let matrix = Matrix::from_vec(data, "sparse".to_string()).unwrap();
        let compressed = MatrixCompression::compress_sparse(&matrix);
        
        assert_eq!(compressed.non_zero_elements.len(), 3);
        assert!(compressed.sparsity > 0.9); // Over 90% zeros
        
        let decompressed = MatrixCompression::decompress_sparse(&compressed).unwrap();
        assert_eq!(decompressed.get(0, 0), Some(1.0));
        assert_eq!(decompressed.get(5, 5), Some(2.0));
        assert_eq!(decompressed.get(9, 9), Some(3.0));
        assert_eq!(decompressed.get(1, 1), Some(0.0));
    }

    #[test]
    fn test_storage_eviction() {
        let mut storage = MatrixStorage::new(100); // Very small limit to force eviction
        
        let matrix1 = Matrix::new(2, 2, "matrix1".to_string());
        let matrix2 = Matrix::new(2, 2, "matrix2".to_string());
        
        storage.store("matrix1".to_string(), matrix1).unwrap();
        
        // This should trigger eviction of matrix1
        storage.store("matrix2".to_string(), matrix2).unwrap();
        
        // matrix1 might be evicted, matrix2 should be present
        assert!(storage.contains("matrix2"));
    }
}