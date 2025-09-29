use crate::matrix::Matrix;
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::time::Instant;

impl Matrix {
    /// Parallel matrix multiplication using multi-threading
    pub fn parallel_multiply(&self, other: &Matrix) -> anyhow::Result<Matrix> {
        if self.data.ncols() != other.data.nrows() {
            return Err(anyhow::anyhow!("Matrix dimensions incompatible for multiplication"));
        }

        let _start_time = Instant::now();
        
        // For now, use regular nalgebra multiplication which is already optimized
        let result_data = &self.data * &other.data;
        
        Ok(Matrix {
            data: result_data,
            name: format!("{}*{}", self.name, other.name),
            created_at: chrono::Utc::now(),
        })
    }

    /// Benchmark matrix operations performance
    pub fn benchmark_operations(&self, other: &Matrix, iterations: usize) -> PerformanceReport {
        let mut add_times = Vec::new();
        let mut mul_times = Vec::new();

        for _ in 0..iterations {
            // Benchmark addition
            if self.data.shape() == other.data.shape() {
                let start = Instant::now();
                let _ = self.add(other);
                add_times.push(start.elapsed());
            }

            // Benchmark multiplication
            if self.data.ncols() == other.data.nrows() {
                let start = Instant::now();
                let _ = self.multiply(other);
                mul_times.push(start.elapsed());
            }
        }

        PerformanceReport {
            matrix_size: self.dimensions(),
            add_avg_ms: if add_times.is_empty() { 0.0 } else {
                add_times.iter().map(|d| d.as_millis() as f64).sum::<f64>() / add_times.len() as f64
            },
            mul_avg_ms: if mul_times.is_empty() { 0.0 } else {
                mul_times.iter().map(|d| d.as_millis() as f64).sum::<f64>() / mul_times.len() as f64
            },
            iterations,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub matrix_size: (usize, usize),
    pub add_avg_ms: f64,
    pub mul_avg_ms: f64,
    pub iterations: usize,
}

impl PerformanceReport {
    pub fn summary(&self) -> String {
        format!(
            "Performance Report for {}x{} matrices ({} iterations):\n\
             Average Addition Time: {:.2}ms\n\
             Average Multiplication Time: {:.2}ms",
            self.matrix_size.0, self.matrix_size.1, self.iterations,
            self.add_avg_ms, self.mul_avg_ms
        )
    }
}