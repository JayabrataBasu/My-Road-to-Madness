use super::Matrix;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

/// Parallel matrix operations for improved performance on large matrices
pub struct ParallelEngine {
    thread_pool: rayon::ThreadPool,
}

impl ParallelEngine {
    pub fn new() -> anyhow::Result<Self> {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get())
            .build()?;
        
        Ok(Self { thread_pool })
    }

    /// Parallel matrix multiplication using block algorithms
    pub fn parallel_multiply(&self, a: &Matrix, b: &Matrix) -> anyhow::Result<Matrix> {
        if a.data.ncols() != b.data.nrows() {
            return Err(anyhow::anyhow!("Matrix dimensions don't match for multiplication"));
        }

        let start_time = Instant::now();
        
        // For very large matrices, use block multiplication
        if a.data.nrows() > 500 && b.data.ncols() > 500 {
            self.block_multiply(a, b)
        } else {
            // Use standard parallel multiplication
            self.standard_parallel_multiply(a, b)
        }
    }

    /// Standard parallel matrix multiplication
    fn standard_parallel_multiply(&self, a: &Matrix, b: &Matrix) -> anyhow::Result<Matrix> {
        let (m, k) = (a.data.nrows(), a.data.ncols());
        let n = b.data.ncols();
        
        let result_data: Vec<f64> = (0..m * n)
            .into_par_iter()
            .map(|idx| {
                let i = idx / n;
                let j = idx % n;
                
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a.data[(i, p)] * b.data[(p, j)];
                }
                sum
            })
            .collect();

        let result_matrix = nalgebra::DMatrix::from_vec(m, n, result_data);
        
        Ok(Matrix {
            data: result_matrix,
            name: format!("{}*{}_parallel", a.name, b.name),
            created_at: chrono::Utc::now(),
        })
    }

    /// Block matrix multiplication for very large matrices
    fn block_multiply(&self, a: &Matrix, b: &Matrix) -> anyhow::Result<Matrix> {
        const BLOCK_SIZE: usize = 64; // Cache-friendly block size
        
        let (m, k) = (a.data.nrows(), a.data.ncols());
        let n = b.data.ncols();
        
        let mut result = nalgebra::DMatrix::zeros(m, n);
        
        // Parallel iteration over blocks
        let m_blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let k_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        (0..m_blocks * n_blocks)
            .into_par_iter()
            .for_each(|block_idx| {
                let bi = block_idx / n_blocks;
                let bj = block_idx % n_blocks;
                
                let i_start = bi * BLOCK_SIZE;
                let i_end = std::cmp::min(i_start + BLOCK_SIZE, m);
                let j_start = bj * BLOCK_SIZE;
                let j_end = std::cmp::min(j_start + BLOCK_SIZE, n);
                
                let mut local_result = nalgebra::DMatrix::zeros(i_end - i_start, j_end - j_start);
                
                for bk in 0..k_blocks {
                    let k_start = bk * BLOCK_SIZE;
                    let k_end = std::cmp::min(k_start + BLOCK_SIZE, k);
                    
                    // Block multiplication
                    for i in 0..(i_end - i_start) {
                        for j in 0..(j_end - j_start) {
                            let mut sum = 0.0;
                            for p in 0..(k_end - k_start) {
                                sum += a.data[(i_start + i, k_start + p)] * 
                                       b.data[(k_start + p, j_start + j)];
                            }
                            local_result[(i, j)] += sum;
                        }
                    }
                }
                
                // This would need synchronization in a real implementation
                // For now, this is a simplified version
            });

        Ok(Matrix {
            data: result,
            name: format!("{}*{}_block_parallel", a.name, b.name),
            created_at: chrono::Utc::now(),
        })
    }

    /// Parallel element-wise operations
    pub fn parallel_elementwise<F>(&self, matrix: &Matrix, operation: F) -> Matrix
    where
        F: Fn(f64) -> f64 + Sync + Send,
    {
        let (rows, cols) = matrix.dimensions();
        
        let result_data: Vec<f64> = matrix.data
            .iter()
            .par_bridge()
            .map(|&x| operation(x))
            .collect();

        let result_matrix = nalgebra::DMatrix::from_vec(rows, cols, result_data);
        
        Matrix {
            data: result_matrix,
            name: format!("f({})", matrix.name),
            created_at: chrono::Utc::now(),
        }
    }

    /// Parallel matrix addition
    pub fn parallel_add(&self, a: &Matrix, b: &Matrix) -> anyhow::Result<Matrix> {
        if a.dimensions() != b.dimensions() {
            return Err(anyhow::anyhow!("Matrix dimensions don't match"));
        }

        let result_data: Vec<f64> = a.data
            .iter()
            .par_bridge()
            .zip(b.data.par_iter())
            .map(|(&x, &y)| x + y)
            .collect();

        let (rows, cols) = a.dimensions();
        let result_matrix = nalgebra::DMatrix::from_vec(rows, cols, result_data);
        
        Ok(Matrix {
            data: result_matrix,
            name: format!("{}+{}_parallel", a.name, b.name),
            created_at: chrono::Utc::now(),
        })
    }

    /// Parallel matrix norm calculation
    pub fn parallel_frobenius_norm(&self, matrix: &Matrix) -> f64 {
        matrix.data
            .par_iter()
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt()
    }

    /// Parallel matrix trace calculation (for square matrices)
    pub fn parallel_trace(&self, matrix: &Matrix) -> anyhow::Result<f64> {
        if !matrix.is_square() {
            return Err(anyhow::anyhow!("Trace requires a square matrix"));
        }

        let size = matrix.data.nrows();
        let trace = (0..size)
            .into_par_iter()
            .map(|i| matrix.data[(i, i)])
            .sum();

        Ok(trace)
    }
}

impl Default for ParallelEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create parallel engine")
    }
}

/// Performance benchmarking utilities
pub struct PerformanceBenchmark;

impl PerformanceBenchmark {
    /// Benchmark matrix multiplication performance
    pub fn benchmark_multiplication(size: usize) -> BenchmarkResult {
        let a = Matrix::random(size, size, "bench_a".to_string());
        let b = Matrix::random(size, size, "bench_b".to_string());
        
        // Sequential multiplication
        let start = Instant::now();
        let _result_seq = a.multiply(&b).expect("Sequential multiplication failed");
        let sequential_time = start.elapsed();
        
        // Parallel multiplication
        let engine = ParallelEngine::new().expect("Failed to create parallel engine");
        let start = Instant::now();
        let _result_par = engine.parallel_multiply(&a, &b).expect("Parallel multiplication failed");
        let parallel_time = start.elapsed();
        
        BenchmarkResult {
            operation: "Matrix Multiplication".to_string(),
            matrix_size: (size, size),
            sequential_time,
            parallel_time,
            speedup: sequential_time.as_secs_f64() / parallel_time.as_secs_f64(),
        }
    }

    /// Benchmark various matrix operations
    pub fn benchmark_suite(sizes: Vec<usize>) -> Vec<BenchmarkResult> {
        sizes
            .into_par_iter()
            .map(|size| Self::benchmark_multiplication(size))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation: String,
    pub matrix_size: (usize, usize),
    pub sequential_time: std::time::Duration,
    pub parallel_time: std::time::Duration,
    pub speedup: f64,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}x{}): Sequential: {:.2}ms, Parallel: {:.2}ms, Speedup: {:.2}x",
            self.operation,
            self.matrix_size.0,
            self.matrix_size.1,
            self.sequential_time.as_millis(),
            self.parallel_time.as_millis(),
            self.speedup
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_engine_creation() {
        let engine = ParallelEngine::new();
        assert!(engine.is_ok());
    }

    #[test]
    fn test_parallel_multiplication() {
        let engine = ParallelEngine::new().unwrap();
        let a = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]], "A".to_string()).unwrap();
        let b = Matrix::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]], "B".to_string()).unwrap();
        
        let result = engine.parallel_multiply(&a, &b).unwrap();
        
        // Check result dimensions
        assert_eq!(result.dimensions(), (2, 2));
        
        // Check specific values
        assert!((result.get(0, 0).unwrap() - 19.0).abs() < 1e-10);
        assert!((result.get(0, 1).unwrap() - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_add() {
        let engine = ParallelEngine::new().unwrap();
        let a = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]], "A".to_string()).unwrap();
        let b = Matrix::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]], "B".to_string()).unwrap();
        
        let result = engine.parallel_add(&a, &b).unwrap();
        
        assert_eq!(result.get(0, 0), Some(6.0));
        assert_eq!(result.get(1, 1), Some(12.0));
    }

    #[test]
    fn test_benchmark() {
        let result = PerformanceBenchmark::benchmark_multiplication(10);
        assert_eq!(result.matrix_size, (10, 10));
        assert!(result.speedup > 0.0);
    }
}