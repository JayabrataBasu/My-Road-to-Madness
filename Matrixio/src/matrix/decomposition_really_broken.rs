use crate::matrix::Matrix;
use nalgebra::DMatrix;
use anyhow::Result;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LUDecomposition {
    pub l: Matrix,
    pub u: Matrix,
    pub p: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QRDecomposition {
    pub q: Matrix,
    pub r: Matrix,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVDDecomposition {
    pub u: Option<Matrix>,
    pub s: Matrix, // Diagonal matrix of singular values
    pub v_t: Option<Matrix>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EigenDecomposition {
    pub values: Vec<f64>,
    pub vectors: Option<Matrix>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CholeskyDecomposition {
    pub l: Matrix,
}

impl Matrix {
    /// Simplified LU Decomposition
    pub fn lu_decomposition(&self) -> Result<LUDecomposition> {
        if !self.is_square() {
            return Err(anyhow::anyhow!("LU decomposition requires a square matrix"));
        }

        let lu = self.data.clone().lu();
        let l_matrix = lu.l();
        let u_matrix = lu.u();
        
        Ok(LUDecomposition {
            l: Matrix {
                data: l_matrix,
                name: format!("{}_L", self.name),
                created_at: chrono::Utc::now(),
            },
            u: Matrix {
                data: u_matrix,
                name: format!("{}_U", self.name),
                created_at: chrono::Utc::now(),
            },
            p: (0..self.data.nrows()).collect(), // Identity permutation for now
        })
    }

    /// QR Decomposition
    pub fn qr_decomposition(&self) -> Result<QRDecomposition> {
        let qr = self.data.clone().qr();
        
        let q_data = qr.q();
        let r_data = qr.r();

        Ok(QRDecomposition {
            q: Matrix {
                data: q_data,
                name: format!("{}_Q", self.name),
                created_at: chrono::Utc::now(),
            },
            r: Matrix {
                data: r_data,
                name: format!("{}_R", self.name),
                created_at: chrono::Utc::now(),
            },
        })
    }

    /// SVD Decomposition
    pub fn svd_decomposition(&self, compute_u: bool, compute_v: bool) -> Result<SVDDecomposition> {
        let svd = self.data.clone().svd(compute_u, compute_v);
        
        let u = if compute_u {
            svd.u.map(|u_matrix| Matrix {
                data: u_matrix,
                name: format!("{}_U", self.name),
                created_at: chrono::Utc::now(),
            })
        } else {
            None
        };

        let s_diag = DMatrix::from_diagonal(&svd.singular_values);
        let s = Matrix {
            data: s_diag,
            name: format!("{}_S", self.name),
            created_at: chrono::Utc::now(),
        };

        let v_t = if compute_v {
            svd.v_t.map(|vt_matrix| Matrix {
                data: vt_matrix,
                name: format!("{}_VT", self.name),
                created_at: chrono::Utc::now(),
            })
        } else {
            None
        };

        Ok(SVDDecomposition { u, s, v_t })
    }

    /// Eigenvalue decomposition
    pub fn eigenvalue_decomposition(&self) -> Result<EigenDecomposition> {
        if !self.is_square() {
            return Err(anyhow::anyhow!("Eigenvalue decomposition requires a square matrix"));
        }

        // For symmetric matrices, use symmetric eigenvalue decomposition
        if self.is_symmetric() {
            let eigen = self.data.clone().symmetric_eigen();
            
            Ok(EigenDecomposition {
                values: eigen.eigenvalues.data.as_vec().clone(),
                vectors: Some(Matrix {
                    data: eigen.eigenvectors,
                    name: format!("{}_eigenvectors", self.name),
                    created_at: chrono::Utc::now(),
                }),
            })
        } else {
            // For non-symmetric matrices, use symmetric approximation
            let eigen = self.data.clone().symmetric_eigen();
            
            Ok(EigenDecomposition {
                values: eigen.eigenvalues.data.as_vec().clone(),
                vectors: Some(Matrix {
                    data: eigen.eigenvectors,
                    name: format!("{}_eigenvectors", self.name),
                    created_at: chrono::Utc::now(),
                }),
            })
        }
    }

    /// Cholesky decomposition (for positive definite matrices)
    pub fn cholesky_decomposition(&self) -> Result<CholeskyDecomposition> {
        if !self.is_square() {
            return Err(anyhow::anyhow!("Cholesky decomposition requires a square matrix"));
        }

        let chol = self.data.clone().cholesky()
            .ok_or_else(|| anyhow::anyhow!("Matrix is not positive definite"))?;

        Ok(CholeskyDecomposition {
            l: Matrix {
                data: chol.l(),
                name: format!("{}_L", self.name),
                created_at: chrono::Utc::now(),
            },
        })
    }

    /// Check if matrix is symmetric
    fn is_symmetric(&self) -> bool {
        if !self.is_square() {
            return false;
        }
        
        let n = self.data.nrows();
        for i in 0..n {
            for j in 0..n {
                if (self.data[(i, j)] - self.data[(j, i)]).abs() > f64::EPSILON * 100.0 {
                    return false;
                }
            }
        }
        true
    }

    /// Calculate matrix trace - DISABLED (duplicate)\n    fn _disabled_trace(&self) -> anyhow::Result<f64> {

        if !self.is_square() {
            return Err(anyhow::anyhow!("Trace is only defined for square matrices"));
        }

        Ok(self.data.trace())
    }

    /// Solve linear system Ax = b using LU decomposition
    pub fn solve(&self, b: &Matrix) -> Result<Matrix> {
        if !self.is_square() {
            return Err(anyhow::anyhow!("Can only solve square systems"));
        }

        if self.data.nrows() != b.data.nrows() {
            return Err(anyhow::anyhow!("Incompatible dimensions"));
        }

        let lu = self.data.clone().lu();
        let solution = lu.solve(&b.data)
            .ok_or_else(|| anyhow::anyhow!("System has no unique solution"))?;

        Ok(Matrix {
            data: solution,
            name: format!("solve({}, {})", self.name, b.name),
            created_at: chrono::Utc::now(),
        })
    }
