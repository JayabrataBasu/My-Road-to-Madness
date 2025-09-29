// Matrix addition compute shader for GPU acceleration
// Input: Two matrices A and B of the same dimensions
// Output: Matrix C where C = A + B

@group(0) @binding(0)
var<storage, read> matrix_a: array<f32>;

@group(0) @binding(1)
var<storage, read> matrix_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> matrix_c: array<f32>;

struct MatrixDimensions {
    rows: u32,
    cols: u32,
    total_elements: u32,
    padding: u32,
}

@group(0) @binding(3)
var<uniform> dims: MatrixDimensions;

// Use 1D workgroup for element-wise operations
const WORKGROUP_SIZE_1D: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE_1D)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Bounds check
    if (index >= dims.total_elements) {
        return;
    }
    
    // Element-wise addition
    matrix_c[index] = matrix_a[index] + matrix_b[index];
}

// Alternative 2D version for better visualization
@compute @workgroup_size(16u, 16u)
fn main_2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    
    // Bounds check
    if (row >= dims.rows || col >= dims.cols) {
        return;
    }
    
    let index = row * dims.cols + col;
    matrix_c[index] = matrix_a[index] + matrix_b[index];
}