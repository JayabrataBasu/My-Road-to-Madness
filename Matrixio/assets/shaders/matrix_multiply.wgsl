// Matrix multiplication compute shader for GPU acceleration
// Input: Two matrices A (m x k) and B (k x n)
// Output: Matrix C (m x n) where C = A * B

@group(0) @binding(0)
var<storage, read> matrix_a: array<f32>;

@group(0) @binding(1)
var<storage, read> matrix_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> matrix_c: array<f32>;

struct MatrixDimensions {
    m: u32,  // rows of A and C
    k: u32,  // cols of A and rows of B
    n: u32,  // cols of B and C
    padding: u32,
}

@group(0) @binding(3)
var<uniform> dims: MatrixDimensions;

// Workgroup size - optimized for most modern GPUs
const WORKGROUP_SIZE: u32 = 16u;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    
    // Bounds check
    if (row >= dims.m || col >= dims.n) {
        return;
    }
    
    var sum = 0.0f;
    
    // Compute dot product of row from A and column from B
    for (var k = 0u; k < dims.k; k++) {
        let a_index = row * dims.k + k;
        let b_index = k * dims.n + col;
        sum += matrix_a[a_index] * matrix_b[b_index];
    }
    
    // Store result
    let c_index = row * dims.n + col;
    matrix_c[c_index] = sum;
}

// Alternative tiled matrix multiplication for better cache efficiency
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn main_tiled(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;
    
    // Shared memory tiles
    var tile_a: array<array<f32, WORKGROUP_SIZE>, WORKGROUP_SIZE>;
    var tile_b: array<array<f32, WORKGROUP_SIZE>, WORKGROUP_SIZE>;
    
    var sum = 0.0f;
    
    // Number of tiles needed
    let num_tiles = (dims.k + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    
    for (var tile = 0u; tile < num_tiles; tile++) {
        // Load tiles into shared memory
        let a_row = row;
        let a_col = tile * WORKGROUP_SIZE + local_col;
        let b_row = tile * WORKGROUP_SIZE + local_row;
        let b_col = col;
        
        // Load tile A
        if (a_row < dims.m && a_col < dims.k) {
            tile_a[local_row][local_col] = matrix_a[a_row * dims.k + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0f;
        }
        
        // Load tile B
        if (b_row < dims.k && b_col < dims.n) {
            tile_b[local_row][local_col] = matrix_b[b_row * dims.n + b_col];
        } else {
            tile_b[local_row][local_col] = 0.0f;
        }
        
        // Synchronize workgroup
        workgroupBarrier();
        
        // Compute partial dot product using tiles
        for (var k = 0u; k < WORKGROUP_SIZE; k++) {
            sum += tile_a[local_row][k] * tile_b[k][local_col];
        }
        
        // Synchronize before loading next tile
        workgroupBarrier();
    }
    
    // Store result
    if (row < dims.m && col < dims.n) {
        let c_index = row * dims.n + col;
        matrix_c[c_index] = sum;
    }
}