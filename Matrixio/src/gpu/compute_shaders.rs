use wgpu::{ComputePipeline, Device, BindGroupLayout, PipelineLayout};

/// Collection of compute shaders for matrix operations
pub struct ComputeShaders {
    pub matrix_multiply: ComputePipeline,
    pub matrix_add: ComputePipeline,
    pub matrix_transpose: ComputePipeline,
    pub matrix_elementwise: ComputePipeline,
}

impl ComputeShaders {
    /// Initialize all compute shaders
    pub fn new(device: &Device) -> Self {
        let matrix_multiply = Self::create_matrix_multiply_pipeline(device);
        let matrix_add = Self::create_matrix_add_pipeline(device);
        let matrix_transpose = Self::create_matrix_transpose_pipeline(device);
        let matrix_elementwise = Self::create_matrix_elementwise_pipeline(device);

        Self {
            matrix_multiply,
            matrix_add,
            matrix_transpose,
            matrix_elementwise,
        }
    }

    /// Create matrix multiplication compute pipeline
    fn create_matrix_multiply_pipeline(device: &Device) -> ComputePipeline {
        let shader_source = include_str!("../../assets/shaders/matrix_multiply.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Multiply Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matrix Multiply Bind Group Layout"),
            entries: &[
                // Matrix A
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Matrix B
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Result Matrix C
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Matrix dimensions
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matrix Multiply Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matrix Multiply Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    /// Create matrix addition compute pipeline
    fn create_matrix_add_pipeline(device: &Device) -> ComputePipeline {
        let shader_source = include_str!("../../assets/shaders/matrix_add.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Add Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matrix Add Bind Group Layout"),
            entries: &[
                // Matrix A
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Matrix B
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Result Matrix C
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Matrix dimensions
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matrix Add Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matrix Add Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    /// Create matrix transpose compute pipeline
    fn create_matrix_transpose_pipeline(device: &Device) -> ComputePipeline {
        let shader_source = r#"
@group(0) @binding(0)
var<storage, read> input_matrix: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_matrix: array<f32>;

struct MatrixDimensions {
    rows: u32,
    cols: u32,
    padding: vec2<u32>,
}

@group(0) @binding(2)
var<uniform> dims: MatrixDimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    
    if (row >= dims.rows || col >= dims.cols) {
        return;
    }
    
    let input_index = row * dims.cols + col;
    let output_index = col * dims.rows + row;
    
    output_matrix[output_index] = input_matrix[input_index];
}
"#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Transpose Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matrix Transpose Bind Group Layout"),
            entries: &[
                // Input matrix
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output matrix
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Matrix dimensions
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matrix Transpose Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matrix Transpose Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    /// Create matrix element-wise operations compute pipeline
    fn create_matrix_elementwise_pipeline(device: &Device) -> ComputePipeline {
        let shader_source = r#"
@group(0) @binding(0)
var<storage, read> input_matrix: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_matrix: array<f32>;

struct OpParams {
    total_elements: u32,
    operation: u32,  // 0=exp, 1=ln, 2=sin, 3=cos, 4=sqrt, 5=abs, etc.
    scalar_param: f32,
    padding: u32,
}

@group(0) @binding(2)
var<uniform> params: OpParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= params.total_elements) {
        return;
    }
    
    let input_value = input_matrix[index];
    var result: f32;
    
    switch (params.operation) {
        case 0u: { result = exp(input_value); }
        case 1u: { result = log(input_value); }
        case 2u: { result = sin(input_value); }
        case 3u: { result = cos(input_value); }
        case 4u: { result = sqrt(input_value); }
        case 5u: { result = abs(input_value); }
        case 6u: { result = input_value * params.scalar_param; }
        case 7u: { result = pow(input_value, params.scalar_param); }
        default: { result = input_value; }
    }
    
    output_matrix[index] = result;
}
"#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Element-wise Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matrix Element-wise Bind Group Layout"),
            entries: &[
                // Input matrix
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output matrix
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Operation parameters
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matrix Element-wise Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matrix Element-wise Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }
}

/// Available element-wise operations for GPU compute
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum ElementwiseOperation {
    Exp = 0,
    Ln = 1,
    Sin = 2,
    Cos = 3,
    Sqrt = 4,
    Abs = 5,
    ScalarMultiply = 6,
    Power = 7,
}