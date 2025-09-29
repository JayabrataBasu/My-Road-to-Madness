pub mod compute_shaders;
pub mod gpu_context;

use crate::matrix::Matrix;
use anyhow::{Result, Context};
use wgpu::{Device, Queue, Buffer, ComputePipeline};
use bytemuck::{Pod, Zeroable};
use futures_intrusive::channel::shared::oneshot_channel;

/// GPU-accelerated matrix operations using compute shaders
pub struct GpuMatrixEngine {
    device: Device,
    queue: Queue,
    matrix_multiply_pipeline: Option<ComputePipeline>,
    matrix_add_pipeline: Option<ComputePipeline>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuMatrix {
    rows: u32,
    cols: u32,
    padding: [u32; 2], // Ensure 16-byte alignment
}

impl GpuMatrixEngine {
    /// Initialize the GPU matrix engine
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("Failed to find a suitable GPU adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .context("Failed to create GPU device")?;

        let mut engine = Self {
            device,
            queue,
            matrix_multiply_pipeline: None,
            matrix_add_pipeline: None,
        };

        engine.init_pipelines().await?;
        Ok(engine)
    }

    /// Initialize compute pipelines
    async fn init_pipelines(&mut self) -> Result<()> {
        // Matrix multiplication pipeline
        let multiply_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Multiply Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../assets/shaders/matrix_multiply.wgsl").into()),
        });

        let multiply_bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let multiply_pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matrix Multiply Pipeline Layout"),
            bind_group_layouts: &[&multiply_bind_group_layout],
            push_constant_ranges: &[],
        });

        self.matrix_multiply_pipeline = Some(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matrix Multiply Pipeline"),
            layout: Some(&multiply_pipeline_layout),
            module: &multiply_shader,
            entry_point: "main",
        }));

        Ok(())
    }

    /// GPU-accelerated matrix multiplication
    pub async fn gpu_multiply(&self, a: &Matrix, b: &Matrix) -> Result<Matrix> {
        if a.data.ncols() != b.data.nrows() {
            return Err(anyhow::anyhow!("Matrix dimensions don't match for multiplication"));
        }

        let pipeline = self.matrix_multiply_pipeline.as_ref()
            .context("Matrix multiply pipeline not initialized")?;

        let (m, k) = (a.data.nrows(), a.data.ncols());
        let n = b.data.ncols();

        // Convert matrices to GPU format
        let a_data: Vec<f32> = a.data.iter().map(|&x| x as f32).collect();
        let b_data: Vec<f32> = b.data.iter().map(|&x| x as f32).collect();

        // Create GPU buffers
        let a_buffer = self.create_storage_buffer(&a_data, "Matrix A Buffer");
        let b_buffer = self.create_storage_buffer(&b_data, "Matrix B Buffer");
        
        let result_size = m * n * std::mem::size_of::<f32>();
        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Buffer"),
            size: result_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Dimensions uniform buffer
        let dimensions = GpuMatrix {
            rows: m as u32,
            cols: k as u32,
            padding: [n as u32, 0],
        };
        
        let dimensions_data = bytemuck::cast_slice(&[dimensions]);
        let dimensions_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dimensions Buffer"),
            size: dimensions_data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        dimensions_buffer.slice(..).get_mapped_range_mut().copy_from_slice(dimensions_data);
        dimensions_buffer.unmap();

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matrix Multiply Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dimensions_buffer.as_entire_binding(),
                },
            ],
        });

        // Create staging buffer for result
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: result_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Execute compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Matrix Multiply Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matrix Multiply Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch with appropriate workgroup size
            let workgroup_size = 16; // Common GPU workgroup size
            let dispatch_x = (n as u32 + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (m as u32 + workgroup_size - 1) / workgroup_size;
            
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, result_size as u64);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read result back from GPU
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        receiver.receive().await.unwrap()?;

        let data = buffer_slice.get_mapped_range();
        let result_f32: &[f32] = bytemuck::cast_slice(&data);
        let result_f64: Vec<f64> = result_f32.iter().map(|&x| x as f64).collect();

        drop(data);
        staging_buffer.unmap();

        // Convert back to Matrix
        let result_matrix = nalgebra::DMatrix::from_vec(m, n, result_f64);
        
        Ok(Matrix {
            data: result_matrix,
            name: format!("{}*{}_gpu", a.name, b.name),
            created_at: chrono::Utc::now(),
        })
    }

    /// Create a storage buffer from data
    fn create_storage_buffer<T: Pod>(&self, data: &[T], label: &str) -> Buffer {
        let contents = bytemuck::cast_slice(data);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: contents.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer.slice(..).get_mapped_range_mut().copy_from_slice(contents);
        buffer.unmap();
        buffer
    }

    /// Check if GPU acceleration is available
    pub fn is_available(&self) -> bool {
        self.matrix_multiply_pipeline.is_some()
    }

    /// Get GPU device info
    pub fn device_info(&self) -> GpuDeviceInfo {
        let adapter_info = self.device.limits();
        
        GpuDeviceInfo {
            max_compute_workgroup_size_x: adapter_info.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: adapter_info.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: adapter_info.max_compute_workgroup_size_z,
            max_compute_workgroups_per_dimension: adapter_info.max_compute_workgroups_per_dimension,
            max_buffer_size: adapter_info.max_buffer_size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_buffer_size: u64,
}

/// Determine if a matrix operation should use GPU acceleration
pub fn should_use_gpu(rows: usize, cols: usize, other_cols: Option<usize>) -> bool {
    let total_elements = rows * cols * other_cols.unwrap_or(1);
    
    // Use GPU for operations involving more than 100,000 elements
    total_elements > 100_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_engine_creation() {
        // This test might fail if no GPU is available
        match GpuMatrixEngine::new().await {
            Ok(engine) => {
                assert!(engine.is_available());
                let info = engine.device_info();
                assert!(info.max_buffer_size > 0);
            }
            Err(_) => {
                // GPU not available, which is fine for testing
                println!("GPU not available for testing");
            }
        }
    }

    #[test]
    fn test_should_use_gpu() {
        assert!(!should_use_gpu(10, 10, Some(10))); // Small matrix
        assert!(should_use_gpu(100, 100, Some(100))); // Large matrix
        assert!(should_use_gpu(500, 500, None)); // Large single matrix
    }
}