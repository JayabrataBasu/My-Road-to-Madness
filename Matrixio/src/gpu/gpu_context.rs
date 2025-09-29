use anyhow::{Result, Context};
use wgpu::{Device, Queue, Instance, Adapter, Surface};

/// GPU context management for matrix operations
pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl GpuContext {
    /// Initialize GPU context
    pub async fn new() -> Result<Self> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
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
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("Failed to create GPU device")?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Get adapter information
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// Get device limits
    pub fn device_limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// Check if GPU compute is supported
    pub fn supports_compute(&self) -> bool {
        self.device.features().contains(wgpu::Features::COMPUTE)
    }

    /// Get maximum buffer size
    pub fn max_buffer_size(&self) -> u64 {
        self.device.limits().max_buffer_size
    }

    /// Get maximum workgroup size
    pub fn max_workgroup_size(&self) -> (u32, u32, u32) {
        let limits = self.device.limits();
        (
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z,
        )
    }
}

/// GPU capability information
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub vendor: String,
    pub device_name: String,
    pub device_type: String,
    pub backend: String,
    pub supports_compute: bool,
    pub max_buffer_size: u64,
    pub max_workgroup_size: (u32, u32, u32),
    pub max_workgroups_per_dimension: u32,
}

impl From<&GpuContext> for GpuCapabilities {
    fn from(context: &GpuContext) -> Self {
        let info = context.adapter_info();
        let limits = context.device_limits();
        
        Self {
            vendor: info.vendor.to_string(),
            device_name: info.name.clone(),
            device_type: format!("{:?}", info.device_type),
            backend: format!("{:?}", info.backend),
            supports_compute: context.supports_compute(),
            max_buffer_size: limits.max_buffer_size,
            max_workgroup_size: (
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ),
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }
}

impl std::fmt::Display for GpuCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GPU Capabilities:")?;
        writeln!(f, "  Vendor: {}", self.vendor)?;
        writeln!(f, "  Device: {}", self.device_name)?;
        writeln!(f, "  Type: {}", self.device_type)?;
        writeln!(f, "  Backend: {}", self.backend)?;
        writeln!(f, "  Compute Support: {}", self.supports_compute)?;
        writeln!(f, "  Max Buffer Size: {} MB", self.max_buffer_size / (1024 * 1024))?;
        writeln!(f, "  Max Workgroup Size: {:?}", self.max_workgroup_size)?;
        writeln!(f, "  Max Workgroups/Dim: {}", self.max_workgroups_per_dimension)?;
        Ok(())
    }
}