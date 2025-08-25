use std::{collections::HashMap, sync::Arc, path::Path};
use anyhow::Result;

pub struct ShaderManager {
    device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    shader_modules: HashMap<String, wgpu::ShaderModule>,
}

impl ShaderManager {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self { device, queue, shader_modules: HashMap::new() }
    }

    pub fn load_wgsl_str(&mut self, name:&str, source:&str) -> Result<()> {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(source.into())
        });
        self.shader_modules.insert(name.to_string(), module);
        Ok(())
    }

    pub fn load_wgsl_file(&mut self, name:&str, path:&Path) -> Result<()> {
        let src = std::fs::read_to_string(path)?;
        self.load_wgsl_str(name, &src)
    }

    pub fn get(&self, name:&str) -> Option<&wgpu::ShaderModule> { self.shader_modules.get(name) }
}
