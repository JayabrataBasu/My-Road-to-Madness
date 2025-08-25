//! Rendering module for 3D graphics and visualization
//!
//! This module handles all aspects of rendering the black hole simulation,
//! including the graphics pipeline, ray tracing, camera system, and shaders.

#![allow(unused_imports)] // Re-exports may be unused internally during partial implementation

pub mod camera;
pub mod ray_tracer;
pub mod renderer;
pub mod shaders;

// Re-export commonly used items
pub use camera::Camera;
pub use renderer::{Renderer, RenderQuality};
pub use ray_tracer::{RayTracingQuality, RayTracer};

use bytemuck;
use wgpu;

/// Common vertex type for 3D rendering
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl Vertex {
    /// Create a new vertex
    pub fn new(position: [f32; 3], normal: [f32; 3], tex_coords: [f32; 2]) -> Self {
        Self {
            position,
            normal,
            tex_coords,
        }
    }

    /// Get the vertex buffer layout for wgpu
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Texture coordinates
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Uniform data passed to shaders
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub time: f32,
    pub black_hole_mass: f32,
    pub black_hole_pos: [f32; 3],
}

impl Uniforms {
    pub fn new() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_pos: [0.0; 3],
            time: 0.0,
            black_hole_mass: 1.0, // Default mass
            black_hole_pos: [0.0; 3],
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
    let vp = camera.view_projection();
    self.view_proj = vp.to_cols_array_2d();
    self.camera_pos = camera.position.to_array();
    }
}

// TODO: Uniforms / Vertex structures are legacy placeholders from earlier pipeline sketches.
// They will be reconciled or removed once the GPU path is implemented in a later Phase 2 step.

/// Rendering error types
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("wgpu error: {0}")]
    Wgpu(#[from] wgpu::Error),
    #[error("Surface error: {0}")]
    Surface(#[from] wgpu::SurfaceError),
    #[error("Shader compilation error: {0}")]
    Shader(String),
    #[error("Resource creation error: {0}")]
    Resource(String),
}

pub type RenderResult<T> = Result<T, RenderError>;
