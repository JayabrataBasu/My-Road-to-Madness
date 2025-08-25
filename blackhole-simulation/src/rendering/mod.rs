//! Rendering module for 3D graphics and visualization
//!
//! This module handles all aspects of rendering the black hole simulation,
//! including the graphics pipeline, ray tracing, camera system, and shaders.

pub mod camera;
pub mod ray_tracer;
pub mod renderer;
pub mod shaders;

pub use renderer::{RenderQuality, Renderer};
