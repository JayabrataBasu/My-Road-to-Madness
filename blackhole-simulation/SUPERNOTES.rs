# Black Hole Simulation - Authoritative Code Architecture Reference

## 🎯 **MASTER DEPENDENCY GRAPH**

```
PHYSICS LAYER (No external dependencies except std/nalgebra)
    constants.rs → [ALL MODULES]
    black_hole.rs → spacetime.rs, geodesics.rs
    spacetime.rs → geodesics.rs
    geodesics.rs → [RENDERING LAYER]

RENDERING LAYER (Depends on PHYSICS for calculations)
    renderer.rs → camera.rs, ray_tracer.rs, shaders.rs
    camera.rs → [SIMULATION LAYER]
    ray_tracer.rs → geodesics.rs, spacetime.rs
    shaders.rs → renderer.rs

SIMULATION LAYER (Orchestrates both layers)
    scene.rs → [ALL MODULES]
    objects.rs → black_hole.rs, geodesics.rs, animation.rs
    animation.rs → geodesics.rs, objects.rs
```

---

## 📁 **src/physics/** - DETAILED SPECIFICATIONS

### **constants.rs**
**EXPORTS** (Public constants used project-wide):
```rust
// Physical Constants
pub const G: f64
pub const C: f64  
pub const SOLAR_MASS: f64
pub const PLANCK_LENGTH: f64
pub const STEFAN_BOLTZMANN: f64

// Black Hole Constants
pub const SCHWARZSCHILD_COEFF: f64
pub const PHOTON_SPHERE_RADIUS: f64
pub const ISCO_RADIUS: f64
pub const CRITICAL_IMPACT_PARAMETER: f64

// Numerical Constants
pub const INTEGRATION_TOLERANCE: f64
pub const MAX_INTEGRATION_STEPS: usize
pub const MIN_TIME_STEP: f64
pub const MAX_TIME_STEP: f64

// Rendering Constants
pub const DEFAULT_CAMERA_DISTANCE: f64
pub const RENDER_QUALITY_LEVELS: [usize; 5]
pub const RAY_MARCHING_STEP_SIZE: f64
```

**IMPORTED BY**: Every single file in the project
**NO IMPORTS**: Only std library

---

### **black_hole.rs**
**PURPOSE**: Core black hole physics and properties

**IMPORTS**:
```rust
use crate::physics::constants::*;
use nalgebra::{Vector3, Vector4};
use std::f64::consts::PI;
```

**EXPORTS** (Public API):
```rust
pub struct BlackHole {
    mass_solar: f64,
    spin: f64,           // a/M dimensionless (0.0 to 0.998)
    charge: f64,         // Q/M dimensionless
    position: Vector3<f64>,
    accretion_rate: f64, // Eddington units
}

// Constructor Methods
impl BlackHole {
    pub fn new_schwarzschild(mass_solar: f64) -> Self
    pub fn new_kerr(mass_solar: f64, spin: f64) -> Self
    pub fn new_reissner_nordstrom(mass_solar: f64, charge: f64) -> Self
    pub fn stellar_mass(mass_solar: f64) -> Self      // 3-50 solar masses
    pub fn intermediate_mass(mass_solar: f64) -> Self // 100-100,000 solar masses
    pub fn supermassive(mass_solar: f64) -> Self      // 10^6-10^10 solar masses
}

// Getter Methods (Used by other modules)
impl BlackHole {
    pub fn mass(&self) -> f64                    // USED BY: spacetime.rs, objects.rs
    pub fn mass_kg(&self) -> f64                 // USED BY: geodesics.rs calculations
    pub fn spin(&self) -> f64                    // USED BY: spacetime.rs (Kerr metric)
    pub fn position(&self) -> Vector3<f64>       // USED BY: renderer.rs, ray_tracer.rs
    pub fn accretion_rate(&self) -> f64          // USED BY: objects.rs (disk brightness)
}

// Calculated Properties (Complex physics - internal methods call these)
impl BlackHole {
    pub fn schwarzschild_radius(&self) -> f64       // USED BY: spacetime.rs, renderer.rs
    pub fn photon_sphere_radius(&self) -> f64       // USED BY: geodesics.rs
    pub fn ergosphere_outer_radius(&self) -> f64    // USED BY: spacetime.rs (Kerr)
    pub fn isco_radius(&self) -> f64               // USED BY: objects.rs (accretion disk)
    pub fn surface_gravity(&self) -> f64           // USED BY: objects.rs (temperature)
    pub fn hawking_temperature(&self) -> f64       // USED BY: objects.rs (thermal radiation)
    pub fn luminosity_eddington(&self) -> f64      // USED BY: objects.rs (disk luminosity)
}

// Tidal Effects (Advanced physics)
impl BlackHole {
    pub fn tidal_acceleration(&self, position: Vector3<f64>, velocity: Vector3<f64>) -> Vector3<f64>
    // USED BY: geodesics.rs for accurate particle dynamics
}
```

**INTERNAL METHODS** (Private, complex calculations):
```rust
fn calculate_horizon_area(&self) -> f64
fn calculate_angular_momentum(&self) -> f64
fn validate_parameters(&self) -> Result<(), PhysicsError>
```

**USED BY**: 
- `spacetime.rs` → metric tensor calculations
- `geodesics.rs` → trajectory calculations
- `objects.rs` → visual scaling and physics
- `renderer.rs` → uniform buffer updates
- `scene.rs` → simulation state management

---

### **spacetime.rs**
**PURPOSE**: Spacetime geometry, metrics, coordinate transformations

**IMPORTS**:
```rust
use crate::physics::{constants::*, black_hole::BlackHole};
use nalgebra::{Vector3, Vector4, Matrix4, SMatrix};
```

**EXPORTS**:
```rust
// Core Metric Types
pub struct MetricTensor {
    components: SMatrix<f64, 4, 4>,  // 4x4 spacetime metric
    coordinate_system: CoordinateSystem,
}

pub enum CoordinateSystem {
    Cartesian,
    Schwarzschild,      // (t, r, θ, φ)
    BoyerLindquist,     // For Kerr black holes
    IsotropicCartesian, // Isotropic coordinates
}

// Coordinate Transformation Functions
pub fn cartesian_to_schwarzschild(position: Vector3<f64>) -> Vector3<f64>
pub fn schwarzschild_to_cartesian(r: f64, theta: f64, phi: f64) -> Vector3<f64>
pub fn cartesian_to_boyer_lindquist(position: Vector3<f64>, black_hole: &BlackHole) -> Vector4<f64>
// USED BY: geodesics.rs, ray_tracer.rs

// Metric Tensor Calculations
impl MetricTensor {
    pub fn schwarzschild(black_hole: &BlackHole, position: Vector4<f64>) -> Self
    pub fn kerr(black_hole: &BlackHole, position: Vector4<f64>) -> Self
    pub fn reissner_nordstrom(black_hole: &BlackHole, position: Vector4<f64>) -> Self
    // USED BY: geodesics.rs for trajectory integration
    
    pub fn determinant(&self) -> f64             // USED BY: geodesics.rs
    pub fn inverse(&self) -> MetricTensor        // USED BY: geodesics.rs
    pub fn christoffel_symbols(&self) -> ChristoffelTensor  // USED BY: geodesics.rs
}

// Spacetime Geometry Calculations
pub fn spacetime_interval(
    metric: &MetricTensor, 
    four_velocity: Vector4<f64>
) -> f64  // USED BY: geodesics.rs

pub fn proper_time_factor(
    black_hole: &BlackHole, 
    position: Vector3<f64>
) -> f64  // USED BY: animation.rs, objects.rs

pub fn gravitational_redshift(
    black_hole: &BlackHole, 
    observer_pos: Vector3<f64>, 
    source_pos: Vector3<f64>
) -> f64  // USED BY: ray_tracer.rs, objects.rs

// Curvature Calculations
pub struct RiemannTensor {
    components: [[[f64; 4]; 4]; 4],
}

impl RiemannTensor {
    pub fn from_metric(metric: &MetricTensor) -> Self
    pub fn ricci_scalar(&self) -> f64           // USED BY: renderer.rs for debug visualization
    pub fn weyl_tensor(&self) -> WeylTensor     // USED BY: advanced gravitational wave effects
}
```

**INTERNAL COMPLEX CALCULATIONS**:
```rust
fn compute_christoffel_symbol(metric: &MetricTensor, i: usize, j: usize, k: usize) -> f64
fn numerical_derivative_metric(metric: &MetricTensor, direction: usize) -> MetricTensor
fn validate_metric_signature(metric: &MetricTensor) -> Result<(), SpacetimeError>
```

**USED BY**:
- `geodesics.rs` → ALL trajectory calculations
- `ray_tracer.rs` → light bending calculations
- `objects.rs` → proper time effects on accretion disk

---

### **geodesics.rs** 
**PURPOSE**: Particle and light trajectories in curved spacetime

**IMPORTS**:
```rust
use crate::physics::{
    constants::*, 
    black_hole::BlackHole, 
    spacetime::{MetricTensor, CoordinateSystem, spacetime_interval}
};
use nalgebra::{Vector3, Vector4, DVector};
use rayon::prelude::*;  // For parallel ray tracing
```

**EXPORTS**:
```rust
// Core Trajectory Types
pub struct Geodesic {
    pub position: Vector4<f64>,     // (t, x, y, z) or (t, r, θ, φ)
    pub four_velocity: Vector4<f64>, // dx^μ/dτ
    pub proper_time: f64,
    pub coordinate_system: CoordinateSystem,
}

pub enum ParticleType {
    Photon,           // Massless, null geodesics
    MassiveParticle,  // Timelike geodesics
    Tachyon,         // Spacelike geodesics (theoretical)
}

// Integration Methods
pub enum IntegrationMethod {
    RungeKutta4,      // Standard 4th order
    DormandPrince,    // Adaptive step size
    Verlet,          // Symplectic for stable orbits
    LeapFrog,        // Alternative symplectic
}

// Main Geodesic Integrator
pub struct GeodesicIntegrator {
    method: IntegrationMethod,
    tolerance: f64,
    max_steps: usize,
    min_step_size: f64,
}

impl GeodesicIntegrator {
    pub fn new(method: IntegrationMethod) -> Self
    
    // PRIMARY FUNCTIONS (Called by ray_tracer.rs millions of times)
    pub fn integrate_photon_geodesic(
        &self,
        black_hole: &BlackHole,
        initial_position: Vector4<f64>,
        initial_direction: Vector3<f64>,
        max_coordinate_time: f64
    ) -> Vec<Vector4<f64>>
    // USED BY: ray_tracer.rs for every ray
    
    pub fn integrate_particle_orbit(
        &self,
        black_hole: &BlackHole,
        initial_position: Vector4<f64>,
        initial_velocity: Vector4<f64>,
        particle_mass: f64,
        orbit_time: f64
    ) -> Vec<Vector4<f64>>
    // USED BY: objects.rs for accretion disk particles
    
    // UTILITY FUNCTIONS
    pub fn find_photon_sphere_orbit(black_hole: &BlackHole) -> Option<Geodesic>
    // USED BY: objects.rs for unstable photon orbits
    
    pub fn calculate_orbital_frequency(
        black_hole: &BlackHole, 
        radius: f64
    ) -> f64
    // USED BY: animation.rs, objects.rs
    
    pub fn deflection_angle(
        black_hole: &BlackHole,
        impact_parameter: f64
    ) -> f64
    // USED BY: ray_tracer.rs for gravitational lensing
    
    pub fn critical_impact_parameter(black_hole: &BlackHole) -> f64
    // USED BY: ray_tracer.rs to determine if ray escapes
}

// Parallel Processing Support
impl GeodesicIntegrator {
    pub fn integrate_ray_batch(
        &self,
        black_hole: &BlackHole,
        ray_origins: &[Vector4<f64>],
        ray_directions: &[Vector3<f64>]
    ) -> Vec<Vec<Vector4<f64>>>
    // USED BY: ray_tracer.rs for performance
}

// Specialized Orbit Calculations
pub fn calculate_isco_orbit(black_hole: &BlackHole) -> Geodesic
pub fn calculate_photon_sphere_orbit(black_hole: &BlackHole) -> Geodesic
pub fn calculate_marginally_bound_orbit(black_hole: &BlackHole) -> Geodesic
// ALL USED BY: objects.rs for special orbital features

// Energy and Angular Momentum
pub fn orbital_energy(
    black_hole: &BlackHole, 
    position: Vector4<f64>, 
    velocity: Vector4<f64>
) -> f64  // USED BY: objects.rs

pub fn orbital_angular_momentum(
    black_hole: &BlackHole, 
    position: Vector4<f64>, 
    velocity: Vector4<f64>
) -> f64  // USED BY: objects.rs
```

**INTERNAL NUMERICAL METHODS** (Performance-critical):
```rust
fn runge_kutta_4_step(
    state: Vector4<f64>, 
    derivative: Vector4<f64>, 
    step_size: f64
) -> Vector4<f64>

fn adaptive_step_size(
    current_error: f64, 
    target_tolerance: f64, 
    current_step: f64
) -> f64

fn geodesic_equation(
    metric: &MetricTensor, 
    position: Vector4<f64>, 
    velocity: Vector4<f64>
) -> Vector4<f64>
```

**USED BY**:
- `ray_tracer.rs` → Every single light ray (PERFORMANCE CRITICAL)
- `objects.rs` → Particle motion in accretion disk
- `animation.rs` → Realistic orbital animations

---

## 🎨 **src/rendering/** - DETAILED SPECIFICATIONS

### **renderer.rs**
**PURPOSE**: Main graphics engine, GPU resource management

**IMPORTS**:
```rust
use crate::{
    physics::{constants::*, black_hole::BlackHole},
    rendering::{
        camera::{Camera, CameraController}, 
        ray_tracer::RayTracer,
        shaders::ShaderManager,
        Vertex, Uniforms, RenderError, RenderResult
    },
    simulation::Scene
};
use wgpu;
use winit::window::Window;
use bytemuck;
```

**EXPORTS**:
```rust
pub struct Renderer {
    // Core wgpu resources
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface,
    surface_config: wgpu::SurfaceConfiguration,
    
    // Rendering pipeline
    render_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    
    // Specialized renderers
    ray_tracer: RayTracer,
    shader_manager: ShaderManager,
    
    // Render state
    screen_size: (u32, u32),
    render_quality: RenderQuality,
}

pub enum RenderQuality {
    Ultra,    // Full resolution ray tracing
    High,     // 75% resolution with upsampling
    Medium,   // 50% resolution
    Low,      // 25% resolution for performance
    Potato,   // Rasterization only
}

impl Renderer {
    // INITIALIZATION (Called once from main.rs)
    pub async fn new(window: &Window) -> RenderResult<Self>
    // Sets up entire graphics pipeline
    
    // FRAME MANAGEMENT (Called every frame from main loop)
    pub fn render(&mut self, scene: &Scene) -> RenderResult<()>
    // CALLS: update_uniforms, ray_tracer.trace_frame, present_frame
    
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>)
    // Recreates surface and buffers
    
    // UNIFORM BUFFER MANAGEMENT (Called every frame)
    pub fn update_uniforms(
        &mut self, 
        camera: &Camera, 
        black_hole: &BlackHole, 
        time: f32
    ) -> RenderResult<()>
    // WRITES TO GPU: camera matrices, black hole data, time
    
    // QUALITY MANAGEMENT
    pub fn set_render_quality(&mut self, quality: RenderQuality)
    pub fn get_current_fps(&self) -> f32
    pub fn adaptive_quality_adjust(&mut self)  // Lower quality if FPS drops
}

// Internal Resource Management
impl Renderer {
    fn create_render_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        shader_manager: &ShaderManager
    ) -> RenderResult<wgpu::RenderPipeline>
    
    fn create_uniform_buffer(device: &wgpu::Device) -> wgpu::Buffer
    fn setup_bind_groups(device: &wgpu::Device, buffer: &wgpu::Buffer) -> wgpu::BindGroup
    fn handle_surface_error(&mut self, error: wgpu::SurfaceError) -> RenderResult<()>
}
```

**INTERNAL STATE MANAGEMENT**:
```rust
struct FrameMetrics {
    fps_counter: FpsCounter,
    frame_time: std::time::Duration,
    draw_calls: u32,
    vertices_rendered: u32,
}

struct RenderSettings {
    vsync: bool,
    msaa_samples: u32,
    anisotropic_filtering: u32,
    adaptive_quality: bool,
}
```

**USED BY**: Main loop in `main.rs`
**USES**: `camera.rs`, `ray_tracer.rs`, `shaders.rs`, `scene.rs`

---

### **camera.rs**
**PURPOSE**: 3D camera system with realistic physics-aware controls

**IMPORTS**:
```rust
use crate::physics::{constants::*, black_hole::BlackHole};
use crate::simulation::{InputState, TimeState};
use nalgebra::{Vector3, Matrix4, Point3, Perspective3};
use glam; // For GPU-compatible matrices
```

**EXPORTS**:
```rust
pub struct Camera {
    // Position and Orientation
    position: Vector3<f64>,      // World position
    target: Vector3<f64>,        // Look-at point
    up: Vector3<f64>,            // Up vector (usually Y)
    
    // Projection parameters
    fov_y: f32,                  // Field of view in radians
    aspect_ratio: f32,           // width/height
    near_plane: f32,
    far_plane: f32,
    
    // Computed matrices (cached)
    view_matrix: glam::Mat4,
    projection_matrix: glam::Mat4,
    view_projection_matrix: glam::Mat4,
    matrix_dirty: bool,
}

impl Camera {
    // CONSTRUCTORS
    pub fn new(window_width: f32, window_height: f32) -> Self
    pub fn new_orbital(black_hole: &BlackHole, distance: f64) -> Self
    // Positions camera at safe distance, looking at black hole
    
    // GETTERS (Used by renderer.rs for uniforms)
    pub fn position(&self) -> Vector3<f64>           // USED BY: renderer.rs uniforms
    pub fn view_matrix(&self) -> glam::Mat4          // USED BY: renderer.rs
    pub fn projection_matrix(&self) -> glam::Mat4    // USED BY: renderer.rs
    pub fn view_projection_matrix(&self) -> glam::Mat4 // USED BY: renderer.rs
    
    // SETTERS
    pub fn set_position(&mut self, position: Vector3<f64>)
    pub fn set_target(&mut self, target: Vector3<f64>)
    pub fn set_aspect_ratio(&mut self, aspect: f32)
    pub fn set_fov(&mut self, fov_radians: f32)
    
    // COORDINATE TRANSFORMATIONS (Used by ray_tracer.rs)
    pub fn screen_to_world_ray(
        &self, 
        screen_x: f32, 
        screen_y: f32, 
        screen_width: f32, 
        screen_height: f32
    ) -> (Vector3<f64>, Vector3<f64>)  // Returns (origin, direction)
    // USED BY: ray_tracer.rs for every pixel
    
    pub fn world_to_screen(&self, world_pos: Vector3<f64>) -> Option<(f32, f32)>
    // USED BY: debug rendering, UI overlays
}

// Camera Movement System
pub struct CameraController {
    // Movement parameters
    move_speed: f64,
    rotation_speed: f32,
    zoom_speed: f32,
    
    // Input state
    input: InputState,
    
    // Camera constraints (prevent falling into black hole)
    min_distance_from_bh: f64,
    max_distance_from_bh: f64,
    
    // Movement modes
    mode: CameraMode,
}

pub enum CameraMode {
    FreeCamera,       // Standard FPS-style camera
    OrbitCamera,      // Orbit around black hole
    FollowParticle,   // Track specific particle
    AutomaticTour,    // Predefined camera path
}

impl CameraController {
    pub fn new() -> Self
    
    // MAIN UPDATE (Called every frame from scene.rs)
    pub fn update(
        &mut self, 
        camera: &mut Camera, 
        input: &InputState, 
        time: &TimeState, 
        black_hole: &BlackHole
    )
    // Updates camera position based on input and physics constraints
    
    // INPUT HANDLING
    pub fn handle_keyboard(&mut self, input: &InputState)
    pub fn handle_mouse_movement(&mut self, delta_x: f32, delta_y: f32)
    pub fn handle_mouse_scroll(&mut self, scroll_delta: f32)
    
    // CAMERA MODES
    pub fn set_mode(&mut self, mode: CameraMode)
    pub fn start_automatic_tour(&mut self, waypoints: Vec<Vector3<f64>>)
}

// Physics-Aware Movement
impl CameraController {
    fn apply_physics_constraints(
        &self, 
        desired_position: Vector3<f64>, 
        black_hole: &BlackHole
    ) -> Vector3<f64>
    // Prevents camera from getting too close to event horizon
    
    fn calculate_orbital_position(
        &self, 
        black_hole: &BlackHole, 
        orbital_radius: f64, 
        orbital_angle: f64
    ) -> Vector3<f64>
    
    fn smooth_movement(
        current: Vector3<f64>, 
        target: Vector3<f64>, 
        delta_time: f32, 
        smoothing: f32
    ) -> Vector3<f64>
}
```

**INTERNAL CALCULATIONS**:
```rust
fn update_matrices(&mut self)  // Recalculates view/projection matrices
fn validate_orientation(&mut self)  // Prevents gimbal lock
fn apply_gravitational_effects(&self, black_hole: &BlackHole) -> f64  // Time dilation
```

**USED BY**: 
- `renderer.rs` → View/projection matrices, camera position
- `ray_tracer.rs` → Ray generation for every pixel
- `scene.rs` → Input handling and updates

---

### **ray_tracer.rs**
**PURPOSE**: CPU-based ray tracing through curved spacetime

**IMPORTS**:
```rust
use crate::{
    physics::{
        constants::*, 
        black_hole::BlackHole, 
        geodesics::{GeodesicIntegrator, IntegrationMethod},
        spacetime::{gravitational_redshift, proper_time_factor}
    },
    rendering::{camera::Camera, Vertex},
    simulation::objects::{AccretionDisk, ParticleSystem, Skybox}
};
use nalgebra::{Vector3, Vector4};
use rayon::prelude::*;
use image::{ImageBuffer, Rgb};
```

**EXPORTS**:
```rust
pub struct RayTracer {
    // Core components
    geodesic_integrator: GeodesicIntegrator,
    
    // Rendering parameters
    image_width: u32,
    image_height: u32,
    samples_per_pixel: u32,
    max_ray_depth: u32,
    
    // Performance settings
    parallel_threads: usize,
    adaptive_sampling: bool,
    quality_level: RayTracingQuality,
    
    // Framebuffer
    framebuffer: Vec<[f32; 4]>,  // RGBA pixels
    depth_buffer: Vec<f32>,
}

pub enum RayTracingQuality {
    Ultra,    // Full geodesic integration for every ray
    High,     // Geodesic with reduced precision
    Medium,   // Hybrid: geodesic for near field, approximation for far
    Low,      // Newtonian approximation with correction
    Preview,  // Simple ray marching
}

impl RayTracer {
    pub fn new(width: u32, height: u32) -> Self
    
    // PRIMARY RENDERING FUNCTION (Called by renderer.rs every frame)
    pub fn trace_frame(
        &mut self,
        camera: &Camera,
        black_hole: &BlackHole,
        scene_objects: &SceneObjects  // From objects.rs
    ) -> &[[f32; 4]]  // Returns framebuffer
    // CALLS: trace_ray for every pixel (potentially millions of calls)
    
    // CORE RAY TRACING (Performance critical - called millions of times)
    fn trace_ray(
        &self,
        ray_origin: Vector3<f64>,
        ray_direction: Vector3<f64>,
        black_hole: &BlackHole,
        scene_objects: &SceneObjects,
        depth: u32
    ) -> [f32; 4]  // Returns RGBA color
    // CALLS: geodesics.rs integration functions
    
    // PARALLEL PROCESSING
    pub fn trace_frame_parallel(
        &mut self,
        camera: &Camera,
        black_hole: &BlackHole,
        scene_objects: &SceneObjects
    ) -> &[[f32; 4]]
    // Divides screen into tiles for parallel processing
    
    // QUALITY MANAGEMENT
    pub fn set_quality(&mut self, quality: RayTracingQuality)
    pub fn adaptive_quality_update(&mut self, target_fps: f32, current_fps: f32)
}

// Scene Intersection
impl RayTracer {
    fn intersect_scene(
        &self,
        ray_positions: &[Vector4<f64>],  // From geodesic integration
        scene_objects: &SceneObjects
    ) -> Option<Intersection>
    
    fn intersect_accretion_disk(
        &self,
        ray_positions: &[Vector4<f64>],
        disk: &AccretionDisk
    ) -> Option<f64>  // Returns intersection distance
    
    fn intersect_event_horizon(
        &self,
        ray_positions: &[Vector4<f64>],
        black_hole: &BlackHole
    ) -> Option<f64>
}

// Lighting and Shading
impl RayTracer {
    fn calculate_lighting(
        &self,
        intersection_point: Vector3<f64>,
        surface_normal: Vector3<f64>,
        material: &Material,
        black_hole: &BlackHole
    ) -> [f32; 3]  // RGB color
    
    fn apply_gravitational_redshift(
        &self,
        base_color: [f32; 3],
        observer_pos: Vector3<f64>,
        source_pos: Vector3<f64>,
        black_hole: &BlackHole
    ) -> [f32; 3]
    
    fn apply_doppler_effect(
        &self,
        base_color: [f32; 3],
        source_velocity: Vector3<f64>,
        observer_velocity: Vector3<f64>
    ) -> [f32; 3]
}

// Performance Optimization
impl RayTracer {
    fn should_use_approximation(&self, distance_to_bh: f64) -> bool
    // Switch to approximation for distant rays
    
    fn adaptive_step_size(&self, ray_position: Vector3<f64>, black_hole: &BlackHole) -> f64
    // Smaller steps near event horizon, larger steps far away
    
    fn early_ray_termination(&self, ray_positions: &[Vector4<f64>]) -> bool
    // Stop tracing if ray escapes to infinity or falls into black hole
}

// Specialized Effects
pub fn gravitational_lensing_effect(
    original_ray: Vector3<f64>,
    black_hole: &BlackHole
) -> Vector3<f64>  // Returns bent ray direction
// USED BY: trace_ray for all light bending

pub fn photon_ring_detection(
    ray_positions: &[Vector4<f64>],
    black_hole: &BlackHole
) -> bool  // Detects if ray orbits multiple times
// USED BY: Advanced visual effects
```

**INTERNAL PERFORMANCE OPTIMIZATIONS**:
```rust
struct RayBatch {
    origins: Vec<Vector3<f64>>,
    directions: Vec<Vector3<f64>>,
    results: Vec<[f32; 4]>,
}

struct TileRenderer {
    tile_size: (u32, u32),
    tiles: Vec<RayBatch>,
}

struct AdaptiveSampler {
    base_samples: u32,
    max_samples: u32,
    variance_threshold: f32,
}
```

**USED BY**: `renderer.rs` (primary consumer)
**USES**: 
- `geodesics.rs` → ALL trajectory calculations (PERFORMANCE CRITICAL)
- `spacetime.rs` → Redshift and metric calculations
- `objects.rs` → Scene intersection testing
- `camera.rs` → Ray generation

---

### **shaders.rs**
**PURPOSE**: WGSL shader compilation and GPU program management

**IMPORTS**:
```rust
use wgpu;
use std::{fs, path::Path};
use crate::rendering::{RenderError, RenderResult, Vertex, Uniforms};
```

**EXPORTS**:
```rust
pub struct ShaderManager {
    device: wgpu::Device,  // Reference to GPU device
    
    // Compiled shaders
    vertex_shader: wgpu::ShaderModule,
    fragment_shader: wgpu::ShaderModule,
    compute_shader: Option<wgpu::ShaderModule>,
    
    // Pipeline layouts
    render_pipeline_layout: wgpu::PipelineLayout,
    compute_pipeline_layout: Option<wgpu::PipelineLayout>,
    
    // Bind group layouts for uniforms
    uniform_bind_group_layout: wgpu::BindGroupLayout
    // Hot reload support (development)
    shader_watch: Option<ShaderWatcher>,
    last_modified: std::collections::HashMap<String, std::time::SystemTime>,
}

impl ShaderManager {
    // INITIALIZATION (Called by renderer.rs during setup)
    pub fn new(device: wgpu::Device) -> RenderResult<Self>
    
    // SHADER LOADING
    pub fn load_vertex_shader(&mut self, path: &Path) -> RenderResult<()>
    pub fn load_fragment_shader(&mut self, path: &Path) -> RenderResult<()>
    pub fn load_compute_shader(&mut self, path: &Path) -> RenderResult<()>
    pub fn reload_all_shaders(&mut self) -> RenderResult<()>  // Development hot-reload
    
    // PIPELINE CREATION (Used by renderer.rs)
    pub fn create_render_pipeline(
        &self,
        surface_format: wgpu::TextureFormat,
        vertex_layout: wgpu::VertexBufferLayout
    ) -> RenderResult<wgpu::RenderPipeline>
    // RETURNS: Complete pipeline for renderer.rs
    
    pub fn create_compute_pipeline(&self) -> RenderResult<wgpu::ComputePipeline>
    // Optional: For GPU-accelerated physics
    
    // BIND GROUP MANAGEMENT
    pub fn create_uniform_bind_group(
        &self,
        uniform_buffer: &wgpu::Buffer
    ) -> wgpu::BindGroup
    // USED BY: renderer.rs for uniform buffer binding
    
    // GETTERS (Used by renderer.rs)
    pub fn uniform_bind_group_layout(&self) -> &wgpu::BindGroupLayout
    pub fn render_pipeline_layout(&self) -> &wgpu::PipelineLayout
}

// Development Tools
impl ShaderManager {
    fn check_shader_changes(&mut self) -> RenderResult<Vec<String>>
    fn validate_shader_compilation(source: &str, stage: wgpu::ShaderStages) -> RenderResult<()>
    fn extract_uniform_layout(source: &str) -> Vec<UniformDefinition>
}

// Shader Resource Definitions
pub struct UniformDefinition {
    pub name: String,
    pub binding: u32,
    pub size: u64,
    pub uniform_type: UniformType,
}

pub enum UniformType {
    Matrix4x4,
    Vector3,
    Vector4,
    Scalar,
    Buffer,
}



```

**SHADER FILE SPECIFICATIONS**:

**assets/shaders/vertex.wgsl**:
```wgsl
// RECEIVES: Vertex data from renderer.rs
// OUTPUTS: Transformed positions to fragment shader
// UNIFORMS: view_projection_matrix, model_matrix, time
```

**assets/shaders/fragment.wgsl**:
```wgsl
// RECEIVES: Interpolated vertex data
// OUTPUTS: Final pixel colors
// UNIFORMS: camera_position, black_hole_data, lighting_params
// PERFORMS: Basic lighting, gravitational effects (simple cases)
```

**USED BY**: `renderer.rs` exclusively
**USES**: Shader files in `assets/shaders/`

---

## 🌌 **src/simulation/** - DETAILED SPECIFICATIONS

### **scene.rs**
**PURPOSE**: Central orchestrator - connects all systems

**IMPORTS**:
```rust
use crate::{
    physics::{constants::*, black_hole::BlackHole},
    rendering::{camera::{Camera, CameraController}, Uniforms},
    simulation::{
        objects::{SceneObjects, AccretionDisk, ParticleSystem, Skybox},
        animation::AnimationSystem,
        InputState, TimeState
    }
};
use winit::{event::KeyEvent, dpi::PhysicalPosition};
```

**EXPORTS**:
```rust
pub struct Scene {
    // Core simulation objects
    black_hole: BlackHole,
    camera: Camera,
    camera_controller: CameraController,
    
    // Scene content
    scene_objects: SceneObjects,
    animation_system: AnimationSystem,
    
    // State management
    input_state: InputState,
    time_state: TimeState,
    simulation_state: SimulationState,
    
    // Configuration
    physics_settings: PhysicsSettings,
    render_settings: RenderSettings,
}

pub enum SimulationState {
    Running,
    Paused,
    StepByStep,
    Rewinding,  // Future: time reversal
}

pub struct PhysicsSettings {
    pub time_scale: f64,           // Speed up/slow down simulation
    pub integration_precision: IntegrationPrecision,
    pub enable_relativistic_effects: bool,
    pub black_hole_interactive: bool,  // Allow user to modify BH properties
}

pub struct RenderSettings {
    pub ray_tracing_quality: RayTracingQuality,
    pub show_debug_info: bool,
    pub show_coordinate_grid: bool,
    pub show_photon_sphere: bool,
    pub show_event_horizon: bool,
}

impl Scene {
    // INITIALIZATION (Called from main.rs)
    pub fn new() -> Self
    pub fn new_with_black_hole(mass_solar: f64, spin: f64) -> Self
    pub fn new_preset(preset: ScenePreset) -> Self
    
    // MAIN UPDATE LOOP (Called every frame from main.rs)
    pub fn update(&mut self)
    // CALLS: update_physics, update_animations, update_camera
    // COORDINATES: All subsystem updates
    
    // INPUT HANDLING (Called from main.rs event loop)
    pub fn handle_input(&mut self, event: &KeyEvent)
    pub fn handle_mouse_move(&mut self, position: PhysicalPosition<f64>)
    pub fn handle_mouse_scroll(&mut self, delta: f32)
    pub fn handle_window_resize(&mut self, width: u32, height: u32)
    
    // GETTERS (Used by renderer.rs)
    pub fn camera(&self) -> &Camera                    // USED BY: renderer.rs uniforms
    pub fn black_hole(&self) -> &BlackHole            // USED BY: renderer.rs uniforms
    pub fn scene_objects(&self) -> &SceneObjects      // USED BY: ray_tracer.rs
    pub fn current_time(&self) -> f32                 // USED BY: renderer.rs uniforms
    
    // SIMULATION CONTROL
    pub fn pause(&mut self)
    pub fn resume(&mut self)
    pub fn reset(&mut self)
    pub fn set_time_scale(&mut self, scale: f64)
    
    // BLACK HOLE MANIPULATION
    pub fn set_black_hole_mass(&mut self, mass_solar: f64)
    pub fn set_black_hole_spin(&mut self, spin: f64)
    pub fn add_accretion_matter(&mut self, amount: f64)
}

// Internal Update Methods
impl Scene {
    fn update_physics(&mut self)
    // Updates particle positions, orbital mechanics
    // CALLS: animation_system.update, scene_objects.update_physics
    
    fn update_camera(&mut self)
    // Handles camera movement and constraints
    // CALLS: camera_controller.update
    
    fn update_debug_info(&mut self)
    // Updates performance metrics, physics debug data
    
    fn handle_special_keys(&mut self, event: &KeyEvent)
    // Special simulation controls (pause, reset, debug toggles)
}

// Scene Presets
pub enum ScenePreset {
    StellarMassBlackHole,      // 10 solar masses, basic accretion disk
    SupermassiveBlackHole,     // 4 million solar masses (Sgr A* style)
    SpinningBlackHole,         // High spin with complex accretion
    BinaryBlackHoles,          // Two black holes (future)
    EducationalDemo,           // Simplified for learning
}

impl ScenePreset {
    pub fn create_scene(&self) -> Scene
    // Factory method for different scenarios
}
```

**INTERNAL STATE MANAGEMENT**:
```rust
struct SimulationMetrics {
    physics_update_time: f64,
    rendering_time: f64,
    total_particles: usize,
    rays_traced_per_frame: usize,
}

struct DebugInfo {
    show_fps: bool,
    show_camera_position: bool,
    show_black_hole_stats: bool,
    show_geodesic_paths: bool,
    show_performance_metrics: bool,
}
```

**USED BY**: `main.rs` exclusively
**USES**: ALL other modules (orchestrates everything)

---

### **objects.rs**
**PURPOSE**: All renderable objects and their physics

**IMPORTS**:
```rust
use crate::{
    physics::{
        constants::*, 
        black_hole::BlackHole, 
        geodesics::{GeodesicIntegrator, orbital_frequency, calculate_isco_orbit},
        spacetime::{proper_time_factor, gravitational_redshift}
    },
    rendering::{Vertex, ray_tracer::Material},
    simulation::{animation::Animatable, TimeState}
};
use nalgebra::{Vector3, Vector4};
use std::f64::consts::PI;
```

**EXPORTS**:
```rust
// Master container for all scene objects
pub struct SceneObjects {
    pub accretion_disk: Option<AccretionDisk>,
    pub particle_system: ParticleSystem,
    pub skybox: Skybox,
    pub debug_objects: DebugObjects,
    pub photon_ring: Option<PhotonRing>,  // Advanced effect
}

impl SceneObjects {
    pub fn new() -> Self
    pub fn new_with_accretion_disk(black_hole: &BlackHole) -> Self
    
    // PHYSICS UPDATE (Called by scene.rs every frame)
    pub fn update_physics(&mut self, black_hole: &BlackHole, delta_time: f64)
    // CALLS: Each object's update method
    
    // RENDERING DATA (Used by ray_tracer.rs)
    pub fn get_intersectable_objects(&self) -> Vec<&dyn Intersectable>
    pub fn get_light_sources(&self) -> Vec<&dyn LightSource>
}

// ACCRETION DISK - Complex astrophysical object
pub struct AccretionDisk {
    // Physical properties
    inner_radius: f64,          // Usually at ISCO
    outer_radius: f64,          // Extends to ~1000 Schwarzschild radii
    mass_accretion_rate: f64,   // kg/s
    inclination_angle: f64,     // Viewing angle in radians
    
    // Mesh data for rendering
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    
    // Physics simulation
    particles: Vec<DiskParticle>,
    temperature_profile: TemperatureProfile,
    velocity_profile: VelocityProfile,
    
    // Animation state
    rotation_angle: f64,
    inner_edge_precession: f64,  // Relativistic precession
}

pub struct DiskParticle {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub temperature: f64,        // Kelvin
    pub density: f64,           // kg/m³
    pub orbital_phase: f64,     // 0 to 2π
    pub lifetime: f64,          // Time until falls into BH or escapes
}

impl AccretionDisk {
    // CONSTRUCTORS
    pub fn new(black_hole: &BlackHole) -> Self
    pub fn shakura_sunyaev_disk(black_hole: &BlackHole, accretion_rate: f64) -> Self
    // Standard thin disk model
    
    pub fn thick_disk(black_hole: &BlackHole, accretion_rate: f64) -> Self
    // For high accretion rates
    
    // PHYSICS CALCULATIONS (Used internally and by ray_tracer.rs)
    pub fn temperature_at_radius(&self, radius: f64, black_hole: &BlackHole) -> f64
    // USED BY: ray_tracer.rs for blackbody radiation color
    
    pub fn surface_brightness(&self, radius: f64, black_hole: &BlackHole) -> f64
    // USED BY: ray_tracer.rs for luminosity
    
    pub fn orbital_velocity(&self, radius: f64, black_hole: &BlackHole) -> f64
    // USED BY: Doppler shift calculations in ray_tracer.rs
    
    pub fn scale_height(&self, radius: f64, black_hole: &BlackHole) -> f64
    // Disk thickness - USED BY: ray_tracer.rs for intersection testing
    
    // UPDATE METHODS (Called by scene.rs)
    pub fn update_particle_positions(&mut self, black_hole: &BlackHole, delta_time: f64)
    // CALLS: geodesics.rs for particle trajectories
    
    pub fn update_temperature_profile(&mut self, black_hole: &BlackHole)
    // Recalculates based on accretion rate changes
    
    // RENDERING INTERFACE (Used by ray_tracer.rs)
    pub fn intersect_ray(&self, ray_origin: Vector3<f64>, ray_direction: Vector3<f64>) -> Option<f64>
    pub fn get_material_at_point(&self, point: Vector3<f64>) -> Material
    pub fn get_surface_normal(&self, point: Vector3<f64>) -> Vector3<f64>
}

// PARTICLE SYSTEM - Individual orbiting particles
pub struct ParticleSystem {
    particles: Vec<OrbitingParticle>,
    max_particles: usize,
    spawn_rate: f64,            // particles per second
    particle_lifetime: f64,     // seconds before cleanup
    
    // Visual properties
    particle_size: f32,
    particle_brightness: f32,
    trail_length: usize,        // For motion trails
}

pub struct OrbitingParticle {
    pub current_position: Vector3<f64>,
    pub trajectory: Vec<Vector4<f64>>,  // Precomputed geodesic path
    pub trajectory_index: usize,        // Current position in trajectory
    pub birth_time: f64,
    pub color: [f32; 3],               // RGB based on temperature
    pub trail_positions: Vec<Vector3<f64>>,
}

impl ParticleSystem {
    pub fn new(max_particles: usize) -> Self
    
    // SPAWNING (Called by scene.rs based on accretion rate)
    pub fn spawn_particle_at_radius(
        &mut self, 
        radius: f64, 
        black_hole: &BlackHole,
        geodesic_integrator: &GeodesicIntegrator
    )
    // CALLS: geodesics.rs to precompute trajectory
    
    // UPDATE (Called every frame)
    pub fn update_particles(&mut self, delta_time: f64)
    // Advances particles along precomputed trajectories
    
    pub fn cleanup_dead_particles(&mut self)
    // Removes particles that fell into BH or escaped
    
    // RENDERING INTERFACE
    pub fn get_visible_particles(&self) -> &[OrbitingParticle]
    // USED BY: ray_tracer.rs for point light sources
}

// SKYBOX - Background stars and cosmic objects
pub struct Skybox {
    star_positions: Vec<Vector3<f64>>,   // Unit sphere positions
    star_magnitudes: Vec<f32>,           // Visual brightness
    star_colors: Vec<[f32; 3]>,         // RGB colors
    
    // Special background objects
    distant_galaxies: Vec<DistantGalaxy>,
    cosmic_microwave_background: bool,
    
    // Texture data (if using texture-based skybox)
    texture_data: Option<Vec<u8>>,
}

impl Skybox {
    pub fn new_procedural() -> Self      // Generate random realistic star field
    pub fn new_from_catalog() -> Self    // Use real star catalog data
    
    // RENDERING INTERFACE (Used by ray_tracer.rs for background)
    pub fn get_background_color(&self, direction: Vector3<f64>) -> [f32; 3]
    // CALLED: When ray escapes to infinity
    
    pub fn apply_gravitational_lensing(
        &self, 
        original_direction: Vector3<f64>,
        lensed_direction: Vector3<f64>
    ) -> [f32; 3]
    // Shows how background is distorted by black hole
}

// DEBUG VISUALIZATION OBJECTS
pub struct DebugObjects {
    pub show_event_horizon: bool,
    pub show_photon_sphere: bool,
    pub show_isco: bool,
    pub show_ergosphere: bool,
    pub show_coordinate_grid: bool,
    pub show_geodesic_paths: Vec<Vec<Vector4<f64>>>,  // Precomputed paths to display
}

impl DebugObjects {
    pub fn generate_horizon_sphere(&self, black_hole: &BlackHole) -> Vec<Vertex>
    pub fn generate_photon_sphere(&self, black_hole: &BlackHole) -> Vec<Vertex>
    pub fn generate_coordinate_grid(&self, black_hole: &BlackHole) -> Vec<Vertex>
    
    // USED BY: renderer.rs for debug visualization
}

// ADVANCED EFFECTS
pub struct PhotonRing {
    // Photons that orbit the black hole multiple times
    ring_geodesics: Vec<Vec<Vector4<f64>>>,
    ring_brightness: f32,
    ring_width: f64,
}

// MATERIAL SYSTEM (Used by ray_tracer.rs)
pub struct Material {
    pub albedo: [f32; 3],              // Base color
    pub emissive: [f32; 3],            // Self-illumination
    pub temperature: f64,              // For blackbody radiation
    pub metallic: f32,                 // PBR material properties
    pub roughness: f32,
    pub opacity: f32,
}

// INTERSECTION TESTING (Used by ray_tracer.rs)
pub trait Intersectable {
    fn intersect(&self, ray_origin: Vector3<f64>, ray_direction: Vector3<f64>) -> Option<IntersectionData>;
    fn get_bounding_box(&self) -> BoundingBox;
}

pub struct IntersectionData {
    pub distance: f64,
    pub point: Vector3<f64>,
    pub normal: Vector3<f64>,
    pub material: Material,
    pub uv_coordinates: (f32, f32),    // Texture coordinates
}

// LIGHT SOURCES (Used by ray_tracer.rs for illumination)
pub trait LightSource {
    fn get_light_at_point(&self, point: Vector3<f64>) -> LightData;
    fn is_occluded(&self, from: Vector3<f64>, to: Vector3<f64>) -> bool;
}

pub struct LightData {
    pub color: [f32; 3],
    pub intensity: f32,
    pub direction: Vector3<f64>,       // For directional lights
    pub attenuation: f32,              // Distance falloff
}
```

**INTERNAL CALCULATIONS**:
```rust
// Temperature profiles for accretion disk
struct TemperatureProfile {
    inner_temp: f64,         // Temperature at inner edge
    temp_index: f64,         // Power law index (typically -0.75)
}

struct VelocityProfile {
    keplerian: bool,         // True for thin disk
    relativistic_corrections: bool,
}

// Mesh generation utilities
fn generate_disk_mesh(
    inner_radius: f64, 
    outer_radius: f64, 
    radial_segments: usize, 
    angular_segments: usize
) -> (Vec<Vertex>, Vec<u32>)

fn calculate_disk_normal(position: Vector3<f64>, inclination: f64) -> Vector3<f64>
```

**USED BY**: 
- `scene.rs` → Updates and physics management
- `ray_tracer.rs` → Intersection testing and material properties
- `animation.rs` → Animatable objects

**USES**: 
- `black_hole.rs` → Physical properties for scaling
- `geodesics.rs` → Particle trajectories and orbital mechanics
- `spacetime.rs` → Proper time and redshift effects

---

### **animation.rs**
**PURPOSE**: Time-based animations and smooth interpolations

**IMPORTS**:
```rust
use crate::{
    physics::{
        constants::*, 
        black_hole::BlackHole, 
        geodesics::{orbital_frequency, proper_time_factor}
    },
    simulation::{objects::{AccretionDisk, ParticleSystem, OrbitingParticle}, TimeState}
};
use nalgebra::{Vector3, Vector4};
```

**EXPORTS**:
```rust
pub struct AnimationSystem {
    // Global animation state
    global_time: f64,
    time_scale: f64,           // Speed multiplier for entire simulation
    paused: bool,
    
    // Animation controllers
    disk_animator: DiskAnimator,
    particle_animator: ParticleAnimator,
    camera_animator: CameraAnimator,
    
    // Interpolation system
    interpolators: Vec<Box<dyn Interpolator>>,
    
    // Physics-based timing
    proper_time_factor: f64,   // Gravitational time dilation
    coordinate_time: f64,      // Observer time
}

impl AnimationSystem {
    pub fn new() -> Self
    
    // MAIN UPDATE (Called by scene.rs every frame)
    pub fn update(
        &mut self, 
        delta_time: f64, 
        black_hole: &BlackHole,
        scene_objects: &mut crate::simulation::objects::SceneObjects
    )
    // CALLS: All animator update methods
    
    // TIME CONTROL
    pub fn set_time_scale(&mut self, scale: f64)
    pub fn pause(&mut self)
    pub fn resume(&mut self)
    pub fn reset_time(&mut self)
    
    // PHYSICS-AWARE TIMING
    pub fn update_gravitational_time_effects(&mut self, observer_position: Vector3<f64>, black_hole: &BlackHole)
    // Calculates proper time vs coordinate time
}

// ACCRETION DISK ANIMATION
pub struct DiskAnimator {
    // Rotation parameters
    inner_edge_angular_velocity: f64,    // rad/s at ISCO
    outer_edge_angular_velocity: f64,    // Slower rotation at outer edge
    differential_rotation: bool,         // Enable realistic rotation profile
    
    // Precession effects
    relativistic_precession_rate: f64,   // Frame dragging effects
    precession_accumulated: f64,
    
    // Temporal variations
    accretion_rate_variations: AccretionVariations,
    flickering_timescale: f64,          // Short-term brightness variations
    
    // Animation state
    current_rotation_angle: f64,
    last_update_time: f64,
}

impl DiskAnimator {
    pub fn new(black_hole: &BlackHole) -> Self
    
    // UPDATE METHODS (Called by AnimationSystem)
    pub fn update_disk_rotation(
        &mut self, 
        disk: &mut AccretionDisk, 
        delta_time: f64, 
        black_hole: &BlackHole
    )
    // Updates rotation angles based on Keplerian orbits + relativistic corrections
    
    pub fn update_temperature_variations(
        &mut self, 
        disk: &mut AccretionDisk, 
        delta_time: f64
    )
    // Simulates thermal fluctuations in accretion disk
    
    pub fn update_precession_effects(
        &mut self, 
        disk: &mut AccretionDisk, 
        delta_time: f64, 
        black_hole: &BlackHole
    )
    // Frame dragging causes disk to precess
}

// PARTICLE SYSTEM ANIMATION
pub struct ParticleAnimator {
    // Spawning control
    spawn_timer: f64,
    spawn_interval: f64,               // Time between new particles
    spawn_radius_distribution: SpawnDistribution,
    
    // Trajectory management
    trajectory_update_interval: f64,   // How often to recalculate geodesics
    trajectory_cache: std::collections::HashMap<u64, Vec<Vector4<f64>>>,
    
    // Visual effects
    trail_fade_rate: f32,             // How quickly particle trails fade
    brightness_variation: BrightnessVariation,
}

impl ParticleAnimator {
    pub fn new() -> Self
    
    pub fn update_particle_spawning(
        &mut self, 
        particle_system: &mut ParticleSystem, 
        delta_time: f64,
        black_hole: &BlackHole
    )
    // Controls when and where new particles are created
    
    pub fn update_particle_motion(
        &mut self, 
        particle_system: &mut ParticleSystem, 
        delta_time: f64
    )
    // Advances particles along their geodesic trajectories
    
    pub fn update_visual_effects(
        &mut self, 
        particle_system: &mut ParticleSystem, 
        delta_time: f64
    )
    // Updates trails, brightness, color temperature
}

// CAMERA ANIMATION (For automatic tours and smooth movements)
pub struct CameraAnimator {
    // Keyframe system
    keyframes: Vec<CameraKeyframe>,
    current_keyframe: usize,
    keyframe_progress: f32,            // 0.0 to 1.0 between keyframes
    
    // Smooth interpolation
    position_interpolator: Vec3Interpolator,
    rotation_interpolator: QuaternionInterpolator,
    
    // Automatic behaviors
    orbital_mode: bool,
    orbital_radius: f64,
    orbital_speed: f64,
    orbital_angle: f64,
}

pub struct CameraKeyframe {
    pub time: f64,
    pub position: Vector3<f64>,
    pub look_at: Vector3<f64>,
    pub field_of_view: f32,
    pub transition_type: TransitionType,
}

pub enum TransitionType {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bezier(Vector3<f64>, Vector3<f64>),  // Control points for smooth curves
}

impl CameraAnimator {
    pub fn new() -> Self
    
    pub fn add_keyframe(&mut self, keyframe: CameraKeyframe)
    pub fn start_automatic_tour(&mut self)
    pub fn enable_orbital_mode(&mut self, radius: f64, speed: f64)
    
    pub fn update_camera_animation(
        &mut self, 
        camera: &mut crate::rendering::camera::Camera, 
        delta_time: f64
    )
    // Smoothly moves camera between keyframes or in orbital mode
}

// INTERPOLATION SYSTEM
pub trait Interpolator {
    fn update(&mut self, delta_time: f64);
    fn is_complete(&self) -> bool;
    fn get_current_value(&self) -> f64;
}

pub struct FloatInterpolator {
    start_value: f64,
    end_value: f64,
    current_value: f64,
    duration: f64,
    elapsed_time: f64,
    easing_function: EasingFunction,
}

pub struct Vec3Interpolator {
    start: Vector3<f64>,
    end: Vector3<f64>,
    current: Vector3<f64>,
    duration: f64,
    elapsed_time: f64,
    easing_function: EasingFunction,
}

pub enum EasingFunction {
    Linear,
    QuadraticIn,
    QuadraticOut,
    CubicInOut,
    SineInOut,
    Custom(fn(f64) -> f64),
}

impl FloatInterpolator {
    pub fn new(start: f64, end: f64, duration: f64, easing: EasingFunction) -> Self
    
    pub fn lerp(start: f64, end: f64, t: f64) -> f64  // Linear interpolation
    pub fn slerp(start: Vector3<f64>, end: Vector3<f64>, t: f64) -> Vector3<f64>  // Spherical
}

// PHYSICS-BASED ANIMATIONS
pub struct PhysicsAnimator {
    // Gravitational wave effects (future feature)
    gw_amplitude: f64,
    gw_frequency: f64,
    
    // Tidal effects
    tidal_deformation_factor: f64,
    
    // Relativistic jets (for spinning black holes)
    jet_precession_angle: f64,
    jet_precession_period: f64,
}

// TIMING UTILITIES
pub struct AccretionVariations {
    base_rate: f64,
    variation_amplitude: f64,      // Fraction of base rate
    variation_timescale: f64,      // Seconds for one cycle
    variation_phase: f64,          // Current phase in cycle
}

impl AccretionVariations {
    pub fn get_current_rate(&self, time: f64) -> f64
    // Returns time-varying accretion rate
}

pub struct BrightnessVariation {
    base_brightness: f32,
    flicker_amplitude: f32,
    flicker_frequency: f64,
    random_seed: u64,
}

impl BrightnessVariation {
    pub fn get_brightness_factor(&self, time: f64, particle_id: u64) -> f32
    // Returns brightness multiplier for individual particles
}

// ANIMATION PRESETS
pub enum AnimationPreset {
    QuietAccretion,        // Steady, smooth accretion
    ActiveAccretion,       // Variable accretion with flares
    QuasarMode,           // Extremely active with jets
    TidalDisruption,      // Star being torn apart
    BinaryInspiral,       // Two black holes spiraling (future)
}

impl AnimationPreset {
    pub fn apply_to_system(&self, animation_system: &mut AnimationSystem)
    // Configures animation parameters for different scenarios
}
```

**INTERNAL UTILITIES**:
```rust
// Smooth interpolation functions
fn smooth_step(t: f64) -> f64                    // S-curve interpolation
fn smoother_step(t: f64) -> f64                  // Even smoother S-curve
fn ease_in_out_cubic(t: f64) -> f64             // Cubic easing

// Physics calculations
fn calculate_keplerian_frequency(radius: f64, mass: f64) -> f64
fn calculate_precession_rate(radius: f64, spin: f64, mass: f64) -> f64
fn relativistic_orbital_velocity(radius: f64, mass: f64) -> f64

// Random number generation for variations
struct NoiseGenerator {
    seed: u64,
    octaves: usize,
    frequency: f64,
    amplitude: f64,
}
```

**USED BY**: `scene.rs` (primary orchestrator)
**USES**: 
- `objects.rs` → Animates all renderable objects
- `geodesics.rs` → Physics-based motion calculations
- `spacetime.rs` → Time dilation effects

---

## 🔄 **CRITICAL CROSS-MODULE DATA FLOW**

### **Performance-Critical Paths (Optimized)**:

1. **Ray Tracing Pipeline** (60+ FPS target):
   ```
   renderer.rs → ray_tracer.rs → geodesics.rs → spacetime.rs
   ```
   - **Called**: Millions of times per frame
   - **Optimization**: Parallel processing, adaptive quality, caching
   - **Data Flow**: Camera rays → Geodesic integration → Pixel colors

2. **Animation Update** (Every frame):
   ```
   scene.rs → animation.rs → objects.rs → geodesics.rs
   ```
   - **Called**: Once per frame per animated object
   - **Optimization**: Cached traject
   - **Data Flow**: Time delta → Object state updates → Physics calculations

3. **Uniform Buffer Updates** (Every frame):
   ```
   scene.rs → renderer.rs → GPU uniforms
   ```
   - **Called**: Once per frame
   - **Optimization**: Only update changed data, batch transfers
   - **Data Flow**: Scene state → GPU memory → Shaders

### **Memory-Critical Dependencies**:

1. **Geodesic Trajectory Caching**:
   ```rust
   // In geodesics.rs - MUST be accessible by:
   trajectory_cache: HashMap<TrajectoryKey, Vec<Vector4<f64>>>
   ```
   - **Used by**: `ray_tracer.rs`, `objects.rs`, `animation.rs`
   - **Memory Impact**: 10-100MB typical, scales with scene complexity
   - **Invalidation**: When black hole properties change

2. **Vertex Buffer Management**:
   ```rust
   // In objects.rs - MUST be accessible by renderer.rs:
   pub fn get_vertex_data(&self) -> (&[Vertex], &[u32])
   ```
   - **Used by**: `renderer.rs` for GPU upload
   - **Memory Impact**: 1-50MB depending on mesh detail
   - **Update Pattern**: Only when geometry changes

---

## 🎯 **MANDATORY METHOD SIGNATURES** 

**These EXACT signatures must be implemented to avoid import/unused variable issues:**

### **From physics/black_hole.rs - MUST export**:
```rust
impl BlackHole {
    pub fn mass(&self) -> f64                          // → spacetime.rs, objects.rs, renderer.rs
    pub fn spin(&self) -> f64                         // → spacetime.rs, geodesics.rs
    pub fn position(&self) -> Vector3<f64>            // → renderer.rs, ray_tracer.rs
    pub fn schwarzschild_radius(&self) -> f64         // → spacetime.rs, objects.rs, renderer.rs
    pub fn isco_radius(&self) -> f64                  // → objects.rs
    pub fn photon_sphere_radius(&self) -> f64         // → geodesics.rs, objects.rs
}
```

### **From physics/spacetime.rs - MUST export**:
```rust
pub fn cartesian_to_schwarzschild(pos: Vector3<f64>) -> Vector3<f64>  // → geodesics.rs
pub fn schwarzschild_to_cartesian(r: f64, theta: f64, phi: f64) -> Vector3<f64>  // → geodesics.rs
pub fn gravitational_redshift(bh: &BlackHole, obs: Vector3<f64>, src: Vector3<f64>) -> f64  // → ray_tracer.rs
pub fn proper_time_factor(bh: &BlackHole, pos: Vector3<f64>) -> f64    // → objects.rs, animation.rs

impl MetricTensor {
    pub fn schwarzschild(bh: &BlackHole, pos: Vector4<f64>) -> Self    // → geodesics.rs
    pub fn kerr(bh: &BlackHole, pos: Vector4<f64>) -> Self             // → geodesics.rs
    pub fn christoffel_symbols(&self) -> ChristoffelTensor             // → geodesics.rs
    pub fn determinant(&self) -> f64                                   // → geodesics.rs
}
```

### **From physics/geodesics.rs - MUST export**:
```rust
impl GeodesicIntegrator {
    pub fn integrate_photon_geodesic(
        &self, bh: &BlackHole, pos: Vector4<f64>, dir: Vector3<f64>, max_time: f64
    ) -> Vec<Vector4<f64>>                                             // → ray_tracer.rs
    
    pub fn integrate_particle_orbit(
        &self, bh: &BlackHole, pos: Vector4<f64>, vel: Vector4<f64>, mass: f64, time: f64
    ) -> Vec<Vector4<f64>>                                             // → objects.rs
    
    pub fn calculate_orbital_frequency(bh: &BlackHole, radius: f64) -> f64  // → animation.rs, objects.rs
    pub fn deflection_angle(bh: &BlackHole, impact_param: f64) -> f64       // → ray_tracer.rs
}
```

### **From rendering/camera.rs - MUST export**:
```rust
impl Camera {
    pub fn position(&self) -> Vector3<f64>                            // → renderer.rs, scene.rs
    pub fn view_matrix(&self) -> glam::Mat4                           // → renderer.rs
    pub fn projection_matrix(&self) -> glam::Mat4                     // → renderer.rs
    pub fn view_projection_matrix(&self) -> glam::Mat4                // → renderer.rs
    pub fn screen_to_world_ray(&self, x: f32, y: f32, w: f32, h: f32) -> (Vector3<f64>, Vector3<f64>)  // → ray_tracer.rs
    pub fn set_aspect_ratio(&mut self, aspect: f32)                  // → renderer.rs
}

impl CameraController {
    pub fn update(&mut self, camera: &mut Camera, input: &InputState, time: &TimeState, bh: &BlackHole)  // → scene.rs
}
```

### **From rendering/ray_tracer.rs - MUST export**:
```rust
impl RayTracer {
    pub fn trace_frame(
        &mut self, camera: &Camera, bh: &BlackHole, objects: &SceneObjects
    ) -> &[[f32; 4]]                                                  // → renderer.rs
    
    pub fn set_quality(&mut self, quality: RayTracingQuality)        // → renderer.rs, scene.rs
}
```

### **From rendering/renderer.rs - MUST export**:
```rust
impl Renderer {
    pub async fn new(window: &Window) -> RenderResult<Self>           // → main.rs
    pub fn render(&mut self, scene: &Scene) -> RenderResult<()>      // → main.rs
    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>)    // → main.rs
}
```

### **From simulation/objects.rs - MUST export**:
```rust
impl SceneObjects {
    pub fn update_physics(&mut self, bh: &BlackHole, delta_time: f64)  // → scene.rs
    pub fn get_intersectable_objects(&self) -> Vec<&dyn Intersectable>  // → ray_tracer.rs
}

impl AccretionDisk {
    pub fn new(bh: &BlackHole) -> Self                               // → scene.rs
    pub fn temperature_at_radius(&self, radius: f64, bh: &BlackHole) -> f64  // → ray_tracer.rs
    pub fn intersect_ray(&self, origin: Vector3<f64>, dir: Vector3<f64>) -> Option<f64>  // → ray_tracer.rs
    pub fn get_material_at_point(&self, point: Vector3<f64>) -> Material  // → ray_tracer.rs
}

impl ParticleSystem {
    pub fn update_particles(&mut self, delta_time: f64)              // → scene.rs
    pub fn get_visible_particles(&self) -> &[OrbitingParticle]       // → ray_tracer.rs
}

pub trait Intersectable {
    fn intersect(&self, origin: Vector3<f64>, dir: Vector3<f64>) -> Option<IntersectionData>;  // → ray_tracer.rs
}
```

### **From simulation/scene.rs - MUST export**:
```rust
impl Scene {
    pub fn new() -> Self                                             // → main.rs
    pub fn update(&mut self)                                         // → main.rs
    pub fn handle_input(&mut self, event: &KeyEvent)                // → main.rs
    pub fn handle_mouse_move(&mut self, pos: PhysicalPosition<f64>) // → main.rs
    pub fn camera(&self) -> &Camera                                 // → renderer.rs
    pub fn black_hole(&self) -> &BlackHole                         // → renderer.rs
    pub fn scene_objects(&self) -> &SceneObjects                   // → ray_tracer.rs
    pub fn current_time(&self) -> f32                               // → renderer.rs
}
```

### **From simulation/animation.rs - MUST export**:
```rust
impl AnimationSystem {
    pub fn update(&mut self, dt: f64, bh: &BlackHole, objects: &mut SceneObjects)  // → scene.rs
    pub fn set_time_scale(&mut self, scale: f64)                    // → scene.rs
}
```

---

## 🚨 **CRITICAL VARIABLE NAMING CONVENTIONS**

**To avoid unused variable warnings, these EXACT names must be used:**

### **In physics modules**:
```rust
// constants.rs
pub const G: f64                    // ← USED BY: black_hole.rs, spacetime.rs, geodesics.rs
pub const C: f64                    // ← USED BY: spacetime.rs, geodesics.rs
pub const SOLAR_MASS: f64           // ← USED BY: black_hole.rs
pub const SCHWARZSCHILD_COEFF: f64  // ← USED BY: black_hole.rs, spacetime.rs

// black_hole.rs
struct BlackHole {
    mass_solar: f64,                // ← USED BY: spacetime.rs, geodesics.rs, objects.rs
    spin: f64,                      // ← USED BY: spacetime.rs, geodesics.rs
    position: Vector3<f64>,         // ← USED BY: renderer.rs, ray_tracer.rs
}

// spacetime.rs
struct MetricTensor {
    components: SMatrix<f64, 4, 4>, // ← USED BY: geodesics.rs
}

// geodesics.rs
struct GeodesicIntegrator {
    tolerance: f64,                 // ← USED BY: internal calculations
    max_steps: usize,               // ← USED BY: integration loops
}
```

### **In rendering modules**:
```rust
// renderer.rs
struct Renderer {
    device: wgpu::Device,           // ← USED BY: buffer creation, pipeline setup
    queue: wgpu::Queue,             // ← USED BY: command submission
    surface: wgpu::Surface,         // ← USED BY: frame presentation
    render_pipeline: wgpu::RenderPipeline,  // ← USED BY: render passes
    uniform_buffer: wgpu::Buffer,   // ← USED BY: uniform updates
    ray_tracer: RayTracer,          // ← USED BY: frame rendering
}

// camera.rs
struct Camera {
    position: Vector3<f64>,         // ← USED BY: renderer.rs, ray_tracer.rs, scene.rs
    view_matrix: glam::Mat4,        // ← USED BY: renderer.rs uniforms
    projection_matrix: glam::Mat4,  // ← USED BY: renderer.rs uniforms
}

// ray_tracer.rs
struct RayTracer {
    geodesic_integrator: GeodesicIntegrator,  // ← USED BY: trace_ray method
    framebuffer: Vec<[f32; 4]>,     // ← USED BY: pixel output to renderer
}
```

### **In simulation modules**:
```rust
// scene.rs
struct Scene {
    black_hole: BlackHole,          // ← USED BY: renderer.rs, ray_tracer.rs, animation.rs
    camera: Camera,                 // ← USED BY: renderer.rs, ray_tracer.rs
    scene_objects: SceneObjects,    // ← USED BY: ray_tracer.rs, animation.rs
    input_state: InputState,        // ← USED BY: camera controller
    time_state: TimeState,          // ← USED BY: animation system
}

// objects.rs
struct AccretionDisk {
    vertices: Vec<Vertex>,          // ← USED BY: renderer.rs for GPU upload
    particles: Vec<DiskParticle>,   // ← USED BY: animation.rs, ray_tracer.rs
    inner_radius: f64,              // ← USED BY: intersection testing
    outer_radius: f64,              // ← USED BY: intersection testing
}

// animation.rs
struct AnimationSystem {
    disk_animator: DiskAnimator,    // ← USED BY: update method
    particle_animator: ParticleAnimator,  // ← USED BY: update method
    global_time: f64,               // ← USED BY: all animation calculations
}
```

---

## 📋 **BUILD ORDER & DEPENDENCY RESOLUTION**

**To ensure clean compilation, implement in this EXACT order:**

### **Phase 1: Core Physics Foundation**
1. `src/physics/constants.rs` - No dependencies
2. `src/physics/black_hole.rs` - Imports: constants
3. `src/physics/spacetime.rs` - Imports: constants, black_hole
4. `src/physics/geodesics.rs` - Imports: constants, black_hole, spacetime
5. `src/physics/mod.rs` - Exports all physics modules

### **Phase 2: Basic Rendering Infrastructure**
1. `src/rendering/shaders.rs` - No physics dependencies
2. `src/rendering/camera.rs` - Imports: physics/constants
3. `src/rendering/ray_tracer.rs` - Imports: ALL physics modules, camera
4. `src/rendering/renderer.rs` - Imports: camera, ray_tracer, shaders
5. `src/rendering/mod.rs` - Exports all rendering modules

### **Phase 3: Simulation Layer**
1. `src/simulation/objects.rs` - Imports: ALL physics modules
2. `src/simulation/animation.rs` - Imports: physics, objects
3. `src/simulation/scene.rs` - Imports: physics, rendering, objects, animation
4. `src/simulation/mod.rs` - Exports all simulation modules

### **Phase 4: Main Application**
1. `src/main.rs` - Imports: rendering::Renderer, simulation::Scene

**Each phase MUST compile without warnings before proceeding to the next.**

---

## ⚠️ **CRITICAL IMPLEMENTATION NOTES**

### **Error Handling Consistency**:
```rust
// ALL physics functions return Result<T, PhysicsError>
// ALL rendering functions return Result<T, RenderError>  
// ALL simulation functions return Result<T, SimulationError>
```

### **Thread Safety Requirements**:
```rust
// These structs MUST be Send + Sync for parallel ray tracing:
- BlackHole
- MetricTensor
- GeodesicIntegrator
- SpacetimeMetric
```

### **Performance Benchmarks**:
```rust
// MANDATORY performance targets:
- Ray tracing: 1M+ rays/second on CPU
- Geodesic integration: 10K+ integrations/second
- Frame rate: 30+ FPS at 1080p on integrated graphics
- Memory usage: < 500MB total
```

### **Numerical Precision Requirements**:
```rust
// Use f64 for ALL physics calculations
// Use f32 only for GPU data (rendering)
// NEVER mix f32/f64 in physics calculations
```

---

## 🎯 **FINAL ARCHITECTURAL VALIDATION**

**Before writing ANY code, verify these dependencies are correct:**

1. **No circular imports**: Each layer only depends on lower layers
2. **No unused public methods**: Every pub fn is called by another module
3. **No missing imports**: Every type used is properly imported
4. **Consistent error propagation**: All errors bubble up to main.rs appropriately
5. **Memory safety**: No dangling pointers, all references have proper lifetimes

This architecture ensures a **robust, maintainable, and performant** black hole simulation that can be extended with advanced features while maintaining scientific accuracy and real-time performance on CPU-only systems.

**START CODING ONLY AFTER CONFIRMING THIS ARCHITECTURE MAKES COMPLETE SENSE TO YOU.**