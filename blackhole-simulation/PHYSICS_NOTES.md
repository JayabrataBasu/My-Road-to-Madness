# Black Hole Simulation - Complete File Architecture & Function Guide

## 🎯 Core Architecture Philosophy

The project will follow a **layered architecture** where:
- **Physics Layer**: Pure scientific calculations, no graphics dependencies
- **Rendering Layer**: Graphics pipeline, no physics knowledge except through interfaces
- **Simulation Layer**: Orchestration layer that connects physics and rendering

---

## 📁 **src/physics/** - Scientific Computing Core

### **constants.rs**
**Purpose**: Central repository for all physical and mathematical constants
**Contains**:
- Physical constants (G, c, solar mass, Planck constant)
- Black hole specific constants (Schwarzschild coefficient, photon sphere radius)
- Numerical integration parameters (tolerance, max steps)
- Default simulation values

**Impact on Project**: Every physics calculation references these constants, ensuring consistency across all modules. Changes here affect accuracy of entire simulation.

### **black_hole.rs**
**Purpose**: Represents black hole objects with their physical properties
**Functions**:
- `BlackHole::new()` - Constructor with mass, spin, charge
- `schwarzschild_radius()` - Calculate event horizon size
- `ergosphere_radius()` - For rotating black holes (Kerr metric)
- `surface_gravity()` - Calculate gravitational field strength
- `temperature()` - Hawking radiation temperature
- `luminosity()` - Accretion disk brightness calculations

**Synergy**: Used by `spacetime.rs` for metric calculations, `geodesics.rs` for trajectory computations, and `objects.rs` for visual representation.

### **spacetime.rs**
**Purpose**: Handles spacetime geometry and coordinate transformations
**Functions**:
- `MetricTensor::schwarzschild()` - Schwarzschild metric components
- `MetricTensor::kerr()` - Rotating black hole metric
- `coordinate_transform()` - Convert between coordinate systems (Cartesian ↔ Schwarzschild ↔ Boyer-Lindquist)
- `spacetime_interval()` - Calculate proper time/distance
- `curvature_scalar()` - Riemann curvature calculations
- `redshift_factor()` - Gravitational redshift computation

**Synergy**: Provides geometric foundation for `geodesics.rs`, feeds data to `ray_tracer.rs` for light bending, used by `renderer.rs` for accurate visual effects.

### **geodesics.rs**
**Purpose**: Calculates trajectories of particles and light in curved spacetime
**Functions**:
- `GeodesicIntegrator::new()` - Initialize with integration method (RK4, Verlet)
- `integrate_photon_path()` - Trace light ray through spacetime
- `integrate_particle_orbit()` - Calculate massive particle trajectories
- `find_impact_parameter()` - Critical parameter for gravitational lensing
- `calculate_deflection_angle()` - Light bending angle
- `orbital_frequency()` - For orbiting matter in accretion disk

**Synergy**: Core engine for `ray_tracer.rs`, provides realistic trajectories for `objects.rs` (orbiting particles), feeds `animation.rs` for realistic motion.

---

## 🎨 **src/rendering/** - Graphics Pipeline

### **renderer.rs**
**Purpose**: Main rendering engine and GPU resource management
**Functions**:
- `Renderer::new()` - Initialize wgpu context, choose software/hardware backend
- `create_render_pipeline()` - Set up vertex/fragment shader pipeline
- `setup_uniform_buffers()` - Create GPU buffers for camera/physics data
- `render()` - Main render loop coordination
- `resize()` - Handle window resizing
- `update_uniforms()` - Send physics data to shaders each frame

**Synergy**: Orchestrates all rendering components, receives scene data from `scene.rs`, uses camera from `camera.rs`, manages shaders from `shaders.rs`.

### **camera.rs**
**Purpose**: 3D camera system with realistic controls
**Functions**:
- `Camera::new()` - Initialize with position, orientation, field of view
- `build_view_matrix()` - Create view transformation
- `build_projection_matrix()` - Perspective projection with frustum
- `CameraController::update()` - Handle user input for movement
- `screen_to_world_ray()` - Convert mouse position to 3D ray (crucial for ray tracing)
- `orbital_camera_mode()` - Orbit around black hole automatically

**Synergy**: Provides view matrices to `renderer.rs`, generates rays for `ray_tracer.rs`, receives input from `scene.rs`.

### **ray_tracer.rs**
**Purpose**: Implements ray tracing through curved spacetime
**Functions**:
- `RayTracer::new()` - Initialize with scene geometry
- `cast_primary_rays()` - Generate rays from camera through each pixel
- `trace_ray()` - Follow single ray through spacetime using physics from `geodesics.rs`
- `intersect_objects()` - Test ray intersection with scene objects
- `calculate_lighting()` - Compute final pixel color
- `parallel_trace()` - Multi-threaded ray tracing using rayon
- `adaptive_sampling()` - Reduce quality for performance when needed

**Synergy**: Uses `geodesics.rs` for accurate light paths, queries `objects.rs` for intersections, receives camera rays from `camera.rs`, outputs pixels to `renderer.rs`.

### **shaders.rs**
**Purpose**: Manages WGSL shaders and GPU programs
**Functions**:
- `ShaderManager::new()` - Load and compile shaders
- `load_vertex_shader()` - Basic 3D transformations
- `load_fragment_shader()` - Lighting and color calculations
- `load_compute_shader()` - Optional GPU-accelerated physics
- `hot_reload_shaders()` - Development feature for shader iteration
- `create_pipeline_layout()` - Define uniform buffer layouts

**Synergy**: Provides compiled shaders to `renderer.rs`, receives uniform data structure definitions from physics modules.

---

## 🌌 **src/simulation/** - Orchestration Layer

### **scene.rs**
**Purpose**: Central hub that manages entire simulation state
**Functions**:
- `Scene::new()` - Initialize with black hole, camera, objects
- `update()` - Main update loop called each frame
- `handle_input()` - Process keyboard/mouse input
- `update_physics()` - Step forward physics simulation
- `update_animations()` - Update time-based animations
- `get_render_data()` - Package data for renderer

**Synergy**: **CRITICAL ORCHESTRATOR** - connects all systems. Owns `BlackHole` from physics, `Camera` from rendering, coordinates `objects.rs` and `animation.rs`.

### **objects.rs**
**Purpose**: Manages all renderable objects in the scene
**Functions**:
- `AccretionDisk::new()` - Create disk geometry with temperature gradients
- `ParticleSystem::new()` - Manage orbiting particles
- `Skybox::new()` - Background stars/cosmic microwave background
- `update_disk_rotation()` - Animate based on physics from `geodesics.rs`
- `generate_disk_mesh()` - Create 3D geometry
- `calculate_disk_temperature()` - Physics-based color/brightness

**Synergy**: Uses `BlackHole` properties for realistic scaling, provides geometry to `renderer.rs`, gets motion data from `geodesics.rs`, animated by `animation.rs`.

### **animation.rs**
**Purpose**: Time-based animations and interpolations
**Functions**:
- `AnimationSystem::new()` - Initialize with time management
- `animate_accretion_disk()` - Rotate based on orbital mechanics
- `animate_particles()` - Move particles along geodesic paths
- `interpolate_camera_path()` - Smooth camera movements
- `update_time_dilation_effects()` - Visual effects based on relativity
- `animate_gravitational_waves()` - Future: spacetime ripple effects

**Synergy**: Receives physics calculations from `geodesics.rs`, updates object positions in `objects.rs`, provides smooth motion for visual appeal.

---

## 🔄 **Cross-Module Data Flow & Synergy**

### **Initialization Sequence**:
1. **constants.rs** → Provides values to all physics modules
2. **black_hole.rs** → Creates black hole with physical properties
3. **spacetime.rs** → Uses black hole mass to set up metric
4. **renderer.rs** → Initializes graphics pipeline
5. **camera.rs** → Sets up viewpoint at safe distance from black hole
6. **scene.rs** → Assembles everything into coherent simulation

### **Per-Frame Update Cycle**:
1. **scene.rs** `update()` called by main loop
2. **Input Processing**: `scene.rs` → `camera.rs` (movement) and physics parameters
3. **Physics Update**: `scene.rs` → `geodesics.rs` (particle positions) → `objects.rs` (visual updates)
4. **Animation Update**: `animation.rs` updates all time-dependent effects
5. **Rendering**: `scene.rs` → `renderer.rs` → `ray_tracer.rs` → `geodesics.rs` (for each ray) → final pixels

### **Critical Dependencies**:

**Physics → Rendering**:
- `geodesics.rs` provides light ray paths to `ray_tracer.rs`
- `black_hole.rs` properties determine visual scale in `renderer.rs`
- `spacetime.rs` curvature affects all visual distortion effects

**Rendering → Physics**:
- `camera.rs` position affects `geodesics.rs` ray starting points
- User input through `scene.rs` can modify black hole parameters

**Performance Critical Paths**:
- `ray_tracer.rs` → `geodesics.rs`: Called millions of times per frame
- `renderer.rs` uniform updates: Must be efficient for real-time performance
- `objects.rs` mesh generation: Can be cached and reused

### **Error Propagation**:
- Physics calculation errors bubble up through `scene.rs` to main loop
- Rendering errors handled in `renderer.rs` with graceful degradation
- Input errors caught in `scene.rs` before affecting physics

This architecture will help me ensure **separation of concerns** while maintaining **tight coupling** where performance demands it. Each module is gonna have a clear responsibility, but they work together seamlessly to create a scientifically accurate and visually impressive simulation.