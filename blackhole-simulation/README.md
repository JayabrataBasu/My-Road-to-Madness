# Black Hole Simulation - Component Overview

## 🎯 Physics Module (`src/physics/`)

### `mod.rs`
- Exports all physics-related functionality
- Defines common physics types and traits
- Re-exports constants and utilities

### `constants.rs`
- Physical constants (G, c, solar mass, etc.)
- Conversion factors and units
- Mathematical constants for calculations

### `black_hole.rs`
- **Core Responsibility**: Black hole physics and properties
- Schwarzschild and Kerr metrics
- Event horizon calculations
- Mass, spin, and charge parameters
- Gravitational field strength computation

### `spacetime.rs`
- **Core Responsibility**: Spacetime geometry and curvature
- Metric tensor calculations
- Coordinate transformations (Cartesian ↔ Schwarzschild)
- Spacetime interval computations
- Curvature effects on light and matter

### `geodesics.rs`
- **Core Responsibility**: Particle and light ray trajectories
- Geodesic equation integration
- Ray tracing through curved spacetime
- Gravitational lensing calculations
- Photon orbit computations

---

## 🎨 Rendering Module (`src/rendering/`)

### `mod.rs`
- Rendering system exports and common types
- Graphics pipeline abstractions
- Shader management utilities

### `renderer.rs`
- **Core Responsibility**: Main rendering engine
- wgpu setup and management
- Render pipeline creation
- Frame rendering orchestration
- GPU resource management

### `camera.rs`
- **Core Responsibility**: 3D camera system
- View and projection matrices
- Camera movement and rotation
- Frustum calculations
- Screen-to-world coordinate conversion

### `ray_tracer.rs`
- **Core Responsibility**: Ray tracing implementation
- Ray generation from camera
- Ray-object intersection testing
- Lighting calculations
- Color accumulation and sampling

### `shaders.rs`
- **Core Responsibility**: Shader compilation and management
- WGSL shader loading
- Pipeline state objects
- Uniform buffer management
- Shader hot-reloading (development)

---

## 🌌 Simulation Module (`src/simulation/`)

### `mod.rs`
- Simulation system exports
- Time management utilities
- Common simulation types

### `scene.rs`
- **Core Responsibility**: Scene management and state
- Object hierarchy management
- Camera controller
- Input handling
- Frame update logic

### `objects.rs`
- **Core Responsibility**: Renderable objects in the scene
- Black hole representation
- Accretion disk geometry
- Particle systems
- Background stars/skybox

### `animation.rs`
- **Core Responsibility**: Time-based animations
- Accretion disk rotation
- Particle motion simulation
- Camera path interpolation
- Physics-based animations

---

## 📁 Assets Directory (`assets/`)

### `shaders/`
- **vertex.wgsl**: Vertex shader for 3D transformations
- **fragment.wgsl**: Fragment shader for lighting and effects
- **compute.wgsl**: Optional compute shaders for physics

### `textures/`
- Accretion disk textures
- Star field backgrounds
- Normal maps and effects

---

## 🚀 Development Plan Overview

### Phase 1: Foundation (Week 1-2)
1. **Setup basic window and rendering context**
   - Initialize wgpu with CPU backend
   - Create basic vertex/fragment shaders
   - Set up camera system

2. **Implement core physics types**
   - Define BlackHole struct with basic properties
   - Implement coordinate transformations
   - Create spacetime metric calculations

### Phase 2: Basic Visualization (Week 3-4)
1. **Simple black hole rendering**
   - Render black sphere for event horizon
   - Implement basic camera controls
   - Add simple lighting

2. **Ray tracing foundation**
   - Implement basic ray generation
   - Add sphere intersection testing
   - Simple color calculations

### Phase 3: Physics Integration (Week 5-6)
1. **Gravitational effects**
   - Implement geodesic ray tracing
   - Add gravitational lensing
   - Light bending visualization

2. **Accretion disk**
   - Create disk geometry
   - Add rotation animation
   - Temperature-based coloring

### Phase 4: Advanced Effects (Week 7-8)
1. **Visual enhancements**
   - Gravitational redshift
   - Doppler effects
   - Bloom and post-processing

2. **Performance optimization**
   - Parallel ray tracing
   - Level-of-detail systems
   - Adaptive quality rendering

### Phase 5: Polish and Features (Week 9-10)
1. **Interactive elements**
   - Adjustable black hole parameters
   - Different viewing modes
   - Educational overlays

2. **Final optimization**
   - Profile and optimize hotspots
   - Implement caching systems
   - Add configuration options

---

