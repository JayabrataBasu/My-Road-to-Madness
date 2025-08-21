pub struct Camera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(
        position: [f32; 3],
        target: [f32; 3],
        up: [f32; 3],
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Camera {
            position,
            target,
            up,
            fov_y,
            aspect,
            near,
            far,
        }
    }

    pub fn build_view_matrix(&self) -> [[f32; 4]; 4] {
        // Minimal look-at matrix (stub)
        // In practice, use nalgebra/glam or similar for real math
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    pub fn build_projection_matrix(&self) -> [[f32; 4]; 4] {
        // Minimal perspective projection (stub)
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}

pub struct CameraController {
    pub enabled: bool,
}

impl CameraController {
    pub fn new() -> Self {
        CameraController { enabled: true }
    }

    pub fn update(&mut self, camera: &mut Camera) {
        if self.enabled {
            // Stub: update camera based on input
            camera.position[0] += 0.0; // Use camera to avoid warning
        }
    }
}

/// Convert screen coordinates to a world ray (stub)
pub fn screen_to_world_ray(
    camera: &Camera,
    screen_x: f32,
    screen_y: f32,
    screen_width: f32,
    screen_height: f32,
) -> ([f32; 3], [f32; 3]) {
    // Normalize screen coordinates to [-1, 1]
    let _norm_x = (screen_x / screen_width) * 2.0 - 1.0;
    let _norm_y = (screen_y / screen_height) * 2.0 - 1.0;

    // Return origin and direction (stub)
    (camera.position, [0.0, 0.0, -1.0])
}

/// Orbit camera around a point (stub)
pub fn orbital_camera_mode(camera: &mut Camera, center: [f32; 3], angle: f32) {
    // Calculate new position based on angle
    let radius = ((camera.position[0] - center[0]).powi(2)
        + (camera.position[2] - center[2]).powi(2))
    .sqrt();

    camera.position[0] = center[0] + radius * angle.cos();
    camera.position[2] = center[2] + radius * angle.sin();
}
