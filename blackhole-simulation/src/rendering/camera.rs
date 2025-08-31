#![allow(dead_code)]
use glam::{Mat4, Vec3};

/// Primary perspective camera. Right-handed system; looks down -Z.
pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub right: Vec3,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    pub version: u64,
}

impl Camera {
    pub fn new(position: Vec3, target: Vec3, aspect: f32) -> Self {
        let fov_y = 60f32.to_radians();
        let forward = (target - position).normalize_or_zero();
        let world_up = Vec3::Y;
        let right = forward.cross(world_up).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        Self {
            position,
            forward,
            up,
            right,
            fov_y,
            aspect,
            near: 0.01,
            far: 10_000.0,
            version: 0,
        }
    }

    #[inline]
    pub fn position(&self) -> Vec3 {
        self.position
    }
    #[inline]
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward, self.up)
    }
    #[inline]
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect.max(1e-6), self.near, self.far)
    }
    #[inline]
    pub fn view_projection(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Convert integer pixel coordinates to a world-space ray (origin, dir).
    pub fn screen_to_world_ray(&self, x: u32, y: u32, width: u32, height: u32) -> (Vec3, Vec3) {
        let nx = ((x as f32 + 0.5) / width as f32) * 2.0 - 1.0;
        let ny = 1.0 - ((y as f32 + 0.5) / height as f32) * 2.0; // flip Y
        let tan_half = (self.fov_y * 0.5).tan();
        let dx = nx * tan_half * self.aspect;
        let dy = ny * tan_half;
        let dir_cam = Vec3::new(dx, dy, -1.0).normalize();
        // Camera space to world space
        let dir_world =
            (self.right * dir_cam.x + self.up * dir_cam.y + self.forward * dir_cam.z).normalize();
        (self.position, dir_world)
    }

    /// Jittered subpixel ray: ox, oy in [0,1) offset inside pixel for sampling.
    pub fn screen_to_world_ray_offset(
        &self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        ox: f32,
        oy: f32,
    ) -> (Vec3, Vec3) {
        let nx = ((x as f32 + ox) / width as f32) * 2.0 - 1.0;
        let ny = 1.0 - ((y as f32 + oy) / height as f32) * 2.0; // flip Y
        let tan_half = (self.fov_y * 0.5).tan();
        let dx = nx * tan_half * self.aspect;
        let dy = ny * tan_half;
        let dir_cam = Vec3::new(dx, dy, -1.0).normalize();
        let dir_world =
            (self.right * dir_cam.x + self.up * dir_cam.y + self.forward * dir_cam.z).normalize();
        (self.position, dir_world)
    }

    #[inline]
    pub fn mark_changed(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    pub fn set_orientation_from_yaw_pitch(&mut self, yaw: f32, pitch: f32) {
        let cp = pitch.cos();
        let sp = pitch.sin();
        let cy = yaw.cos();
        let sy = yaw.sin();
        self.forward = glam::Vec3::new(cy * cp, sp, -sy * cp).normalize();
        self.right = self.forward.cross(glam::Vec3::Y).normalize();
        self.up = self.right.cross(self.forward).normalize();
        self.mark_changed();
    }
}

pub enum CameraMode {
    Free,
    Orbit,
}

pub struct CameraController {
    pub mode: CameraMode,
    pub orbit_target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
    pub sensitivity: f32,
}

impl CameraController {
    pub fn new(mode: CameraMode, orbit_target: Vec3) -> Self {
        Self {
            mode,
            orbit_target,
            yaw: 0.0,
            pitch: 0.0,
            speed: 2.0,
            sensitivity: 0.002,
        }
    }

    pub fn update(&mut self, input: &crate::simulation::InputState, dt: f32, cam: &mut Camera) {
        match self.mode {
            CameraMode::Free => self.update_free(input, dt, cam),
            CameraMode::Orbit => self.update_orbit(dt, cam),
        }
    }

    fn update_free(&mut self, input: &crate::simulation::InputState, dt: f32, cam: &mut Camera) {
        let mut dir = Vec3::ZERO;
        if input.forward {
            dir += cam.forward;
        }
        if input.backward {
            dir -= cam.forward;
        }
        if input.right {
            dir += cam.right;
        }
        if input.left {
            dir -= cam.right;
        }
        if input.up {
            dir += cam.up;
        }
        if input.down {
            dir -= cam.up;
        }
        if dir.length_squared() > 0.0 {
            cam.position += dir.normalize() * self.speed * dt;
            cam.mark_changed();
        }
    }

    fn update_orbit(&mut self, dt: f32, cam: &mut Camera) {
        let radius = (cam.position - self.orbit_target).length().max(0.1);
        self.yaw += dt * 0.2;
        let x = self.orbit_target.x + radius * self.yaw.cos();
        let z = self.orbit_target.z + radius * self.yaw.sin();
        cam.position = Vec3::new(x, cam.position.y, z);
        cam.forward = (self.orbit_target - cam.position).normalize();
        cam.right = cam.forward.cross(Vec3::Y).normalize();
        cam.up = cam.right.cross(cam.forward).normalize();
        cam.mark_changed();
    }
}
