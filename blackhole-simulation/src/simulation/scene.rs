use crate::physics::black_hole::BlackHole;
use crate::rendering::camera::{Camera, CameraController, CameraMode};
use crate::simulation::objects::{BlackHoleObject, SimObject};
use crate::simulation::{InputState, TimeState};
use glam::Vec3;
use std::sync::Arc;
use winit::event::{DeviceEvent, KeyEvent, MouseScrollDelta, WindowEvent};

pub struct Scene {
    pub camera: Arc<Camera>,
    pub controller: CameraController,
    pub time: TimeState,
    pub input: InputState,
    pub objects: Vec<SimObject>,
}

impl Scene {
    pub fn new() -> Self {
        let bh = Arc::new(BlackHole::new_schwarzschild(10.0).expect("valid BH"));
        let camera = Arc::new(Camera::new(
            Vec3::new(0.0, 2.0, 10.0),
            Vec3::ZERO,
            16.0 / 9.0,
        ));
        let controller = CameraController::new(CameraMode::Free, Vec3::ZERO);
        let mut objects = Vec::new();
        objects.push(SimObject::BlackHole(BlackHoleObject::new(
            bh.clone(),
            Vec3::ZERO,
        )));
        Self {
            camera,
            controller,
            time: TimeState::default(),
            input: InputState::default(),
            objects,
        }
    }

    pub fn update(&mut self) {
        self.time.update();
        // Update camera
        // For now create a mutable reference clone of Arc (temporarily clone underlying). This will change when refactoring to interior mutability if needed.
        if let Some(cam_mut) = Arc::get_mut(&mut self.camera) {
            self.controller
                .update(&self.input, self.time.delta_time, cam_mut);
            // Apply mouse look if any delta
            let (dx, dy) = self.input.mouse_delta;
            if dx.abs() > 0.0 || dy.abs() > 0.0 {
                let yaw_delta = dx;
                let pitch_delta = dy;
                // Reconstruct right from forward/up
                let right = cam_mut.forward.cross(cam_mut.up).normalize();
                // Yaw: rotate forward around world up (Y)
                let rot_yaw = glam::Quat::from_axis_angle(glam::Vec3::Y, -yaw_delta);
                let rot_pitch = glam::Quat::from_axis_angle(right, -pitch_delta);
                let new_forward = (rot_yaw * rot_pitch * cam_mut.forward).normalize();
                cam_mut.forward = new_forward;
                cam_mut.right = cam_mut.forward.cross(glam::Vec3::Y).normalize();
                cam_mut.up = cam_mut.right.cross(cam_mut.forward).normalize();
                cam_mut.mark_changed();
            }
        }
        self.input.reset_mouse_delta();
    }

    pub fn handle_keyboard(&mut self, event: &KeyEvent) {
        self.input.handle_keyboard(event);
    }

    pub fn handle_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => self.handle_keyboard(event),
            WindowEvent::CursorMoved { position: _, .. } => { /* TODO: accumulate mouse delta via DeviceEvent */
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let MouseScrollDelta::LineDelta(_, y) = delta {
                    self.controller.speed = (self.controller.speed + y).clamp(1.0, 200.0);
                }
            }
            _ => {}
        }
    }

    pub fn handle_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.input.mouse_delta.0 += delta.0 as f32;
            self.input.mouse_delta.1 += delta.1 as f32;
        }
    }
}
