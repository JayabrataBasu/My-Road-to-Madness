use crate::physics::black_hole::BlackHole;
use crate::rendering::camera::{Camera, CameraController, CameraMode};
use crate::simulation::objects::{BlackHoleObject, SimObject};
use crate::simulation::{InputState, TimeState};
use glam::Vec3;
use std::sync::{Arc, RwLock};
use winit::event::{DeviceEvent, KeyEvent, MouseScrollDelta, WindowEvent};

pub struct Scene {
    pub camera: Arc<RwLock<Camera>>,
    pub controller: CameraController,
    pub time: TimeState,
    pub input: InputState,
    pub objects: Vec<SimObject>,
}

impl Scene {
    pub fn new() -> Self {
        let bh = Arc::new(BlackHole::new_schwarzschild(10.0).expect("valid BH"));
        let camera = Arc::new(RwLock::new(Camera::new(
            Vec3::new(0.0, 2.0, 10.0),
            Vec3::ZERO,
            16.0 / 9.0,
        )));
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
        {
            let mut cam = self.camera.write().unwrap();
            self.controller.update(&self.input, self.time.delta_time, &mut cam);
            let (dx, dy) = self.input.mouse_delta;
            if dx.abs() > 0.0 || dy.abs() > 0.0 {
                self.controller.yaw -= dx; // invert for typical right-handed feel
                self.controller.pitch = (self.controller.pitch - dy).clamp(-1.553343, 1.553343); // ~ +/-89 deg
                cam.set_orientation_from_yaw_pitch(self.controller.yaw, self.controller.pitch);
            }
        }
        if let Some(fps) = self.time.fps_sample() { log::info!("FPS: {:.1}", fps); }
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
