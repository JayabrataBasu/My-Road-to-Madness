use crate::physics::black_hole::BlackHole;
use crate::rendering::camera::{Camera, CameraController, CameraMode};
use crate::rendering::ray_tracer::SamplePattern;
use crate::simulation::objects::{BlackHoleObject, SimObject};
use crate::simulation::{InputState, TimeState};
use glam::Vec3;
use parking_lot::RwLock;
use std::sync::Arc;
use winit::event::{DeviceEvent, KeyEvent, MouseScrollDelta, WindowEvent};

pub struct Scene {
    pub camera: Arc<RwLock<Camera>>,
    pub controller: CameraController,
    pub time: TimeState,
    pub input: InputState,
    pub objects: Vec<SimObject>,
    pub last_fps: Option<f32>,
    // Toggles
    pub show_hud: bool,
    pub show_center_geodesic: bool,
    pub paused: bool,
    pub jitter: bool,
    pub samples: u32,
    pub sample_pattern: SamplePattern,
    pub disk_outer_scale: f32,
    pub disk_thickness_scale: f32,
    pub star_brightness: f32,
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
            last_fps: None,
            show_hud: true,
            show_center_geodesic: true,
            paused: false,
            jitter: true,
            samples: 0,
            sample_pattern: SamplePattern::Halton,
            disk_outer_scale: 1.0,
            disk_thickness_scale: 1.0,
            star_brightness: 1.0,
        }
    }

    pub fn update(&mut self) {
        if !self.paused {
            self.time.update();
        }
        // Handle one-shot star brightness toggle
        if self.input.toggle_star_brightness {
            self.star_brightness = if (self.star_brightness - 1.0).abs() < 0.01 {
                0.3
            } else {
                1.0
            };
            self.input.toggle_star_brightness = false;
        }
        // Adjust disk parameters
        if self.input.increase_disk_radius {
            self.disk_outer_scale = (self.disk_outer_scale * 1.02).min(5.0);
        }
        if self.input.decrease_disk_radius {
            self.disk_outer_scale = (self.disk_outer_scale / 1.02).max(0.2);
        }
        if self.input.increase_disk_thickness {
            self.disk_thickness_scale = (self.disk_thickness_scale * 1.02).min(5.0);
        }
        if self.input.decrease_disk_thickness {
            self.disk_thickness_scale = (self.disk_thickness_scale / 1.02).max(0.2);
        }
        // Update camera
        // For now create a mutable reference clone of Arc (temporarily clone underlying). This will change when refactoring to interior mutability if needed.
        {
            let mut cam = self.camera.write();
            self.controller
                .update(&self.input, self.time.delta_time, &mut cam);
            let (dx, dy) = self.input.mouse_delta;
            if dx.abs() > 0.0 || dy.abs() > 0.0 {
                self.controller.yaw -= dx; // invert for typical right-handed feel
                self.controller.pitch = (self.controller.pitch - dy).clamp(-1.553343, 1.553343); // ~ +/-89 deg
                cam.set_orientation_from_yaw_pitch(self.controller.yaw, self.controller.pitch);
            }
        }
        if let Some(fps) = self.time.fps_sample() {
            log::info!("FPS: {:.1}", fps);
            self.last_fps = Some(fps);
        }
        self.input.reset_mouse_delta();
    }

    pub fn handle_keyboard(&mut self, event: &KeyEvent) {
        self.input.handle_keyboard(event);
    }

    pub fn handle_window_event(&mut self, event: &WindowEvent) {
        if let WindowEvent::KeyboardInput { event, .. } = event {
            use winit::keyboard::{KeyCode, PhysicalKey};
            if let PhysicalKey::Code(code) = event.physical_key {
                if event.state == winit::event::ElementState::Pressed {
                    match code {
                        KeyCode::KeyH => {
                            self.show_hud = !self.show_hud;
                        }
                        KeyCode::KeyG => {
                            self.show_center_geodesic = !self.show_center_geodesic;
                        }
                        KeyCode::KeyP => {
                            self.paused = !self.paused;
                        }
                        KeyCode::KeyJ => {
                            self.jitter = !self.jitter;
                        }
                        KeyCode::KeyB => {
                            // cycle sample pattern
                            self.sample_pattern = match self.sample_pattern {
                                SamplePattern::Halton => SamplePattern::BlueNoise,
                                SamplePattern::BlueNoise => SamplePattern::HaltonBlueCombine,
                                SamplePattern::HaltonBlueCombine => SamplePattern::Halton,
                            };
                        }
                        KeyCode::KeyR => {
                            // recenter camera to a hardcoded HUD-like position and look at origin
                            let mut cam = self.camera.write();
                            let target = glam::Vec3::ZERO;
                            let pos = glam::Vec3::new(0.5, 1.0, 0.02); // Example HUD position
                            let up = glam::Vec3::Y;
                            cam.position = pos;
                            let dir = (target - cam.position).normalize();
                            let right = dir.cross(up).normalize();
                            let up_vec = right.cross(dir).normalize();
                            cam.forward = dir;
                            cam.right = right;
                            cam.up = up_vec;
                            self.controller.yaw = 0.0;
                            self.controller.pitch = 0.0;
                            cam.mark_changed();
                        }
                        KeyCode::KeyO => {
                            // Toggle orbit / free
                            self.controller.mode = match self.controller.mode {
                                CameraMode::Free => CameraMode::Orbit,
                                CameraMode::Orbit => CameraMode::Free,
                            };
                        }
                        _ => {}
                    }
                }
            }
            self.handle_keyboard(event);
        } else if let WindowEvent::MouseWheel { delta, .. } = event {
            if let MouseScrollDelta::LineDelta(_, y) = delta {
                self.controller.speed = (self.controller.speed + y).clamp(0.1, 20.0);
            }
        }
    }

    pub fn handle_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.input.mouse_delta.0 += delta.0 as f32;
            self.input.mouse_delta.1 += delta.1 as f32;
        }
    }

    pub fn hud_text(&self) -> String {
        let cam = self.camera.read();
        let fps = self.last_fps.unwrap_or(0.0);
        if !self.show_hud {
            return String::new();
        }
        let pattern = match self.sample_pattern {
            SamplePattern::Halton => "Halton",
            SamplePattern::BlueNoise => "Blue",
            SamplePattern::HaltonBlueCombine => "Hybrid",
        };
        format!(
            "FPS: {:.1}{}\nSamples: {}  Jitter: {}  Pattern[B]: {}\nPos: ({:.1}, {:.1}, {:.1})\nDir: ({:.2}, {:.2}, {:.2})\nSpeed: {:.1}  Mode[O]: {:?}  Roll(Q/E): {:.2}\nDisk R[1/2]: {:.2}  Thick[3/4]: {:.2}  Stars[V]: {:.1}\n[G] Center Geod: {}  [H] HUD  [P] Pause: {}  [J] Jitter",
            fps,
            if self.paused { " (PAUSED)" } else { "" },
            self.samples,
            if self.jitter { "ON" } else { "OFF" },
            pattern,
            cam.position.x,
            cam.position.y,
            cam.position.z,
            cam.forward.x,
            cam.forward.y,
            cam.forward.z,
            self.controller.speed,
            self.controller.mode,
            self.controller.roll,
            self.disk_outer_scale,
            self.disk_thickness_scale,
            self.star_brightness,
            if self.show_center_geodesic {
                "ON"
            } else {
                "OFF"
            },
            if self.paused { "ON" } else { "OFF" }
        )
    }
}
