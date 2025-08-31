//! Simulation module for scene management and animation
//!
//! This module ties together physics and rendering, managing the overall
//! simulation state, user interaction, and time-based animations.

pub mod objects;
pub mod scene;

// Re-export scene only
pub use scene::Scene;

use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Input handling utilities
pub struct InputState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub roll_left: bool,
    pub roll_right: bool,
    pub mouse_delta: (f32, f32),
    pub increase_disk_radius: bool,
    pub decrease_disk_radius: bool,
    pub increase_disk_thickness: bool,
    pub decrease_disk_thickness: bool,
    pub toggle_star_brightness: bool,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            roll_left: false,
            roll_right: false,
            mouse_delta: (0.0, 0.0),
            increase_disk_radius: false,
            decrease_disk_radius: false,
            increase_disk_thickness: false,
            decrease_disk_thickness: false,
            toggle_star_brightness: false,
        }
    }
}

impl InputState {
    pub fn handle_keyboard(&mut self, event: &KeyEvent) {
        let pressed = event.state == ElementState::Pressed;

        if let PhysicalKey::Code(keycode) = event.physical_key {
            match keycode {
                KeyCode::KeyW | KeyCode::ArrowUp => self.forward = pressed,
                KeyCode::KeyS | KeyCode::ArrowDown => self.backward = pressed,
                KeyCode::KeyA | KeyCode::ArrowLeft => self.left = pressed,
                KeyCode::KeyD | KeyCode::ArrowRight => self.right = pressed,
                KeyCode::Space => self.up = pressed,
                KeyCode::ShiftLeft | KeyCode::ShiftRight => self.down = pressed,
                KeyCode::KeyQ => self.roll_left = pressed,
                KeyCode::KeyE => self.roll_right = pressed,
                KeyCode::Digit1 => self.decrease_disk_radius = pressed,
                KeyCode::Digit2 => self.increase_disk_radius = pressed,
                KeyCode::Digit3 => self.decrease_disk_thickness = pressed,
                KeyCode::Digit4 => self.increase_disk_thickness = pressed,
                KeyCode::KeyV => if pressed { self.toggle_star_brightness = true; },
                _ => {}
            }
        }
    }

    // Removed unused handle_mouse_move & sensitivity

    pub fn reset_mouse_delta(&mut self) {
        self.mouse_delta = (0.0, 0.0);
    }
}

/// Time management for animations and physics
#[derive(Debug, Clone)]
pub struct TimeState {
    pub current_time: f32,
    pub delta_time: f32,
    pub last_frame_time: std::time::Instant,
    pub simulation_speed: f32,
    pub frame_count: u64,
    pub last_fps_instant: std::time::Instant,
}

impl Default for TimeState {
    fn default() -> Self {
        Self {
            current_time: 0.0,
            delta_time: 0.0,
            last_frame_time: std::time::Instant::now(),
            simulation_speed: 1.0,
            frame_count: 0,
            last_fps_instant: std::time::Instant::now(),
        }
    }
}

impl TimeState {
    pub fn update(&mut self) {
        let now = std::time::Instant::now();
        self.delta_time =
            now.duration_since(self.last_frame_time).as_secs_f32() * self.simulation_speed;
        self.current_time += self.delta_time;
        self.last_frame_time = now;
        self.frame_count += 1;
    }

    pub fn fps_sample(&mut self) -> Option<f32> {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_fps_instant).as_secs_f32();
        if elapsed >= 1.0 {
            let fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.last_fps_instant = now;
            Some(fps)
        } else {
            None
        }
    }
}
