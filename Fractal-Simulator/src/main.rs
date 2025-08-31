mod fractal;
mod mandelbrot;
mod pythagorean;
mod ui;

use fractal::Fractal;
use mandelbrot::Mandelbrot;
use pixels::{Error, Pixels, SurfaceTexture};
use pythagorean::PythagoreanTree;
use std::time::{Duration, Instant};
use ui::draw_text;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() -> Result<(), Error> {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Fractal Simulator - Mandelbrot")
        .with_inner_size(LogicalSize::new(800.0, 600.0))
        .with_min_inner_size(LogicalSize::new(320.0, 240.0))
        .build(&event_loop)
        .unwrap();

    let size = window.inner_size();
    let mut width = size.width;
    let mut height = size.height;
    let surface_texture = SurfaceTexture::new(width, height, &window);
    let mut pixels = Pixels::new(width, height, surface_texture)?;

    let mut fractal = FractalKind::Mandelbrot(Mandelbrot::new());
    let mut last_epoch = fractal.detail_epoch();
    const MAX_PASSES: u32 = 4; // 8x -> 4x -> 2x -> 1x
    let mut pass: u32 = 0;
    let mut need_clear = true;
    let mut last_redraw_instant = Instant::now();
    let mut final_pass_complete = false;

    let mut input_state = InputState::default();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll; // continuous rendering
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => {
                    handle_keyboard_input(input, &mut input_state, control_flow);
                }
                WindowEvent::Resized(new_size) => {
                    width = new_size.width.max(1);
                    height = new_size.height.max(1);
                    if let Err(e) = pixels.resize_surface(width, height) {
                        eprintln!("Resize surface error: {e}");
                    }
                    if let Err(e) = pixels.resize_buffer(width, height) {
                        eprintln!("Resize buffer error: {e}");
                    }
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    width = new_inner_size.width.max(1);
                    height = new_inner_size.height.max(1);
                    if let Err(e) = pixels.resize_surface(width, height) {
                        eprintln!("Scale factor resize surface error: {e}");
                    }
                    if let Err(e) = pixels.resize_buffer(width, height) {
                        eprintln!("Scale factor resize buffer error: {e}");
                    }
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                // Update simulation state
                apply_input(&mut fractal, &mut input_state, width, height, &mut need_clear);
                fractal.update(0.016); // assume ~60fps
                if fractal.detail_epoch() != last_epoch { pass = 0; need_clear = true; final_pass_complete = false; last_epoch = fractal.detail_epoch(); }
                // Only request redraw if we still refining or something changed recently or periodic heartbeat
                if !final_pass_complete || need_clear || input_state.show_help || last_redraw_instant.elapsed() > Duration::from_millis(1500) {
                    window.request_redraw();
                } else {
                    // Sleep a bit to reduce CPU when idle
                    *control_flow = ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(10));
                }
            }
            Event::RedrawRequested(_) => {
                let stride = 1 << (MAX_PASSES - 1 - pass);
                if need_clear { pixels.frame_mut().fill(0); need_clear = false; }
                fractal.render_partial(pixels.frame_mut(), width, height, stride);
                // Overlay UI
                let info = format!("{} | {} | pass {} stride {}", fractal.name(), fractal.info_string(), pass+1, stride);
                draw_text(pixels.frame_mut(), width, height, 5,5,&info, [255,255,255]);
                if input_state.show_help {
                    let help = "Controls: Arrows pan  +/- zoom  Tab switch fractal  H toggle help  Esc quit";
                    draw_text(pixels.frame_mut(), width, height, 5,18, help, [180,220,255]);
                }
                if stride > 1 { pass +=1; } else { final_pass_complete = true; }
                last_redraw_instant = Instant::now();
                if let Err(e) = pixels.render() {
                    eprintln!("pixels.render() failed: {e}");
                    *control_flow = ControlFlow::Exit;
                }
            }
            _ => {}
        }
    });
}

#[derive(Default)]
struct InputState {
    up: bool,
    down: bool,
    left: bool,
    right: bool,
    zoom_in: bool,
    zoom_out: bool,
    switch_next: bool,
    show_help: bool,
}

fn handle_keyboard_input(
    input: KeyboardInput,
    input_state: &mut InputState,
    control_flow: &mut ControlFlow,
) {
    if let Some(keycode) = input.virtual_keycode {
        let pressed = input.state == ElementState::Pressed;
        match keycode {
            VirtualKeyCode::Escape => {
                if pressed {
                    *control_flow = ControlFlow::Exit;
                }
            }
            VirtualKeyCode::Up => input_state.up = pressed,
            VirtualKeyCode::Down => input_state.down = pressed,
            VirtualKeyCode::Left => input_state.left = pressed,
            VirtualKeyCode::Right => input_state.right = pressed,
            VirtualKeyCode::Equals | VirtualKeyCode::Plus | VirtualKeyCode::NumpadAdd => {
                input_state.zoom_in = pressed
            }
            VirtualKeyCode::Minus | VirtualKeyCode::NumpadSubtract => {
                input_state.zoom_out = pressed
            }
            VirtualKeyCode::Tab => {
                if pressed {
                    input_state.switch_next = true;
                }
            }
            VirtualKeyCode::H => {
                if pressed {
                    input_state.show_help = !input_state.show_help;
                }
            }
            _ => {}
        }
    }
}

fn apply_input(
    fractal: &mut FractalKind,
    input: &mut InputState,
    width: u32,
    height: u32,
    need_clear: &mut bool,
) {
    let pan_factor = 0.02; // fraction of scale per frame when key held
    let aspect = width as f64 / height as f64;
    let mut dx = 0.0;
    let mut dy = 0.0;
    if input.left {
        dx -= pan_factor * aspect;
    }
    if input.right {
        dx += pan_factor * aspect;
    }
    if input.up {
        dy -= pan_factor;
    }
    if input.down {
        dy += pan_factor;
    }
    if dx != 0.0 || dy != 0.0 {
        fractal.pan(dx, dy);
        *need_clear = true;
    }
    if input.zoom_in {
        fractal.zoom(0.95);
        *need_clear = true;
    }
    if input.zoom_out {
        fractal.zoom(1.05);
        *need_clear = true;
    }
    if input.switch_next {
        *fractal = match fractal {
            FractalKind::Mandelbrot(_) => FractalKind::Pythagorean(PythagoreanTree::new()),
            FractalKind::Pythagorean(_) => FractalKind::Mandelbrot(Mandelbrot::new()),
        };
        *need_clear = true;
        input.switch_next = false;
    }
}

enum FractalKind {
    Mandelbrot(Mandelbrot),
    Pythagorean(PythagoreanTree),
}

impl FractalKind {
    fn with_fractal_mut<T>(&mut self, mut f: impl FnMut(&mut dyn Fractal) -> T) -> T {
        match self {
            Self::Mandelbrot(m) => f(m),
            Self::Pythagorean(p) => f(p),
        }
    }
    fn with_fractal<T>(&self, mut f: impl FnMut(&dyn Fractal) -> T) -> T {
        match self {
            Self::Mandelbrot(m) => f(m),
            Self::Pythagorean(p) => f(p),
        }
    }
}

impl Fractal for FractalKind {
    fn update(&mut self, dt: f32) {
        self.with_fractal_mut(|f| f.update(dt));
    }
    fn render_partial(&mut self, frame: &mut [u8], w: u32, h: u32, stride: u32) {
        self.with_fractal_mut(|f| f.render_partial(frame, w, h, stride));
    }
    fn pan(&mut self, dx: f64, dy: f64) {
        self.with_fractal_mut(|f| f.pan(dx, dy));
    }
    fn zoom(&mut self, factor: f64) {
        self.with_fractal_mut(|f| f.zoom(factor));
    }
    fn name(&self) -> &'static str {
        self.with_fractal(|f| f.name())
    }
    fn info_string(&self) -> String {
        self.with_fractal(|f| f.info_string())
    }
    fn detail_epoch(&self) -> u64 {
        self.with_fractal(|f| f.detail_epoch())
    }
}
