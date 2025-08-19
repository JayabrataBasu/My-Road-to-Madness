use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod physics;
mod rendering;
mod simulation;

use rendering::Renderer;
use simulation::Scene;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::init();

    // Create event loop and window
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Black Hole Simulation")
        .with_inner_size(winit::dpi::PhysicalSize::new(1024, 768))
        .build(&event_loop)?;

    // Initialize renderer and scene
    let mut renderer = pollster::block_on(Renderer::new(&window))?;
    let mut scene = Scene::new();

    println!("Black Hole Simulation Started");
    println!("Controls: Mouse to look around, WASD to move, Esc to quit");

    event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(size) => {
                        renderer.resize(size);
                    }
                    WindowEvent::RedrawRequested => {
                        // Update simulation
                        scene.update();

                        // Render frame
                        match renderer.render(&scene) {
                            Ok(_) => {}
                            Err(e) => eprintln!("Render error: {}", e),
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        scene.handle_input(&event);
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        scene.handle_mouse_move(position);
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
