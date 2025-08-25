use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

mod physics;
mod rendering;
mod simulation;

use rendering::{RenderQuality, Renderer};
use simulation::Scene;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Black Hole Simulation")
        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
        .build(&event_loop)?;
    let window = std::sync::Arc::new(window);

    // Build scene
    let mut scene = Scene::new();
    use simulation::objects::SimObject;
    let bh_arc = match &scene.objects[0] {
        SimObject::BlackHole(bhobj) => bhobj.bh.clone(),
    };
    let camera_arc = scene.camera.clone();
    let mut renderer = pollster::block_on(Renderer::new(
        &window,
        RenderQuality::High,
        camera_arc,
        bh_arc,
    ))?;

    println!("Black Hole Simulation Started");
    println!("Controls: WASD move, wheel = speed, Esc = close (window close)");

    let win_id = window.id();
    let win_clone = window.clone();
    event_loop.run(move |event, target| match event {
        Event::WindowEvent { event, window_id } if window_id == win_id => match event {
            WindowEvent::CloseRequested => target.exit(),
            WindowEvent::Resized(size) => renderer.resize(size),
            WindowEvent::RedrawRequested => {
                scene.update();
                // Sync renderer jitter with scene toggle
                renderer.set_jitter(scene.jitter);
                renderer.set_sample_pattern(scene.sample_pattern);
                let hud = scene.hud_text();
                if let Err(e) = renderer.render(Some(&hud), scene.show_center_geodesic) {
                    eprintln!("Render error: {e}");
                }
                scene.samples = renderer.sample_count();
            }
            _ => scene.handle_window_event(&event),
        },
        Event::DeviceEvent { event, .. } => scene.handle_device_event(&event),
        Event::AboutToWait => win_clone.request_redraw(),
        _ => {}
    })?;
    Ok(())
}
