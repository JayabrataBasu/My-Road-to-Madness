use pixels::{Pixels, SurfaceTexture};
use rayon::prelude::*;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

// --- Constants ---
const WIDTH: usize = 512;
const HEIGHT: usize = 512;

// --- Simulation State and Parameters ---
struct Simulation {
    u_grid: Vec<f64>,
    v_grid: Vec<f64>,
    // Parameters for the Gray-Scott model
    f_rate: f64,
    k_rate: f64,
    diff_u: f64,
    diff_v: f64,
    delta_t: f64, // Time step
}

// --- Simulation Logic Implementation ---
impl Simulation {
    /// Creates a new simulation with a default initial state.
    fn new() -> Self {
        let mut u_grid = vec![1.0; WIDTH * HEIGHT];
        let v_grid = vec![0.0; WIDTH * HEIGHT];

        // Seed the simulation with a block of U and V
        let seed_size = 20;
        let start_x = WIDTH / 2 - seed_size / 2;
        let start_y = HEIGHT / 2 - seed_size / 2;

        for y in start_y..(start_y + seed_size) {
            for x in start_x..(start_x + seed_size) {
                let index = y * WIDTH + x;
                if index < u_grid.len() {
                    u_grid[index] = 0.5;
                    // v_grid[index] = 0.25; // This is now done in the main initialization
                }
            }
        }

        let mut sim = Self {
            u_grid,
            v_grid,
            // "Mitosis" / Coral-like growth preset
            f_rate: 0.0545,
            k_rate: 0.062,
            diff_u: 1.0,
            diff_v: 0.5,
            delta_t: 1.0,
        };

        // Initialize the V grid with the seeded U values
        sim.v_grid = sim
            .u_grid
            .iter()
            .map(|&u| if u < 1.0 { 0.25 } else { 0.0 })
            .collect();

        sim
    }

    /// Calculates the 2D Laplacian for a given grid and coordinates.
    fn laplacian(&self, grid: &[f64], x: usize, y: usize) -> f64 {
        let mut sum = 0.0;
        // Center weight
        sum += grid[y * WIDTH + x] * -1.0;
        // Adjacent neighbors
        sum += grid[y * WIDTH + (x - 1)] * 0.2;
        sum += grid[y * WIDTH + (x + 1)] * 0.2;
        sum += grid[(y - 1) * WIDTH + x] * 0.2;
        sum += grid[(y + 1) * WIDTH + x] * 0.2;
        // Diagonal neighbors
        sum += grid[(y - 1) * WIDTH + (x - 1)] * 0.05;
        sum += grid[(y - 1) * WIDTH + (x + 1)] * 0.05;
        sum += grid[(y + 1) * WIDTH + (x - 1)] * 0.05;
        sum += grid[(y + 1) * WIDTH + (x + 1)] * 0.05;
        sum
    }

    /// Updates the state of the simulation by one time step.
    fn update(&mut self) {
        let mut next_u = self.u_grid.clone();
        let mut next_v = self.v_grid.clone();

        // Use Rayon to parallelize the computation over the grid rows
        next_u
            .par_chunks_mut(WIDTH)
            .zip(next_v.par_chunks_mut(WIDTH))
            .enumerate()
            .for_each(|(y, (u_row, v_row))| {
                // Skip the borders to avoid out-of-bounds access
                if y == 0 || y == HEIGHT - 1 {
                    return;
                }
                for x in 1..(WIDTH - 1) {
                    let u = self.u_grid[y * WIDTH + x];
                    let v = self.v_grid[y * WIDTH + x];

                    let laplace_u = self.laplacian(&self.u_grid, x, y);
                    let laplace_v = self.laplacian(&self.v_grid, x, y);

                    let reaction = u * v * v;

                    let u_prime = self.diff_u * laplace_u - reaction + self.f_rate * (1.0 - u);
                    let v_prime =
                        self.diff_v * laplace_v + reaction - (self.k_rate + self.f_rate) * v;

                    u_row[x] = (u + u_prime * self.delta_t).max(0.0).min(1.0);
                    v_row[x] = (v + v_prime * self.delta_t).max(0.0).min(1.0);
                }
            });

        self.u_grid = next_u;
        self.v_grid = next_v;
    }

    /// Draws the simulation state to the provided pixel frame buffer.
    fn draw(&self, frame: &mut [u8]) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            // Visualize the V concentration as grayscale
            let v = self.v_grid[i];
            let color = (v * 255.0) as u8;

            let rgba = [color, color, color, 0xff]; // R, G, B, A
            pixel.copy_from_slice(&rgba);
        }
    }
}

// --- Main Application Entry Point ---
fn main() -> Result<(), pixels::Error> {
    let event_loop = EventLoop::new().unwrap();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Gray-Scott Simulation in Rust 🦀")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH as u32, HEIGHT as u32, surface_texture)?
    };

    let mut simulation = Simulation::new();

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    elwt.exit();
                }
                Event::AboutToWait => {
                    // Update the simulation state multiple times per frame for stability
                    for _ in 0..4 {
                        simulation.update();
                    }
                    window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    simulation.draw(pixels.frame_mut());
                    if let Err(err) = pixels.render() {
                        eprintln!("pixels.render() failed: {err}");
                        elwt.exit();
                    }
                }
                _ => (),
            }
        })
        .unwrap();

    Ok(())
}
