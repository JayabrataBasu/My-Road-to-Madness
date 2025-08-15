use pixels::{Error, Pixels, SurfaceTexture};
use rand::Rng;
use std::time::{Duration, Instant};
use winit::{
    dpi::LogicalSize,
    event::{Event, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;

const WIDTH: u32 = 200;
const HEIGHT: u32 = 200;
const CELL_SIZE: u32 = 3;
const FPS: u64 = 10;

struct GameOfLife {
    grid: Vec<Vec<bool>>,
}

impl GameOfLife {
    fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        let grid = (0..HEIGHT)
            .map(|_| (0..WIDTH).map(|_| rng.gen_bool(0.2)).collect())
            .collect();
        Self { grid }
    }

    fn update(&mut self) {
        let mut new_grid = self.grid.clone();
        for y in 0..HEIGHT as usize {
            for x in 0..WIDTH as usize {
                let live_neighbors = self.live_neighbor_count(x, y);
                let cell = self.grid[y][x];
                new_grid[y][x] = matches!((cell, live_neighbors), (true, 2 | 3) | (false, 3));
            }
        }
        self.grid = new_grid;
    }

    fn live_neighbor_count(&self, x: usize, y: usize) -> u8 {
        let mut count = 0;
        for dy in [-1, 0, 1] {
            for dx in [-1, 0, 1] {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = (x as isize + dx).rem_euclid(WIDTH as isize) as usize;
                let ny = (y as isize + dy).rem_euclid(HEIGHT as isize) as usize;
                if self.grid[ny][nx] {
                    count += 1;
                }
            }
        }
        count
    }

    fn draw(&self, frame: &mut [u8]) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let x = (i % (WIDTH * CELL_SIZE) as usize) / CELL_SIZE as usize;
            let y = (i / (WIDTH * CELL_SIZE) as usize) / CELL_SIZE as usize;

            let rgba = if self.grid[y][x] {
                [0x00, 0xff, 0x00, 0xff] // Green
            } else {
                [0x00, 0x00, 0x00, 0xff] // Black
            };

            pixel.copy_from_slice(&rgba);
        }
    }
}

fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();

    let window = {
        let size = LogicalSize::new((WIDTH * CELL_SIZE) as f64, (HEIGHT * CELL_SIZE) as f64);
        WindowBuilder::new()
            .with_title("Conway's Game of Life - Rust")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH * CELL_SIZE, HEIGHT * CELL_SIZE, surface_texture)?
    };

    let mut game = GameOfLife::new_random();
    let mut last_update = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        if input.update(&event) {
            if input.key_pressed(VirtualKeyCode::Escape) || input.quit() {
                *control_flow = ControlFlow::Exit;
                return;
            }
        }

        if last_update.elapsed() >= Duration::from_millis(1000 / FPS) {
            game.update();
            last_update = Instant::now();
        }

        if let Event::RedrawRequested(_) = event {
            game.draw(pixels.frame_mut());
            if pixels.render().is_err() {
                *control_flow = ControlFlow::Exit;
            }
        }

        window.request_redraw();
    });
}
