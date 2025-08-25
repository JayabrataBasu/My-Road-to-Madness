//! Physics root module: exposes constants, black hole model, spacetime metrics
//! and geodesic integration utilities. All physics uses f64 and geometric
//! units internally (G = c = 1 after normalization).

pub mod constants;      // Fundamental & numerical constants
pub mod black_hole;     // BlackHole struct & derived properties
pub mod spacetime;      // Metric tensors & coordinate transforms
pub mod geodesics;      // Geodesic data structures & integrators

// Re-export commonly accessed symbols for ergonomic downstream use.

// Type aliases (optional convenience)
pub type Vec3 = nalgebra::Vector3<f64>;
pub type Vec4 = nalgebra::Vector4<f64>;
pub type Mat4 = nalgebra::Matrix4<f64>;

