# Matrixio - High-Performance Matrix Calculator

A robust, high-performance matrix calculation software designed to handle large-scale matrix operations with an intuitive user interface.

## Features

### Core Capabilities
- **Large Matrix Support**: Handle matrices well beyond the typical 3x3 or 4x4 limitations
- **High Performance**: Optimized with BLAS/LAPACK backends and optional GPU acceleration
- **Intuitive UI**: Clean, responsive interface built with egui
- **Multi-threading**: Parallel computation for large matrix operations
- **Memory Efficient**: Smart memory management for handling large datasets

### Matrix Operations
- Basic Operations: Addition, Subtraction, Multiplication, Division
- Advanced Operations: Inverse, Determinant, Eigenvalues/Eigenvectors
- Decompositions: LU, QR, SVD, Cholesky
- Specialized Functions: Transpose, Trace, Rank, Condition Number
- Element-wise Operations: Power, Exponential, Logarithm, Trigonometric

### Data Management
- Import/Export: CSV, JSON formats
- Save/Load: Project files with matrix collections
- Real-time Visualization: Matrix heatmaps and structure visualization

## Architecture

### Core Components
1. **Matrix Engine** (`src/matrix/`): High-performance matrix operations
2. **UI Layer** (`src/ui/`): User interface and interaction handling
3. **GPU Backend** (`src/gpu/`): Optional GPU acceleration
4. **Data Layer** (`src/data/`): File I/O and serialization

### Performance Optimizations
- BLAS/LAPACK integration for linear algebra operations
- SIMD vectorization where applicable
- Multi-threaded operations using Rayon
- Memory pool allocation for large matrices
- Optional GPU compute shaders for parallel operations

## Usage

```bash
# Build and run
cargo run --release

# Run with GPU acceleration (if supported)
cargo run --release --features gpu

# Run tests
cargo test

# Benchmark performance
cargo bench
```

## System Requirements

### Minimum
- 4GB RAM
- Dual-core processor
- 100MB disk space

### Recommended
- 16GB+ RAM for large matrices (1000x1000+)
- Multi-core processor
- Dedicated GPU (for GPU acceleration)
- SSD storage

## Building from Source

1. Install Rust: https://rustup.rs/
2. Clone the repository
3. Install system dependencies (BLAS/LAPACK)
4. Build: `cargo build --release`

## License

MIT License - see LICENSE file for details.