# CUDA PDE Solver

A high-performance CUDA-accelerated solver for custom partial differential equations (PDEs) with runtime equation parsing and dynamic kernel generation.

## Running the Solver

```bash
# Basic usage with heat equation
./CudaProjDifferentials "0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)"

# Reaction-diffusion equation
./CudaProjDifferentials "0.1 * (d2u/dx2 + d2u/dy2) + u * (1 - u)"

# Custom time steps
./CudaProjDifferentials "0.2 * (d2u/dx2 + d2u/dy2) + u * (1 - u / 100)" 2000
```

## Supported Equation Syntax

### Variables

- `u` - The solution variable
- `x`, `y`, `z` - Spatial coordinates
- `t` - Time coordinate

### Derivatives

- `du/dx`, `du/dy`, `du/dz` - First-order partial derivatives
- `d2u/dx2`, `d2u/dy2`, `d2u/dz2` - Second-order partial derivatives
- `du/dt` - Time derivative

### Mathematical Functions

- `sin(x)`, `cos(x)`, `tan(x)` - Trigonometric functions
- `exp(x)`, `log(x)` - Exponential and natural logarithm
- `sqrt(x)`, `abs(x)` - Square root and absolute value

### Operators

- `+`, `-`, `*`, `/` - Basic arithmetic
- `^` - Exponentiation
- `()` - Parentheses for grouping

## Output

The solver generates two main output files:

### `output.txt`

Contains a 2D slice of the 3D solution at the middle z-plane:

```
# 2D slice at z=32 from custom PDE solution
# Equation: 0.2 * (d2u/dx2 + d2u/dy2) + u * (1 - u / 100)
# Grid size: 64x64
# Format: x y value
0 0 0
1 0 4.39717e-13
...
```

### `equation_info.txt`

Contains the parsed equation and generated CUDA code:
```
Current equation: 0.2 * (d2u/dx2 + d2u/dy2) + u * (1 - u / 100)
Generated CUDA code: ((0.200000 * ((d2u / dx2) + (d2u / dy2))) + (grid[idx] * (1.000000 - (grid[idx] / 100.000000))))
```
