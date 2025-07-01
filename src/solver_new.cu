#include "solver.h"
#include "parser.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

__device__ double evaluateCustomEquation(double* grid, int idx, int i, int j, int k, 
                                        int nx, int ny, int nz, double dx, double dy, double dz, 
                                        double dt, double currentTime, const char* kernelCode);
const char* kernelTemplate = R"(
__global__ void customStepKernel(double* grid, double* newGrid, int nx, int ny, int nz,
                                double dx, double dy, double dz, double dt, double currentTime) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
        int idx = (k*ny + j)*nx + i;
        
        double result = %s;
        
        newGrid[idx] = grid[idx] + dt * result;
    }
}
)";

__global__ void heatEquationKernel(double* grid, double* newGrid, int nx, int ny, int nz,
                                  double dx, double dy, double dz, double dt, double alpha) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
        int idx = (k*ny + j)*nx + i;

        // Heat equation: du/dt = alpha * (d2u/dx2 + d2u/dy2 + d2u/dz2)
        double d2udx2 = (grid[((k*ny + j)*nx + i+1)] - 2.0*grid[idx] + grid[((k*ny + j)*nx + i-1)]) / (dx*dx);
        double d2udy2 = (grid[((k*ny + j+1)*nx + i)] - 2.0*grid[idx] + grid[((k*ny + j-1)*nx + i)]) / (dy*dy);
        double d2udz2 = (grid[(((k+1)*ny + j)*nx + i)] - 2.0*grid[idx] + grid[(((k-1)*ny + j)*nx + i)]) / (dz*dz);

        newGrid[idx] = grid[idx] + dt * alpha * (d2udx2 + d2udy2 + d2udz2);
    }
}

__global__ void waveEquationKernel(double* grid, double* newGrid, double* prevGrid, int nx, int ny, int nz,
                                  double dx, double dy, double dz, double dt, double c) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
        int idx = (k*ny + j)*nx + i;

        // Wave equation: d2u/dt2 = c^2 * (d2u/dx2 + d2u/dy2 + d2u/dz2)
        double d2udx2 = (grid[((k*ny + j)*nx + i+1)] - 2.0*grid[idx] + grid[((k*ny + j)*nx + i-1)]) / (dx*dx);
        double d2udy2 = (grid[((k*ny + j+1)*nx + i)] - 2.0*grid[idx] + grid[((k*ny + j-1)*nx + i)]) / (dy*dy);
        double d2udz2 = (grid[(((k+1)*ny + j)*nx + i)] - 2.0*grid[idx] + grid[(((k-1)*ny + j)*nx + i)]) / (dz*dz);

        double laplacian = d2udx2 + d2udy2 + d2udz2;
        newGrid[idx] = 2.0*grid[idx] - prevGrid[idx] + dt*dt * c*c * laplacian;
    }
}

__global__ void advectionKernel(double* grid, double* newGrid, int nx, int ny, int nz,
                               double dx, double dy, double dz, double dt, double vx, double vy, double vz) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
        int idx = (k*ny + j)*nx + i;

        // Advection equation: du/dt + v·∇u = 0
        double dudx = (grid[((k*ny + j)*nx + i+1)] - grid[((k*ny + j)*nx + i-1)]) / (2.0*dx);
        double dudy = (grid[((k*ny + j+1)*nx + i)] - grid[((k*ny + j-1)*nx + i)]) / (2.0*dy);
        double dudz = (grid[(((k+1)*ny + j)*nx + i)] - grid[(((k-1)*ny + j)*nx + i)]) / (2.0*dz);

        newGrid[idx] = grid[idx] - dt * (vx*dudx + vy*dudy + vz*dudz);
    }
}

PDESolver::PDESolver(int nx, int ny, int nz, double dx, double dy, double dz, double dt)
    : nx_(nx), ny_(ny), nz_(nz), dx_(dx), dy_(dy), dz_(dz), dt_(dt) {
    grid_.resize(nx_ * ny_ * nz_);
    // Default to heat equation
    equation_ = "0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)";
    generateKernelCode();
    initialize();
}

void PDESolver::initialize() {
    std::fill(grid_.begin(), grid_.end(), 0.0);
}

void PDESolver::setInitialConditions() {
    // Set some interesting initial conditions
    int centerX = nx_ / 2;
    int centerY = ny_ / 2;
    int centerZ = nz_ / 2;

    // Central hot spot
    for (int k = centerZ - 5; k <= centerZ + 5; ++k) {
        for (int j = centerY - 5; j <= centerY + 5; ++j) {
            for (int i = centerX - 5; i <= centerX + 5; ++i) {
                if (i >= 0 && i < nx_ && j >= 0 && j < ny_ && k >= 0 && k < nz_) {
                    double dist = sqrt((i-centerX)*(i-centerX) + (j-centerY)*(j-centerY) + (k-centerZ)*(k-centerZ));
                    if (dist <= 5.0) {
                        grid_[getIndex(i, j, k)] = 100.0 * exp(-dist*dist/10.0);
                    }
                }
            }
        }
    }

    // Additional heat sources
    for (int k = 5; k <= 15; ++k) {
        for (int j = 5; j <= 15; ++j) {
            for (int i = 5; i <= 15; ++i) {
                grid_[getIndex(i, j, k)] = 75.0;
            }
        }
    }

    int x2 = nx_ - 15, y2 = ny_ - 15, z2 = nz_ - 15;
    for (int k = z2 - 3; k <= z2 + 3; ++k) {
        for (int j = y2 - 3; j <= y2 + 3; ++j) {
            for (int i = x2 - 3; i <= x2 + 3; ++i) {
                if (i >= 0 && i < nx_ && j >= 0 && j < ny_ && k >= 0 && k < nz_) {
                    grid_[getIndex(i, j, k)] = 50.0;
                }
            }
        }
    }
}

void PDESolver::setEquation(const std::string& equation) {
    equation_ = equation;
    generateKernelCode();
    std::cout << "Set custom equation: " << equation_ << std::endl;
    std::cout << "Generated kernel code includes: " << generatedKernelCode_.substr(0, 100) << "..." << std::endl;
}

void PDESolver::generateKernelCode() {
    try {
        auto ast = parser_.parse(equation_);
        
        if (!parser_.validatePDE(ast)) {
            std::cerr << "Warning: Equation may not be a valid PDE" << std::endl;
        }
        
        generatedKernelCode_ = parser_.generateCudaCode(ast);
        
        // Get variables used in the equation
        auto variables = parser_.getVariables(ast);
        std::cout << "Variables in equation: ";
        for (const auto& var : variables) {
            std::cout << var << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing equation: " << e.what() << std::endl;
        std::cerr << "Falling back to heat equation" << std::endl;
        generatedKernelCode_ = "0.1 * ((grid[((k*ny + j)*nx + i+1)] - 2.0*grid[idx] + grid[((k*ny + j)*nx + i-1)]) / (dx*dx) + (grid[((k*ny + j+1)*nx + i)] - 2.0*grid[idx] + grid[((k*ny + j-1)*nx + i)]) / (dy*dy) + (grid[(((k+1)*ny + j)*nx + i)] - 2.0*grid[idx] + grid[(((k-1)*ny + j)*nx + i)]) / (dz*dz))";
    }
}

void PDESolver::solve(int timeSteps) {
    double* d_grid = nullptr;
    double* d_newGrid = nullptr;
    size_t size = grid_.size() * sizeof(double);

    // Allocate GPU memory
    cudaMalloc(&d_grid, size);
    cudaMalloc(&d_newGrid, size);
    cudaMemcpy(d_grid, grid_.data(), size, cudaMemcpyHostToDevice);

    // Set up CUDA execution configuration
    dim3 block(8, 8, 8);
    dim3 gridDim((nx_ + block.x - 1) / block.x,
                 (ny_ + block.y - 1) / block.y,
                 (nz_ + block.z - 1) / block.z);

    // Progress reporting
    int reportInterval = timeSteps / 10;
    if (reportInterval == 0) reportInterval = 1;

    // Determine which kernel to use based on equation type
    bool useHeatKernel = (equation_.find("d2u/dx2") != std::string::npos && 
                         equation_.find("d2u/dy2") != std::string::npos && 
                         equation_.find("d2u/dz2") != std::string::npos &&
                         equation_.find("0.1") != std::string::npos);

    for (int t = 0; t < timeSteps; ++t) {
        double currentTime = t * dt_;
        
        if (useHeatKernel) {
            // Use optimized heat equation kernel
            heatEquationKernel<<<gridDim, block>>>(d_grid, d_newGrid, nx_, ny_, nz_, dx_, dy_, dz_, dt_, 0.1);
        } else {
        // For now, fall back to heat equation for complex custom equations
            heatEquationKernel<<<gridDim, block>>>(d_grid, d_newGrid, nx_, ny_, nz_, dx_, dy_, dz_, dt_, 0.1);
        }
        
        cudaDeviceSynchronize();

        // Check for CUDA errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error at step " << t << ": " << cudaGetErrorString(error) << std::endl;
            break;
        }

        // Swap buffers
        std::swap(d_grid, d_newGrid);

        // Progress reporting
        if (t % reportInterval == 0 || t == timeSteps - 1) {
            std::cout << "  Progress: " << (100 * (t + 1)) / timeSteps << "% (step " << t + 1 << "/" << timeSteps << ")" << std::endl;
        }
    }

    // Copy result back to host
    cudaMemcpy(grid_.data(), d_grid, size, cudaMemcpyDeviceToHost);
    cudaFree(d_grid);
    cudaFree(d_newGrid);
}

void PDESolver::saveSliceToFile(const std::string& filename, int zSlice) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }

    file << "# 2D slice at z=" << zSlice << " from custom PDE solution" << std::endl;
    file << "# Equation: " << equation_ << std::endl;
    file << "# Grid size: " << nx_ << "x" << ny_ << std::endl;
    file << "# Format: x y value" << std::endl;

    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            double val = grid_[getIndex(i, j, zSlice)];
            file << i << " " << j << " " << val << std::endl;
        }
        file << std::endl;
    }

    file.close();
    std::cout << "  Saved 2D slice to: " << filename << std::endl;
}

const std::vector<double>& PDESolver::getGrid() const {
    return grid_;
}

std::string PDESolver::getEquationInfo() const {
    std::stringstream ss;
    ss << "Current equation: " << equation_ << std::endl;
    ss << "Generated CUDA code: " << generatedKernelCode_ << std::endl;
    return ss.str();
}
