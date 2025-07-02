#include "solver.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <numeric>
#include <string>

void printWelcome() {
    std::cout << "=== CUDA PDE Solver - Custom Equation Version ===" << std::endl;
    std::cout << "This solver can handle custom partial differential equations." << std::endl;
    std::cout << std::endl;
    std::cout << "Supported syntax:" << std::endl;
    std::cout << "  Variables: u (solution), x, y, z, t" << std::endl;
    std::cout << "  Derivatives: du/dx, d2u/dx2, d2u/dy2, d2u/dz2" << std::endl;
    std::cout << "  Functions: sin, cos, tan, exp, log, sqrt, abs" << std::endl;
    std::cout << "  Operators: +, -, *, /, ^" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: ./CudaProjDifferentials \"<equation>\" [time_steps]" << std::endl;
    std::cout << "Example: ./CudaProjDifferentials \"0.1 * (d2u/dx2 + d2u/dy2)\" 1000" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    printWelcome();

    if (argc < 2) {
        std::cerr << "Error: No equation provided." << std::endl;
        std::cerr << "Usage: ./CudaProjDifferentials \"<equation>\" [time_steps]" << std::endl;
        return 1;
    }

    std::string equation = argv[1];

    // Grid parameters
    int nx = 64, ny = 64, nz = 64;
    double dx = 0.1, dy = 0.1, dz = 0.1, dt = 0.001;

    std::cout << "Grid size: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "Spatial resolution: dx=" << dx << ", dy=" << dy << ", dz=" << dz << std::endl;
    std::cout << "Time step: dt=" << dt << std::endl;
    std::cout << std::endl;

    // Create solver
    PDESolver solver(nx, ny, nz, dx, dy, dz, dt);

    // Set the equation
    solver.setEquation(equation);
    std::cout << std::endl;

    // Set initial conditions
    std::cout << "Setting initial conditions..." << std::endl;
    solver.setInitialConditions();

    // Ask for number of time steps
    int timeSteps = 1000;
    if (argc > 2) {
        try {
            timeSteps = std::stoi(argv[2]);
        } catch (...) {
            std::cout << "Invalid time steps provided, using default: " << timeSteps << std::endl;
        }
    }

    // Solve the PDE
    std::cout << "Solving PDE for " << timeSteps << " time steps..." << std::endl;
    std::cout << "Equation: du/dt = " << equation << std::endl;
    std::cout << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(timeSteps);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Solving completed in " << duration.count() << " ms" << std::endl;

    // Print statistics
    const auto& grid = solver.getGrid();
    double minVal = *std::min_element(grid.begin(), grid.end());
    double maxVal = *std::max_element(grid.begin(), grid.end());
    double avgVal = std::accumulate(grid.begin(), grid.end(), 0.0) / grid.size();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Final solution statistics:" << std::endl;
    std::cout << "  Min value: " << minVal << std::endl;
    std::cout << "  Max value: " << maxVal << std::endl;
    std::cout << "  Average value: " << avgVal << std::endl;

    // Save results
    std::cout << std::endl << "Saving results..." << std::endl;
    solver.saveSliceToFile("output.txt", nz/2);
    
    // Save equation info
    std::ofstream infoFile("equation_info.txt");
    if (infoFile.is_open()) {
        infoFile << solver.getEquationInfo();
        infoFile.close();
        std::cout << "  Equation info saved to: equation_info.txt" << std::endl;
    }

    std::cout << std::endl << "Program completed successfully!" << std::endl;
    return 0;
}
