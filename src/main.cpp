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
    std::cout << "  Variables: u (solution), x, y, z, t (time)" << std::endl;
    std::cout << "  Derivatives: du/dx, d2u/dx2, d2u/dy2, d2u/dz2" << std::endl;
    std::cout << "  Functions: sin, cos, tan, exp, log, sqrt, abs" << std::endl;
    std::cout << "  Operators: +, -, *, /, ^" << std::endl;
    std::cout << std::endl;
    std::cout << "Example equations:" << std::endl;
    std::cout << "  Heat equation: 0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)" << std::endl;
    std::cout << "  Wave equation: d2u/dx2 + d2u/dy2 + d2u/dz2" << std::endl;
    std::cout << "  Reaction-diffusion: 0.1 * (d2u/dx2 + d2u/dy2) + u * (1 - u)" << std::endl;
    std::cout << "  With source term: 0.1 * (d2u/dx2 + d2u/dy2) + sin(x) * cos(y)" << std::endl;
    std::cout << std::endl;
}

void printPresetEquations() {
    std::cout << "Available preset equations:" << std::endl;
    std::cout << "1. Heat equation (default)" << std::endl;
    std::cout << "2. Fast diffusion" << std::endl;
    std::cout << "3. Anisotropic diffusion" << std::endl;
    std::cout << "4. Reaction-diffusion" << std::endl;
    std::cout << "5. Wave equation (simplified)" << std::endl;
    std::cout << "6. Custom equation" << std::endl;
}

std::string getPresetEquation(int choice) {
    switch (choice) {
        case 1: return "0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)";
        case 2: return "0.5 * (d2u/dx2 + d2u/dy2 + d2u/dz2)";
        case 3: return "0.2 * d2u/dx2 + 0.1 * d2u/dy2 + 0.05 * d2u/dz2";
        case 4: return "0.1 * (d2u/dx2 + d2u/dy2) + u * (1 - u)";
        case 5: return "d2u/dx2 + d2u/dy2 + d2u/dz2";
        default: return "";
    }
}

int main() {
    printWelcome();

    // Grid parameters
    int nx = 64, ny = 64, nz = 64;
    double dx = 0.1, dy = 0.1, dz = 0.1, dt = 0.001;

    std::cout << "Grid size: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "Spatial resolution: dx=" << dx << ", dy=" << dy << ", dz=" << dz << std::endl;
    std::cout << "Time step: dt=" << dt << std::endl;
    std::cout << std::endl;

    // Create solver
    PDESolver solver(nx, ny, nz, dx, dy, dz, dt);

    // Get equation from user
    printPresetEquations();
    std::cout << "Choose an option (1-6): ";
    
    int choice;
    std::cin >> choice;
    std::cin.ignore(); // Consume newline
    
    std::string equation;
    if (choice >= 1 && choice <= 5) {
        equation = getPresetEquation(choice);
        std::cout << "Selected equation: " << equation << std::endl;
    } else if (choice == 6) {
        std::cout << "Enter your custom equation (du/dt = ?): ";
        std::getline(std::cin, equation);
    } else {
        std::cout << "Invalid choice. Using default heat equation." << std::endl;
        equation = getPresetEquation(1);
    }

    // Set the equation
    solver.setEquation(equation);
    std::cout << std::endl;

    // Set initial conditions
    std::cout << "Setting initial conditions..." << std::endl;
    solver.setInitialConditions();

    // Ask for number of time steps
    int timeSteps = 1000;
    std::cout << "Number of time steps (default 1000): ";
    std::string input;
    std::getline(std::cin, input);
    if (!input.empty()) {
        try {
            timeSteps = std::stoi(input);
        } catch (...) {
            std::cout << "Invalid input, using default: " << timeSteps << std::endl;
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
