/**
 * @file test_solver.cpp
 * @brief Unit tests for the PDE solver
 */

#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include <fstream>
#include <string>
#include "solver.h"

/**
 * @class SolverTest
 * @brief Test fixture for solver tests
 */
class SolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Small grid for fast testing
        nx = 32;
        ny = 32;
        nz = 32;
        dx = 0.1;
        dy = 0.1;
        dz = 0.1;
        dt = 0.001;
        
        solver = std::make_unique<PDESolver>(nx, ny, nz, dx, dy, dz, dt);
        solver->initialize();
    }
    
    void TearDown() override {
        solver.reset();
    }
    
    std::unique_ptr<PDESolver> solver;
    int nx, ny, nz;
    double dx, dy, dz, dt;
};

// Initialization tests
TEST_F(SolverTest, InitializationSucceeds) {
    EXPECT_NO_THROW(solver->initialize());
    
    const auto& grid = solver->getGrid();
    EXPECT_EQ(grid.size(), nx * ny * nz);
}

TEST_F(SolverTest, InitialConditionsSet) {
    solver->setInitialConditions();
    
    const auto& grid = solver->getGrid();
    
    // Check that some values are non-zero (Gaussian initial condition)
    bool hasNonZeroValues = false;
    for (double value : grid) {
        if (std::abs(value) > 1e-10) {
            hasNonZeroValues = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZeroValues);
}

// Equation setting tests
TEST_F(SolverTest, SetHeatEquation) {
    EXPECT_NO_THROW(solver->setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)"));
    
    std::string info = solver->getEquationInfo();
    EXPECT_FALSE(info.empty());
    EXPECT_NE(info.find("d2u/dx2"), std::string::npos);
}

TEST_F(SolverTest, SetReactionDiffusionEquation) {
    EXPECT_NO_THROW(solver->setEquation("0.1 * (d2u/dx2 + d2u/dy2) + u * (1 - u)"));
    
    std::string info = solver->getEquationInfo();
    EXPECT_FALSE(info.empty());
}

TEST_F(SolverTest, SetInvalidEquation) {
    EXPECT_THROW(solver->setEquation("invalid equation"), std::exception);
}

// Solving tests
TEST_F(SolverTest, SolveHeatEquationConverges) {
    solver->setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    solver->setInitialConditions();
    
    // Store initial state
    auto initialGrid = solver->getGrid();
    
    // Solve for a few time steps
    EXPECT_NO_THROW(solver->solve(10));
    
    // Check that solution has evolved
    const auto& finalGrid = solver->getGrid();
    bool hasChanged = false;
    for (size_t i = 0; i < initialGrid.size(); ++i) {
        if (std::abs(finalGrid[i] - initialGrid[i]) > 1e-10) {
            hasChanged = true;
            break;
        }
    }
    EXPECT_TRUE(hasChanged);
}

TEST_F(SolverTest, SolvePreservesEnergyConservation) {
    solver->setEquation("d2u/dx2 + d2u/dy2 + d2u/dz2");
    solver->setInitialConditions();
    
    const auto& initialGrid = solver->getGrid();
    double initialSum = std::accumulate(initialGrid.begin(), initialGrid.end(), 0.0);
    
    solver->solve(5);
    
    const auto& finalGrid = solver->getGrid();
    double finalSum = std::accumulate(finalGrid.begin(), finalGrid.end(), 0.0);
    
    // For conservative equations, total should be approximately preserved
    // (allowing for numerical errors and boundary effects)
    EXPECT_NEAR(initialSum, finalSum, std::abs(initialSum) * 0.1);
}

TEST_F(SolverTest, SolveProducesStableResults) {
    solver->setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    solver->setInitialConditions();
    
    solver->solve(50);
    
    const auto& grid = solver->getGrid();
    
    // Check for NaN or infinite values
    for (double value : grid) {
        EXPECT_TRUE(std::isfinite(value)) << "Found non-finite value: " << value;
        EXPECT_FALSE(std::isnan(value)) << "Found NaN value";
    }
    
    // Check that values are within reasonable bounds
    auto [minVal, maxVal] = std::minmax_element(grid.begin(), grid.end());
    EXPECT_GE(*minVal, -1000.0) << "Minimum value too negative: " << *minVal;
    EXPECT_LE(*maxVal, 1000.0) << "Maximum value too large: " << *maxVal;
}

// File I/O tests
TEST_F(SolverTest, SaveSliceToFile) {
    solver->setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    solver->setInitialConditions();
    solver->solve(5);
    
    std::string filename = "test_output.txt";
    EXPECT_NO_THROW(solver->saveSliceToFile(filename, nz/2));
    
    // Check if file was created (basic check)
    std::ifstream file(filename);
    EXPECT_TRUE(file.good());
    file.close();
    
    // Clean up
    std::remove(filename.c_str());
}

// Performance and memory tests
TEST_F(SolverTest, SolverPerformanceReasonable) {
    solver->setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    solver->setInitialConditions();
    
    auto start = std::chrono::high_resolution_clock::now();
    solver->solve(100);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should complete within reasonable time (adjust based on hardware)
    EXPECT_LT(duration.count(), 5000) << "Solver took too long: " << duration.count() << "ms";
}

// Boundary condition tests
TEST_F(SolverTest, BoundaryConditionsEnforced) {
    solver->setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    solver->setInitialConditions();
    solver->solve(10);
    
    const auto& grid = solver->getGrid();
    
    // Check corners (should be zero due to Dirichlet boundary conditions)
    auto getIndex = [this](int i, int j, int k) { return (k * ny + j) * nx + i; };
    
    EXPECT_DOUBLE_EQ(grid[getIndex(0, 0, 0)], 0.0);
    EXPECT_DOUBLE_EQ(grid[getIndex(nx-1, 0, 0)], 0.0);
    EXPECT_DOUBLE_EQ(grid[getIndex(0, ny-1, 0)], 0.0);
    EXPECT_DOUBLE_EQ(grid[getIndex(0, 0, nz-1)], 0.0);
    EXPECT_DOUBLE_EQ(grid[getIndex(nx-1, ny-1, nz-1)], 0.0);
}

// Equation-specific tests
class EquationSolverTest : public SolverTest,
                          public ::testing::WithParamInterface<std::string> {
};

TEST_P(EquationSolverTest, SolveSpecificEquation) {
    const std::string& equation = GetParam();
    
    EXPECT_NO_THROW(solver->setEquation(equation));
    EXPECT_NO_THROW(solver->setInitialConditions());
    EXPECT_NO_THROW(solver->solve(10));
    
    const auto& grid = solver->getGrid();
    
    // Verify solution stability
    for (double value : grid) {
        EXPECT_TRUE(std::isfinite(value)) << "Non-finite value in equation: " << equation;
    }
}

INSTANTIATE_TEST_SUITE_P(
    CommonEquations,
    EquationSolverTest,
    ::testing::Values(
        "0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)",           // Heat equation
        "d2u/dx2 + d2u/dy2 + d2u/dz2",                    // Laplace equation
        "0.1 * (d2u/dx2 + d2u/dy2) + u * (1 - u)",       // Reaction-diffusion
        "0.2 * d2u/dx2 + 0.1 * d2u/dy2 + 0.05 * d2u/dz2", // Anisotropic diffusion
        "0.1 * (d2u/dx2 + d2u/dy2) + sin(x) * cos(y)"    // With source term
    )
);

// Memory and resource management tests
TEST_F(SolverTest, MultipleEquationSwitching) {
    // Test switching between different equations
    std::vector<std::string> equations = {
        "0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)",
        "u * (1 - u)",
        "0.05 * d2u/dx2 + sin(x)"
    };
    
    for (const auto& eq : equations) {
        EXPECT_NO_THROW(solver->setEquation(eq));
        EXPECT_NO_THROW(solver->setInitialConditions());
        EXPECT_NO_THROW(solver->solve(5));
    }
}

TEST_F(SolverTest, RepeatedSolving) {
    solver->setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    
    // Multiple solve calls should work
    for (int i = 0; i < 5; ++i) {
        solver->setInitialConditions();
        EXPECT_NO_THROW(solver->solve(10));
    }
}
