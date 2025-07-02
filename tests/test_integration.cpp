#include <gtest/gtest.h>
#include <memory>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include "solver.h"
#include "parser.h"

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use moderate grid size for integration tests
        nx = 64;
        ny = 64;
        nz = 64;
        dx = 0.1;
        dy = 0.1;
        dz = 0.1;
        dt = 0.001;
        
        solver = std::make_unique<PDESolver>(nx, ny, nz, dx, dy, dz, dt);
        solver->initialize();
    }
    
    void TearDown() override {
        solver.reset();
        
        // Clean up any test files
        cleanupTestFiles();
    }
    
    void cleanupTestFiles() {
        std::vector<std::string> testFiles = {
            "test_output.txt",
            "test_equation_info.txt",
            "integration_test_output.txt"
        };
        
        for (const auto& file : testFiles) {
            std::remove(file.c_str());
        }
    }
    
    // Helper function to check if file exists and has content
    bool fileExistsAndHasContent(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.good()) return false;
        
        file.seekg(0, std::ios::end);
        return file.tellg() > 0;
    }
    
    // Helper to read file content
    std::string readFileContent(const std::string& filename) {
        std::ifstream file(filename);
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
    
    std::unique_ptr<PDESolver> solver;
    int nx, ny, nz;
    double dx, dy, dz, dt;
};

// Complete workflow tests
TEST_F(IntegrationTest, CompleteHeatEquationWorkflow) {
    // 1. Set equation
    std::string equation = "0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)";
    EXPECT_NO_THROW(solver->setEquation(equation));
    
    // 2. Set initial conditions
    EXPECT_NO_THROW(solver->setInitialConditions());
    
    // 3. Solve for multiple time steps
    EXPECT_NO_THROW(solver->solve(100));
    
    // 4. Save results
    std::string outputFile = "integration_test_output.txt";
    EXPECT_NO_THROW(solver->saveSliceToFile(outputFile, nz/2));
    
    // 5. Verify output file
    EXPECT_TRUE(fileExistsAndHasContent(outputFile));
    
    // 6. Check equation info
    std::string info = solver->getEquationInfo();
    EXPECT_FALSE(info.empty());
    EXPECT_NE(info.find("d2u/dx2"), std::string::npos);
}

TEST_F(IntegrationTest, CompleteReactionDiffusionWorkflow) {
    // Test reaction-diffusion equation
    std::string equation = "0.1 * (d2u/dx2 + d2u/dy2) + u * (1 - u)";
    
    EXPECT_NO_THROW(solver->setEquation(equation));
    EXPECT_NO_THROW(solver->setInitialConditions());
    EXPECT_NO_THROW(solver->solve(50));
    
    std::string outputFile = "test_output.txt";
    EXPECT_NO_THROW(solver->saveSliceToFile(outputFile, nz/2));
    EXPECT_TRUE(fileExistsAndHasContent(outputFile));
}

TEST_F(IntegrationTest, CompleteCustomEquationWorkflow) {
    // Test custom equation with trigonometric source
    std::string equation = "0.05 * (d2u/dx2 + d2u/dy2) + sin(x) * cos(y)";
    
    EXPECT_NO_THROW(solver->setEquation(equation));
    EXPECT_NO_THROW(solver->setInitialConditions());
    EXPECT_NO_THROW(solver->solve(25));
    
    const auto& grid = solver->getGrid();
    
    // Verify solution stability
    for (double value : grid) {
        EXPECT_TRUE(std::isfinite(value));
    }
}

// Performance integration tests
TEST_F(IntegrationTest, LargeScalePerformanceTest) {
    // Test with larger problem size
    auto largeSolver = std::make_unique<PDESolver>(128, 128, 64, 0.05, 0.05, 0.1, 0.0005);
    largeSolver->initialize();
    
    largeSolver->setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    largeSolver->setInitialConditions();
    
    auto start = std::chrono::high_resolution_clock::now();
    largeSolver->solve(50);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should complete within reasonable time (adjust based on hardware)
    EXPECT_LT(duration.count(), 30000) << "Large scale test took too long: " << duration.count() << "ms";
}

// Multi-equation switching test
TEST_F(IntegrationTest, MultipleEquationSwitching) {
    std::vector<std::string> equations = {
        "0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)",           // Heat
        "d2u/dx2 + d2u/dy2 + d2u/dz2",                    // Laplace
        "0.1 * (d2u/dx2 + d2u/dy2) + u * (1 - u)",       // Reaction-diffusion
        "0.2 * d2u/dx2 + 0.1 * d2u/dy2"                  // Anisotropic
    };
    
    for (size_t i = 0; i < equations.size(); ++i) {
        EXPECT_NO_THROW(solver->setEquation(equations[i])) 
            << "Failed on equation " << i << ": " << equations[i];
        EXPECT_NO_THROW(solver->setInitialConditions());
        EXPECT_NO_THROW(solver->solve(10));
        
        std::string filename = "test_output_" + std::to_string(i) + ".txt";
        EXPECT_NO_THROW(solver->saveSliceToFile(filename, nz/2));
        EXPECT_TRUE(fileExistsAndHasContent(filename));
        
        // Clean up
        std::remove(filename.c_str());
    }
}

// File I/O integration tests
TEST_F(IntegrationTest, OutputFileFormatValidation) {
    solver->setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    solver->setInitialConditions();
    solver->solve(10);
    
    std::string outputFile = "test_output.txt";
    solver->saveSliceToFile(outputFile, nz/2);
    
    // Read and validate file format
    std::ifstream file(outputFile);
    ASSERT_TRUE(file.good());
    
    std::string line;
    int lineCount = 0;
    while (std::getline(file, line) && lineCount < 10) {
        // Each line should contain space-separated numbers
        std::istringstream iss(line);
        std::string token;
        int tokenCount = 0;
        
        while (iss >> token && tokenCount < 10) {
            // Try to parse as double
            try {
                double value = std::stod(token);
                EXPECT_TRUE(std::isfinite(value)) << "Non-finite value in output file: " << value;
            } catch (const std::exception&) {
                FAIL() << "Invalid number format in output file: " << token;
            }
            tokenCount++;
        }
        
        EXPECT_GT(tokenCount, 0) << "Empty line in output file";
        lineCount++;
    }
    
    EXPECT_GT(lineCount, 0) << "Output file is empty";
}

// Memory and resource management integration tests
TEST_F(IntegrationTest, MemoryLeakTest) {
    // Simulate repeated solving to check for memory leaks
    for (int iteration = 0; iteration < 10; ++iteration) {
        solver->setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
        solver->setInitialConditions();
        solver->solve(20);
        
        const auto& grid = solver->getGrid();
        EXPECT_EQ(grid.size(), nx * ny * nz);
        
        // Basic sanity check
        bool hasFiniteValues = std::all_of(grid.begin(), grid.end(), 
                                          [](double v) { return std::isfinite(v); });
        EXPECT_TRUE(hasFiniteValues) << "Non-finite values in iteration " << iteration;
    }
}

TEST_F(IntegrationTest, ConcurrentSolverInstances) {
    // Test multiple solver instances (though not necessarily concurrent due to CUDA context)
    auto solver2 = std::make_unique<PDESolver>(32, 32, 32, 0.1, 0.1, 0.1, 0.001);
    solver2->initialize();
    
    // Set different equations
    solver->setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    solver2->setEquation("0.2 * d2u/dx2 + 0.1 * d2u/dy2");
    
    solver->setInitialConditions();
    solver2->setInitialConditions();
    
    // Solve both
    EXPECT_NO_THROW(solver->solve(10));
    EXPECT_NO_THROW(solver2->solve(10));
    
    // Both should have valid results
    const auto& grid1 = solver->getGrid();
    const auto& grid2 = solver2->getGrid();
    
    for (double v : grid1) EXPECT_TRUE(std::isfinite(v));
    for (double v : grid2) EXPECT_TRUE(std::isfinite(v));
}

// Error handling integration tests
TEST_F(IntegrationTest, ErrorRecoveryTest) {
    // Test recovery from invalid equation
    EXPECT_THROW(solver->setEquation("invalid equation"), std::exception);
    
    // Should still be able to set valid equation afterward
    EXPECT_NO_THROW(solver->setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)"));
    EXPECT_NO_THROW(solver->setInitialConditions());
    EXPECT_NO_THROW(solver->solve(5));
}

// Numerical accuracy integration tests
TEST_F(IntegrationTest, ConvergenceTest) {
    // Test that solution converges as expected for heat equation
    solver->setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    solver->setInitialConditions();
    
    // Store solutions at different time steps
    std::vector<std::vector<double>> solutions;
    
    for (int steps : {10, 20, 50}) {
        solver->setInitialConditions(); // Reset
        solver->solve(steps);
        solutions.push_back(solver->getGrid());
    }
    
    // Solutions should show progression (for heat equation, energy should decrease)
    auto computeEnergy = [](const std::vector<double>& grid) {
        return std::accumulate(grid.begin(), grid.end(), 0.0, 
                              [](double sum, double v) { return sum + v*v; });
    };
    
    double energy1 = computeEnergy(solutions[0]);
    double energy2 = computeEnergy(solutions[1]);
    double energy3 = computeEnergy(solutions[2]);
    
    // For heat equation, energy should generally decrease over time
    EXPECT_GE(energy1, energy2 * 0.9) << "Energy not decreasing as expected";
    EXPECT_GE(energy2, energy3 * 0.9) << "Energy not decreasing as expected";
}

// Parameterized integration tests
class EquationIntegrationTest : public IntegrationTest,
                               public ::testing::WithParamInterface<std::pair<std::string, int>> {
};

TEST_P(EquationIntegrationTest, TestEquationIntegration) {
    const auto& [equation, timeSteps] = GetParam();
    
    EXPECT_NO_THROW(solver->setEquation(equation)) << "Failed to set equation: " << equation;
    EXPECT_NO_THROW(solver->setInitialConditions());
    EXPECT_NO_THROW(solver->solve(timeSteps));
    
    const auto& grid = solver->getGrid();
    
    // Verify solution validity
    for (double value : grid) {
        EXPECT_TRUE(std::isfinite(value)) << "Non-finite value in equation: " << equation;
    }
    
    // Save and verify output
    std::string outputFile = "integration_test_output.txt";
    EXPECT_NO_THROW(solver->saveSliceToFile(outputFile, nz/2));
    EXPECT_TRUE(fileExistsAndHasContent(outputFile));
}

INSTANTIATE_TEST_SUITE_P(
    VariousEquations,
    EquationIntegrationTest,
    ::testing::Values(
        std::make_pair("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)", 50),        // Heat equation
        std::make_pair("d2u/dx2 + d2u/dy2", 30),                          // 2D Laplace
        std::make_pair("0.1 * d2u/dx2 + 0.2 * d2u/dy2", 40),             // Anisotropic
        std::make_pair("0.05 * (d2u/dx2 + d2u/dy2) + sin(x)", 25),       // With source
        std::make_pair("0.1 * (d2u/dx2 + d2u/dy2) + u * (1 - u)", 20)    // Reaction-diffusion
    )
);
