/**
 * @file simple_test_runner.cpp
 * @brief Simple test runner as fallback when GoogleTest is unavailable
 * 
 * This provides a minimal testing framework for basic validation
 * of the CUDA PDE solver functionality without external dependencies.
 */

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <exception>
#include <memory>
#include <chrono>
#include "parser.h"
#include "solver.h""

// Simple test framework
class SimpleTest {
public:
    struct TestCase {
        std::string name;
        std::function<void()> test;
    };
    
    static void addTest(const std::string& name, std::function<void()> test) {
        tests.push_back({name, test});
    }
    
    static void runAllTests() {
        int passed = 0;
        int failed = 0;
        
        std::cout << "Running " << tests.size() << " tests...\n\n";
        
        for (const auto& test : tests) {
            try {
                std::cout << "[ RUN      ] " << test.name << std::endl;
                test.test();
                std::cout << "[       OK ] " << test.name << std::endl;
                passed++;
            } catch (const std::exception& e) {
                std::cout << "[  FAILED  ] " << test.name << " - " << e.what() << std::endl;
                failed++;
            }
        }
        
        std::cout << "\n==========\n";
        std::cout << "Tests passed: " << passed << std::endl;
        std::cout << "Tests failed: " << failed << std::endl;
        
        if (failed > 0) {
            exit(1);
        }
    }
    
private:
    static std::vector<TestCase> tests;
};

std::vector<SimpleTest::TestCase> SimpleTest::tests;

// Simple assertion macros
#define SIMPLE_ASSERT(condition) \
    if (!(condition)) { \
        throw std::runtime_error("Assertion failed: " #condition); \
    }

#define SIMPLE_ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        throw std::runtime_error("Assertion failed: " #a " == " #b); \
    }

#define SIMPLE_ASSERT_NE(a, b) \
    if ((a) == (b)) { \
        throw std::runtime_error("Assertion failed: " #a " != " #b); \
    }

#define SIMPLE_ASSERT_THROW(stmt, exception_type) \
    { \
        bool caught = false; \
        try { stmt; } \
        catch (const exception_type&) { caught = true; } \
        if (!caught) { \
            throw std::runtime_error("Expected exception not thrown: " #stmt); \
        } \
    }

// Test registration macro
#define SIMPLE_TEST(test_name) \
    void test_##test_name(); \
    static bool dummy_##test_name = (SimpleTest::addTest(#test_name, test_##test_name), true); \
    void test_##test_name()

// Parser tests
SIMPLE_TEST(ParseSimpleNumber) {
    EquationParser parser;
    auto ast = parser.parse("42.5");
    SIMPLE_ASSERT(ast != nullptr);
    SIMPLE_ASSERT_EQ(ast->type, NodeType::NUMBER);
    SIMPLE_ASSERT(std::abs(ast->value - 42.5) < 1e-10);
}

SIMPLE_TEST(ParseVariable) {
    EquationParser parser;
    auto ast = parser.parse("x");
    SIMPLE_ASSERT(ast != nullptr);
    SIMPLE_ASSERT_EQ(ast->type, NodeType::VARIABLE);
    SIMPLE_ASSERT_EQ(ast->variable, "x");
}

SIMPLE_TEST(ParseAddition) {
    EquationParser parser;
    auto ast = parser.parse("x + y");
    SIMPLE_ASSERT(ast != nullptr);
    SIMPLE_ASSERT_EQ(ast->type, NodeType::BINARY_OP);
    SIMPLE_ASSERT_EQ(ast->op, Operator::ADD);
    SIMPLE_ASSERT_EQ(ast->children.size(), 2);
}

SIMPLE_TEST(ParseSinFunction) {
    EquationParser parser;
    auto ast = parser.parse("sin(x)");
    SIMPLE_ASSERT(ast != nullptr);
    SIMPLE_ASSERT_EQ(ast->type, NodeType::FUNCTION);
    SIMPLE_ASSERT_EQ(ast->func, Function::SIN);
    SIMPLE_ASSERT_EQ(ast->children.size(), 1);
}

SIMPLE_TEST(ParseFirstDerivative) {
    SIMPLE_ASSERT(DerivativeParser::isDerivative("du/dx"));
    auto info = DerivativeParser::parseDerivative("du/dx");
    SIMPLE_ASSERT_EQ(info.variable, "x");
    SIMPLE_ASSERT_EQ(info.order, 1);
    SIMPLE_ASSERT_EQ(info.expression, "u");
}

SIMPLE_TEST(ParseSecondDerivative) {
    SIMPLE_ASSERT(DerivativeParser::isDerivative("d2u/dx2"));
    auto info = DerivativeParser::parseDerivative("d2u/dx2");
    SIMPLE_ASSERT_EQ(info.variable, "x");
    SIMPLE_ASSERT_EQ(info.order, 2);
    SIMPLE_ASSERT_EQ(info.expression, "u");
}

SIMPLE_TEST(ParseHeatEquation) {
    EquationParser parser;
    auto ast = parser.parse("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    SIMPLE_ASSERT(ast != nullptr);
    SIMPLE_ASSERT(parser.validatePDE(ast));
}

SIMPLE_TEST(ParseInvalidEquation) {
    EquationParser parser;
    SIMPLE_ASSERT_THROW(parser.parse("x + + y"), std::exception);
}

SIMPLE_TEST(GenerateCudaCode) {
    EquationParser parser;
    auto ast = parser.parse("x + y");
    SIMPLE_ASSERT(ast != nullptr);
    std::string code = parser.generateCudaCode(ast);
    SIMPLE_ASSERT(!code.empty());
}

// Solver tests
SIMPLE_TEST(SolverInitialization) {
    PDESolver solver(32, 32, 32, 0.1, 0.1, 0.1, 0.001);
    solver.initialize();
    const auto& grid = solver.getGrid();
    SIMPLE_ASSERT_EQ(grid.size(), 32 * 32 * 32);
}

SIMPLE_TEST(SolverSetEquation) {
    PDESolver solver(32, 32, 32, 0.1, 0.1, 0.1, 0.001);
    solver.initialize();
    solver.setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    std::string info = solver.getEquationInfo();
    SIMPLE_ASSERT(!info.empty());
}

SIMPLE_TEST(SolverSolveSteps) {
    PDESolver solver(16, 16, 16, 0.1, 0.1, 0.1, 0.001);
    solver.initialize();
    solver.setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    solver.setInitialConditions();
    
    auto initialGrid = solver.getGrid();
    solver.solve(10);
    const auto& finalGrid = solver.getGrid();
    
    // Solution should have evolved
    bool hasChanged = false;
    for (size_t i = 0; i < initialGrid.size(); ++i) {
        if (std::abs(finalGrid[i] - initialGrid[i]) > 1e-10) {
            hasChanged = true;
            break;
        }
    }
    SIMPLE_ASSERT(hasChanged);
}

SIMPLE_TEST(SolverStability) {
    PDESolver solver(16, 16, 16, 0.1, 0.1, 0.1, 0.001);
    solver.initialize();
    solver.setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    solver.setInitialConditions();
    solver.solve(50);
    
    const auto& grid = solver.getGrid();
    for (double value : grid) {
        SIMPLE_ASSERT(std::isfinite(value));
    }
}

SIMPLE_TEST(SolverSaveOutput) {
    PDESolver solver(16, 16, 16, 0.1, 0.1, 0.1, 0.001);
    solver.initialize();
    solver.setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    solver.setInitialConditions();
    solver.solve(5);
    
    std::string filename = "simple_test_output.txt";
    solver.saveSliceToFile(filename, 8);
    
    // Check if file exists
    std::ifstream file(filename);
    SIMPLE_ASSERT(file.good());
    file.close();
    
    // Clean up
    std::remove(filename.c_str());
}

// Integration tests
SIMPLE_TEST(CompleteWorkflow) {
    PDESolver solver(32, 32, 32, 0.1, 0.1, 0.1, 0.001);
    solver.initialize();
    solver.setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    solver.setInitialConditions();
    solver.solve(25);
    
    std::string filename = "workflow_test_output.txt";
    solver.saveSliceToFile(filename, 16);
    
    std::ifstream file(filename);
    SIMPLE_ASSERT(file.good());
    file.close();
    std::remove(filename.c_str());
}

SIMPLE_TEST(MultipleEquations) {
    PDESolver solver(16, 16, 16, 0.1, 0.1, 0.1, 0.001);
    solver.initialize();
    
    std::vector<std::string> equations = {
        "0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)",
        "d2u/dx2 + d2u/dy2",
        "0.05 * d2u/dx2 + 0.1 * d2u/dy2"
    };
    
    for (const auto& eq : equations) {
        solver.setEquation(eq);
        solver.setInitialConditions();
        solver.solve(5);
        
        const auto& grid = solver.getGrid();
        for (double value : grid) {
            SIMPLE_ASSERT(std::isfinite(value));
        }
    }
}

SIMPLE_TEST(PerformanceTest) {
    PDESolver solver(64, 64, 32, 0.1, 0.1, 0.1, 0.001);
    solver.initialize();
    solver.setEquation("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    solver.setInitialConditions();
    
    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(100);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Performance test took: " << duration.count() << "ms" << std::endl;
    
    // Should complete within reasonable time
    SIMPLE_ASSERT(duration.count() < 10000);
}

int main() {
    std::cout << "=== CUDA PDE Solver Simple Test Suite ===\n" << std::endl;
    
    try {
        SimpleTest::runAllTests();
        std::cout << "\nAll tests passed! ✓" << std::endl;
        return 0;
    } catch (...) {
        std::cout << "\nTest suite failed! ✗" << std::endl;
        return 1;
    }
}
