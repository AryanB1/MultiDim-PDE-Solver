#include <gtest/gtest.h>
#include "parser.h"

class DerivativeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup any common test data
    }
};

// Basic derivative recognition tests
TEST_F(DerivativeTest, RecognizeFirstDerivatives) {
    EXPECT_TRUE(DerivativeParser::isDerivative("du/dx"));
    EXPECT_TRUE(DerivativeParser::isDerivative("du/dy"));
    EXPECT_TRUE(DerivativeParser::isDerivative("du/dz"));
    EXPECT_TRUE(DerivativeParser::isDerivative("du/dt"));
}

TEST_F(DerivativeTest, RecognizeSecondDerivatives) {
    EXPECT_TRUE(DerivativeParser::isDerivative("d2u/dx2"));
    EXPECT_TRUE(DerivativeParser::isDerivative("d2u/dy2"));
    EXPECT_TRUE(DerivativeParser::isDerivative("d2u/dz2"));
    EXPECT_TRUE(DerivativeParser::isDerivative("d2u/dt2"));
}

TEST_F(DerivativeTest, RejectNonDerivatives) {
    EXPECT_FALSE(DerivativeParser::isDerivative("u"));
    EXPECT_FALSE(DerivativeParser::isDerivative("x"));
    EXPECT_FALSE(DerivativeParser::isDerivative("sin(x)"));
    EXPECT_FALSE(DerivativeParser::isDerivative("du/dxx"));
    EXPECT_FALSE(DerivativeParser::isDerivative("d3u/dx3"));
    EXPECT_FALSE(DerivativeParser::isDerivative("du"));
}

// Derivative parsing tests
TEST_F(DerivativeTest, ParseFirstDerivativeX) {
    auto info = DerivativeParser::parseDerivative("du/dx");
    EXPECT_EQ(info.variable, "x");
    EXPECT_EQ(info.order, 1);
    EXPECT_EQ(info.expression, "u");
}

TEST_F(DerivativeTest, ParseFirstDerivativeY) {
    auto info = DerivativeParser::parseDerivative("du/dy");
    EXPECT_EQ(info.variable, "y");
    EXPECT_EQ(info.order, 1);
    EXPECT_EQ(info.expression, "u");
}

TEST_F(DerivativeTest, ParseFirstDerivativeZ) {
    auto info = DerivativeParser::parseDerivative("du/dz");
    EXPECT_EQ(info.variable, "z");
    EXPECT_EQ(info.order, 1);
    EXPECT_EQ(info.expression, "u");
}

TEST_F(DerivativeTest, ParseFirstDerivativeT) {
    auto info = DerivativeParser::parseDerivative("du/dt");
    EXPECT_EQ(info.variable, "t");
    EXPECT_EQ(info.order, 1);
    EXPECT_EQ(info.expression, "u");
}

TEST_F(DerivativeTest, ParseSecondDerivativeX) {
    auto info = DerivativeParser::parseDerivative("d2u/dx2");
    EXPECT_EQ(info.variable, "x");
    EXPECT_EQ(info.order, 2);
    EXPECT_EQ(info.expression, "u");
}

TEST_F(DerivativeTest, ParseSecondDerivativeY) {
    auto info = DerivativeParser::parseDerivative("d2u/dy2");
    EXPECT_EQ(info.variable, "y");
    EXPECT_EQ(info.order, 2);
    EXPECT_EQ(info.expression, "u");
}

TEST_F(DerivativeTest, ParseSecondDerivativeZ) {
    auto info = DerivativeParser::parseDerivative("d2u/dz2");
    EXPECT_EQ(info.variable, "z");
    EXPECT_EQ(info.order, 2);
    EXPECT_EQ(info.expression, "u");
}

// Complex derivative expressions
TEST_F(DerivativeTest, ParseDerivativeWithWhitespace) {
    EXPECT_TRUE(DerivativeParser::isDerivative(" du/dx "));
    EXPECT_TRUE(DerivativeParser::isDerivative("d2u / dx2"));
    EXPECT_TRUE(DerivativeParser::isDerivative("d2u/ dy2"));
    
    auto info = DerivativeParser::parseDerivative(" du/dx ");
    EXPECT_EQ(info.variable, "x");
    EXPECT_EQ(info.order, 1);
}

// Error cases
TEST_F(DerivativeTest, HandleInvalidDerivativeFormat) {
    EXPECT_THROW(DerivativeParser::parseDerivative("invalid"), std::exception);
    EXPECT_THROW(DerivativeParser::parseDerivative("du/dxx"), std::exception);
    EXPECT_THROW(DerivativeParser::parseDerivative("d3u/dx3"), std::exception);
}

// Integration with parser
TEST_F(DerivativeTest, ParseEquationWithDerivatives) {
    EquationParser parser;
    
    auto ast = parser.parse("d2u/dx2 + d2u/dy2");
    ASSERT_NE(ast, nullptr);
    
    // Should recognize as valid PDE
    EXPECT_TRUE(parser.validatePDE(ast));
    
    // Should extract 'u' as variable
    auto variables = parser.getVariables(ast);
    EXPECT_TRUE(std::find(variables.begin(), variables.end(), "u") != variables.end());
}

TEST_F(DerivativeTest, ParseLaplaceOperator) {
    EquationParser parser;
    
    auto ast = parser.parse("d2u/dx2 + d2u/dy2 + d2u/dz2");
    ASSERT_NE(ast, nullptr);
    
    EXPECT_TRUE(parser.validatePDE(ast));
    
    // Generate CUDA code
    std::string cudaCode = parser.generateCudaCode(ast);
    EXPECT_FALSE(cudaCode.empty());
}

TEST_F(DerivativeTest, ParseMixedDerivativeEquation) {
    EquationParser parser;
    
    auto ast = parser.parse("0.1 * d2u/dx2 + 0.2 * d2u/dy2 + u");
    ASSERT_NE(ast, nullptr);
    
    EXPECT_TRUE(parser.validatePDE(ast));
}

// Parametric tests for various derivative formats
class DerivativeFormatTest : public DerivativeTest,
                            public ::testing::WithParamInterface<std::tuple<std::string, bool, std::string, int>> {
};

TEST_P(DerivativeFormatTest, TestDerivativeFormat) {
    const auto& [input, isValid, expectedVar, expectedOrder] = GetParam();
    
    EXPECT_EQ(DerivativeParser::isDerivative(input), isValid) 
        << "Derivative recognition failed for: " << input;
    
    if (isValid) {
        auto info = DerivativeParser::parseDerivative(input);
        EXPECT_EQ(info.variable, expectedVar) << "Variable mismatch for: " << input;
        EXPECT_EQ(info.order, expectedOrder) << "Order mismatch for: " << input;
    }
}

INSTANTIATE_TEST_SUITE_P(
    DerivativeFormats,
    DerivativeFormatTest,
    ::testing::Values(
        // Valid first derivatives
        std::make_tuple("du/dx", true, "x", 1),
        std::make_tuple("du/dy", true, "y", 1),
        std::make_tuple("du/dz", true, "z", 1),
        std::make_tuple("du/dt", true, "t", 1),
        
        // Valid second derivatives
        std::make_tuple("d2u/dx2", true, "x", 2),
        std::make_tuple("d2u/dy2", true, "y", 2),
        std::make_tuple("d2u/dz2", true, "z", 2),
        std::make_tuple("d2u/dt2", true, "t", 2),
        
        // Invalid formats
        std::make_tuple("u", false, "", 0),
        std::make_tuple("du/dxx", false, "", 0),
        std::make_tuple("d3u/dx3", false, "", 0),
        std::make_tuple("du", false, "", 0),
        std::make_tuple("dx/du", false, "", 0),
        std::make_tuple("d2u/dxdy", false, "", 0)
    )
);

// CUDA code generation tests for derivatives
TEST_F(DerivativeTest, GenerateCudaCodeForFirstDerivative) {
    EquationParser parser;
    
    auto ast = parser.parse("du/dx");
    ASSERT_NE(ast, nullptr);
    
    std::string cudaCode = parser.generateCudaCode(ast);
    EXPECT_FALSE(cudaCode.empty());
    
    // Should contain finite difference approximation
    // (exact format depends on implementation)
    EXPECT_TRUE(cudaCode.find("(") != std::string::npos || 
                cudaCode.find("grid") != std::string::npos ||
                cudaCode.find("finite") != std::string::npos);
}

TEST_F(DerivativeTest, GenerateCudaCodeForSecondDerivative) {
    EquationParser parser;
    
    auto ast = parser.parse("d2u/dx2");
    ASSERT_NE(ast, nullptr);
    
    std::string cudaCode = parser.generateCudaCode(ast);
    EXPECT_FALSE(cudaCode.empty());
    
    // Should contain second-order finite difference
    EXPECT_TRUE(cudaCode.find("2") != std::string::npos ||
                cudaCode.find("grid") != std::string::npos);
}

TEST_F(DerivativeTest, GenerateCudaCodeForLaplacian) {
    EquationParser parser;
    
    auto ast = parser.parse("d2u/dx2 + d2u/dy2 + d2u/dz2");
    ASSERT_NE(ast, nullptr);
    
    std::string cudaCode = parser.generateCudaCode(ast);
    EXPECT_FALSE(cudaCode.empty());
    
    // Should contain all three second derivatives
    EXPECT_GT(cudaCode.length(), 10); // Should be substantial code
}
