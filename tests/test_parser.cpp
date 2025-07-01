/**
 * @file test_parser.cpp
 * @brief Unit tests for the equation parser
 */

#include <gtest/gtest.h>
#include <memory>
#include "parser.h"

/**
 * @class ParserTest
 * @brief Test fixture for parser tests
 */
class ParserTest : public ::testing::Test {
protected:
    void SetUp() override {
        parser = std::make_unique<EquationParser>();
    }
    
    void TearDown() override {
        parser.reset();
    }
    
    std::unique_ptr<EquationParser> parser;
};

// Basic tokenization tests
TEST_F(ParserTest, TokenizeSimpleNumber) {
    auto ast = parser->parse("42.5");
    ASSERT_NE(ast, nullptr);
    EXPECT_EQ(ast->type, NodeType::NUMBER);
    EXPECT_DOUBLE_EQ(ast->value, 42.5);
}

TEST_F(ParserTest, TokenizeVariable) {
    auto ast = parser->parse("x");
    ASSERT_NE(ast, nullptr);
    EXPECT_EQ(ast->type, NodeType::VARIABLE);
    EXPECT_EQ(ast->variable, "x");
}

TEST_F(ParserTest, TokenizeSimpleAddition) {
    auto ast = parser->parse("x + y");
    ASSERT_NE(ast, nullptr);
    EXPECT_EQ(ast->type, NodeType::BINARY_OP);
    EXPECT_EQ(ast->op, Operator::ADD);
    ASSERT_EQ(ast->children.size(), 2);
    EXPECT_EQ(ast->children[0]->variable, "x");
    EXPECT_EQ(ast->children[1]->variable, "y");
}

TEST_F(ParserTest, TokenizeOperatorPrecedence) {
    auto ast = parser->parse("x + y * z");
    ASSERT_NE(ast, nullptr);
    EXPECT_EQ(ast->type, NodeType::BINARY_OP);
    EXPECT_EQ(ast->op, Operator::ADD);
    
    // Should be parsed as x + (y * z)
    ASSERT_EQ(ast->children.size(), 2);
    EXPECT_EQ(ast->children[0]->variable, "x");
    EXPECT_EQ(ast->children[1]->type, NodeType::BINARY_OP);
    EXPECT_EQ(ast->children[1]->op, Operator::MUL);
}

TEST_F(ParserTest, TokenizeParentheses) {
    auto ast = parser->parse("(x + y) * z");
    ASSERT_NE(ast, nullptr);
    EXPECT_EQ(ast->type, NodeType::BINARY_OP);
    EXPECT_EQ(ast->op, Operator::MUL);
    
    // Should be parsed as (x + y) * z
    ASSERT_EQ(ast->children.size(), 2);
    EXPECT_EQ(ast->children[0]->type, NodeType::BINARY_OP);
    EXPECT_EQ(ast->children[0]->op, Operator::ADD);
    EXPECT_EQ(ast->children[1]->variable, "z");
}

// Function parsing tests
TEST_F(ParserTest, ParseSinFunction) {
    auto ast = parser->parse("sin(x)");
    ASSERT_NE(ast, nullptr);
    EXPECT_EQ(ast->type, NodeType::FUNCTION);
    EXPECT_EQ(ast->func, Function::SIN);
    ASSERT_EQ(ast->children.size(), 1);
    EXPECT_EQ(ast->children[0]->variable, "x");
}

TEST_F(ParserTest, ParseNestedFunctions) {
    auto ast = parser->parse("sin(cos(x))");
    ASSERT_NE(ast, nullptr);
    EXPECT_EQ(ast->type, NodeType::FUNCTION);
    EXPECT_EQ(ast->func, Function::SIN);
    ASSERT_EQ(ast->children.size(), 1);
    EXPECT_EQ(ast->children[0]->type, NodeType::FUNCTION);
    EXPECT_EQ(ast->children[0]->func, Function::COS);
}

// Derivative parsing tests
TEST_F(ParserTest, ParseFirstDerivative) {
    EXPECT_TRUE(DerivativeParser::isDerivative("du/dx"));
    
    auto derivInfo = DerivativeParser::parseDerivative("du/dx");
    EXPECT_EQ(derivInfo.variable, "x");
    EXPECT_EQ(derivInfo.order, 1);
    EXPECT_EQ(derivInfo.expression, "u");
}

TEST_F(ParserTest, ParseSecondDerivative) {
    EXPECT_TRUE(DerivativeParser::isDerivative("d2u/dx2"));
    
    auto derivInfo = DerivativeParser::parseDerivative("d2u/dx2");
    EXPECT_EQ(derivInfo.variable, "x");
    EXPECT_EQ(derivInfo.order, 2);
    EXPECT_EQ(derivInfo.expression, "u");
}

TEST_F(ParserTest, ParseMixedDerivatives) {
    EXPECT_TRUE(DerivativeParser::isDerivative("d2u/dy2"));
    EXPECT_TRUE(DerivativeParser::isDerivative("d2u/dz2"));
    
    auto derivInfo = DerivativeParser::parseDerivative("d2u/dy2");
    EXPECT_EQ(derivInfo.variable, "y");
    EXPECT_EQ(derivInfo.order, 2);
}

// Complex equation parsing tests
TEST_F(ParserTest, ParseHeatEquation) {
    auto ast = parser->parse("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)");
    ASSERT_NE(ast, nullptr);
    
    // Validate that it contains derivatives
    auto variables = parser->getVariables(ast);
    EXPECT_TRUE(std::find(variables.begin(), variables.end(), "u") != variables.end());
}

TEST_F(ParserTest, ParseReactionDiffusion) {
    auto ast = parser->parse("0.1 * (d2u/dx2 + d2u/dy2) + u * (1 - u)");
    ASSERT_NE(ast, nullptr);
    
    // Should be valid PDE
    EXPECT_TRUE(parser->validatePDE(ast));
}

TEST_F(ParserTest, ParseEquationWithTrigonometric) {
    auto ast = parser->parse("0.05 * (d2u/dx2 + d2u/dy2) + sin(x) * cos(y)");
    ASSERT_NE(ast, nullptr);
    
    auto variables = parser->getVariables(ast);
    EXPECT_TRUE(std::find(variables.begin(), variables.end(), "x") != variables.end());
    EXPECT_TRUE(std::find(variables.begin(), variables.end(), "y") != variables.end());
    EXPECT_TRUE(std::find(variables.begin(), variables.end(), "u") != variables.end());
}

// CUDA code generation tests
TEST_F(ParserTest, GenerateCudaCodeSimple) {
    auto ast = parser->parse("x + y");
    ASSERT_NE(ast, nullptr);
    
    std::string cudaCode = parser->generateCudaCode(ast);
    EXPECT_FALSE(cudaCode.empty());
    EXPECT_NE(cudaCode.find("+"), std::string::npos);
}

TEST_F(ParserTest, GenerateCudaCodeWithFunction) {
    auto ast = parser->parse("sin(x)");
    ASSERT_NE(ast, nullptr);
    
    std::string cudaCode = parser->generateCudaCode(ast);
    EXPECT_FALSE(cudaCode.empty());
    EXPECT_NE(cudaCode.find("sin"), std::string::npos);
}

// Error handling tests
TEST_F(ParserTest, HandleInvalidSyntax) {
    EXPECT_THROW(parser->parse("x + + y"), std::exception);
    EXPECT_THROW(parser->parse("sin("), std::exception);
    EXPECT_THROW(parser->parse(")x"), std::exception);
}

TEST_F(ParserTest, HandleUnknownFunction) {
    EXPECT_THROW(parser->parse("unknownfunc(x)"), std::exception);
}

TEST_F(ParserTest, HandleMismatchedParentheses) {
    EXPECT_THROW(parser->parse("(x + y"), std::exception);
    EXPECT_THROW(parser->parse("x + y)"), std::exception);
}

// Validation tests
TEST_F(ParserTest, ValidatePDETrue) {
    auto ast = parser->parse("d2u/dx2 + d2u/dy2");
    ASSERT_NE(ast, nullptr);
    EXPECT_TRUE(parser->validatePDE(ast));
}

TEST_F(ParserTest, ValidatePDEFalse) {
    auto ast = parser->parse("x + y");
    ASSERT_NE(ast, nullptr);
    EXPECT_FALSE(parser->validatePDE(ast));
}

// Variable extraction tests
TEST_F(ParserTest, ExtractVariables) {
    auto ast = parser->parse("x + y * sin(z) + d2u/dx2");
    ASSERT_NE(ast, nullptr);
    
    auto variables = parser->getVariables(ast);
    EXPECT_TRUE(std::find(variables.begin(), variables.end(), "x") != variables.end());
    EXPECT_TRUE(std::find(variables.begin(), variables.end(), "y") != variables.end());
    EXPECT_TRUE(std::find(variables.begin(), variables.end(), "z") != variables.end());
    EXPECT_TRUE(std::find(variables.begin(), variables.end(), "u") != variables.end());
}

// Parametric tests for different equations
class EquationParsingTest : public ParserTest, 
                           public ::testing::WithParamInterface<std::pair<std::string, bool>> {
};

TEST_P(EquationParsingTest, ParseVariousEquations) {
    const auto& [equation, shouldSucceed] = GetParam();
    
    if (shouldSucceed) {
        auto ast = parser->parse(equation);
        EXPECT_NE(ast, nullptr) << "Failed to parse: " << equation;
        
        // Generate CUDA code to ensure it's valid
        std::string cudaCode = parser->generateCudaCode(ast);
        EXPECT_FALSE(cudaCode.empty()) << "Failed to generate CUDA code for: " << equation;
    } else {
        EXPECT_THROW(parser->parse(equation), std::exception) 
            << "Should have failed to parse: " << equation;
    }
}

INSTANTIATE_TEST_SUITE_P(
    ValidEquations,
    EquationParsingTest,
    ::testing::Values(
        // Valid equations
        std::make_pair("d2u/dx2", true),
        std::make_pair("0.1 * d2u/dx2", true),
        std::make_pair("d2u/dx2 + d2u/dy2", true),
        std::make_pair("0.1 * (d2u/dx2 + d2u/dy2 + d2u/dz2)", true),
        std::make_pair("u * (1 - u)", true),
        std::make_pair("sin(x) * cos(y)", true),
        std::make_pair("exp(x) + log(y)", true),
        std::make_pair("sqrt(abs(x))", true),
        
        // Invalid equations
        std::make_pair("", false),
        std::make_pair("x + + y", false),
        std::make_pair("sin(", false),
        std::make_pair(")x(", false),
        std::make_pair("unknownfunc(x)", false)
    )
);
