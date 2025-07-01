#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>

// Forward declarations
struct ASTNode;
using ASTNodePtr = std::unique_ptr<ASTNode>;

// Enumeration for different node types
enum class NodeType {
    NUMBER,
    VARIABLE,
    BINARY_OP,
    UNARY_OP,
    FUNCTION,
    DERIVATIVE
};

// Enumeration for operators
enum class Operator {
    ADD, SUB, MUL, DIV, POW
};

// Enumeration for functions
enum class Function {
    SIN, COS, TAN, EXP, LOG, SQRT, ABS
};

// Abstract syntax tree node
struct ASTNode {
    NodeType type;
    
    // Data for different node types
    double value;              // For NUMBER
    std::string variable;      // For VARIABLE
    Operator op;              // For BINARY_OP, UNARY_OP
    Function func;            // For FUNCTION
    std::vector<ASTNodePtr> children;
    
    ASTNode(NodeType t) : type(t), value(0.0), op(Operator::ADD), func(Function::SIN) {}
};

// Token structure for lexer
struct Token {
    enum Type {
        NUMBER, IDENTIFIER, OPERATOR, LPAREN, RPAREN, COMMA, END
    };
    
    Type type;
    std::string value;
    double numValue;
    
    Token(Type t, const std::string& v = "", double n = 0.0) 
        : type(t), value(v), numValue(n) {}
};

class EquationParser {
public:
    EquationParser();
    
    // Parse equation string into AST
    ASTNodePtr parse(const std::string& equation);
    
    // Generate CUDA code for the parsed equation
    std::string generateCudaCode(const ASTNodePtr& ast) const;
    
    // Validate that the equation is a valid PDE
    bool validatePDE(const ASTNodePtr& ast) const;
    
    // Get list of variables used in the equation
    std::vector<std::string> getVariables(const ASTNodePtr& ast) const;

private:
    std::vector<Token> tokenize(const std::string& equation);
    ASTNodePtr parseExpression();
    ASTNodePtr parseTerm();
    ASTNodePtr parseFactor();
    ASTNodePtr parseFunction(const std::string& funcName);
    
    std::vector<Token> tokens_;
    size_t currentToken_;
    
    // Helper functions
    bool isFunction(const std::string& name) const;
    Function getFunctionType(const std::string& name) const;
    std::string generateNodeCode(const ASTNodePtr& node) const;
    void collectVariables(const ASTNodePtr& node, std::vector<std::string>& vars) const;
};

// Derivative parser for handling d/dx, d2/dx2, etc.
class DerivativeParser {
public:
    struct DerivativeInfo {
        std::string variable;  // x, y, z, t
        int order;            // 1 for first derivative, 2 for second, etc.
        std::string expression; // the expression being differentiated
    };
    
    static bool isDerivative(const std::string& expr);
    static DerivativeInfo parseDerivative(const std::string& expr);
};
