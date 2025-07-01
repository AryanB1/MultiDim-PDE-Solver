#include "parser.h"
#include <regex>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>

EquationParser::EquationParser() : currentToken_(0) {}

ASTNodePtr EquationParser::parse(const std::string& equation) {
    tokens_ = tokenize(equation);
    currentToken_ = 0;
    return parseExpression();
}

std::vector<Token> EquationParser::tokenize(const std::string& equation) {
    std::vector<Token> tokens;
    std::string cleaned = equation;
    
    // Remove spaces
    cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(), ::isspace), cleaned.end());
    
    size_t i = 0;
    while (i < cleaned.length()) {
        char c = cleaned[i];
        
        if (std::isdigit(c) || c == '.') {
            // Parse number
            std::string numStr;
            while (i < cleaned.length() && (std::isdigit(cleaned[i]) || cleaned[i] == '.')) {
                numStr += cleaned[i++];
            }
            tokens.emplace_back(Token::NUMBER, numStr, std::stod(numStr));
        }
        else if (std::isalpha(c)) {
            // Parse identifier or function
            std::string id;
            while (i < cleaned.length() && (std::isalnum(cleaned[i]) || cleaned[i] == '_')) {
                id += cleaned[i++];
            }
            
            // Check for derivatives like d2u/dx2
            if (id == "d" && i < cleaned.length() && std::isdigit(cleaned[i])) {
                // Handle derivative notation
                std::string derivStr = "d";
                while (i < cleaned.length() && (std::isalnum(cleaned[i]) || cleaned[i] == '/' || cleaned[i] == '_')) {
                    derivStr += cleaned[i++];
                }
                tokens.emplace_back(Token::IDENTIFIER, derivStr);
            }
            else {
                tokens.emplace_back(Token::IDENTIFIER, id);
            }
        }
        else if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^') {
            tokens.emplace_back(Token::OPERATOR, std::string(1, c));
            i++;
        }
        else if (c == '(') {
            tokens.emplace_back(Token::LPAREN);
            i++;
        }
        else if (c == ')') {
            tokens.emplace_back(Token::RPAREN);
            i++;
        }
        else if (c == ',') {
            tokens.emplace_back(Token::COMMA);
            i++;
        }
        else {
            i++; // Skip unknown characters
        }
    }
    
    tokens.emplace_back(Token::END);
    return tokens;
}

ASTNodePtr EquationParser::parseExpression() {
    auto left = parseTerm();
    
    while (currentToken_ < tokens_.size() && 
           tokens_[currentToken_].type == Token::OPERATOR &&
           (tokens_[currentToken_].value == "+" || tokens_[currentToken_].value == "-")) {
        
        std::string op = tokens_[currentToken_].value;
        currentToken_++;
        auto right = parseTerm();
        
        auto node = std::make_unique<ASTNode>(NodeType::BINARY_OP);
        node->op = (op == "+") ? Operator::ADD : Operator::SUB;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }
    
    return left;
}

ASTNodePtr EquationParser::parseTerm() {
    auto left = parseFactor();
    
    while (currentToken_ < tokens_.size() && 
           tokens_[currentToken_].type == Token::OPERATOR &&
           (tokens_[currentToken_].value == "*" || tokens_[currentToken_].value == "/")) {
        
        std::string op = tokens_[currentToken_].value;
        currentToken_++;
        auto right = parseFactor();
        
        auto node = std::make_unique<ASTNode>(NodeType::BINARY_OP);
        node->op = (op == "*") ? Operator::MUL : Operator::DIV;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }
    
    return left;
}

ASTNodePtr EquationParser::parseFactor() {
    if (currentToken_ >= tokens_.size()) {
        throw std::runtime_error("Unexpected end of expression");
    }
    
    Token& token = tokens_[currentToken_];
    
    if (token.type == Token::NUMBER) {
        auto node = std::make_unique<ASTNode>(NodeType::NUMBER);
        node->value = token.numValue;
        currentToken_++;
        return node;
    }
    else if (token.type == Token::IDENTIFIER) {
        std::string id = token.value;
        currentToken_++;
        
        // Check if it's a function call
        if (currentToken_ < tokens_.size() && tokens_[currentToken_].type == Token::LPAREN) {
            return parseFunction(id);
        }
        // Check if it's a derivative
        else if (DerivativeParser::isDerivative(id)) {
            auto node = std::make_unique<ASTNode>(NodeType::DERIVATIVE);
            node->variable = id;
            return node;
        }
        else {
            // It's a variable
            auto node = std::make_unique<ASTNode>(NodeType::VARIABLE);
            node->variable = id;
            return node;
        }
    }
    else if (token.type == Token::LPAREN) {
        currentToken_++;
        auto expr = parseExpression();
        if (currentToken_ >= tokens_.size() || tokens_[currentToken_].type != Token::RPAREN) {
            throw std::runtime_error("Missing closing parenthesis");
        }
        currentToken_++;
        return expr;
    }
    else if (token.type == Token::OPERATOR && token.value == "-") {
        // Unary minus
        currentToken_++;
        auto operand = parseFactor();
        auto node = std::make_unique<ASTNode>(NodeType::UNARY_OP);
        node->op = Operator::SUB;
        node->children.push_back(std::move(operand));
        return node;
    }
    
    throw std::runtime_error("Unexpected token: " + token.value);
}

ASTNodePtr EquationParser::parseFunction(const std::string& funcName) {
    if (!isFunction(funcName)) {
        throw std::runtime_error("Unknown function: " + funcName);
    }
    
    auto node = std::make_unique<ASTNode>(NodeType::FUNCTION);
    node->func = getFunctionType(funcName);
    
    // Expect opening parenthesis
    if (currentToken_ >= tokens_.size() || tokens_[currentToken_].type != Token::LPAREN) {
        throw std::runtime_error("Expected '(' after function name");
    }
    currentToken_++;
    
    // Parse argument
    node->children.push_back(parseExpression());
    
    // Expect closing parenthesis
    if (currentToken_ >= tokens_.size() || tokens_[currentToken_].type != Token::RPAREN) {
        throw std::runtime_error("Expected ')' after function argument");
    }
    currentToken_++;
    
    return node;
}

bool EquationParser::isFunction(const std::string& name) const {
    static const std::vector<std::string> functions = {
        "sin", "cos", "tan", "exp", "log", "sqrt", "abs"
    };
    return std::find(functions.begin(), functions.end(), name) != functions.end();
}

Function EquationParser::getFunctionType(const std::string& name) const {
    if (name == "sin") return Function::SIN;
    if (name == "cos") return Function::COS;
    if (name == "tan") return Function::TAN;
    if (name == "exp") return Function::EXP;
    if (name == "log") return Function::LOG;
    if (name == "sqrt") return Function::SQRT;
    if (name == "abs") return Function::ABS;
    throw std::runtime_error("Unknown function: " + name);
}

std::string EquationParser::generateCudaCode(const ASTNodePtr& ast) const {
    return generateNodeCode(ast);
}

std::string EquationParser::generateNodeCode(const ASTNodePtr& node) const {
    if (!node) return "0.0";
    
    switch (node->type) {
        case NodeType::NUMBER:
            return std::to_string(node->value);
            
        case NodeType::VARIABLE:
            if (node->variable == "u") return "grid[idx]";
            if (node->variable == "x") return "(" + std::to_string(1.0) + " * i * dx)";
            if (node->variable == "y") return "(" + std::to_string(1.0) + " * j * dy)";
            if (node->variable == "z") return "(" + std::to_string(1.0) + " * k * dz)";
            if (node->variable == "t") return "currentTime";
            return node->variable;
            
        case NodeType::BINARY_OP: {
            std::string left = generateNodeCode(node->children[0]);
            std::string right = generateNodeCode(node->children[1]);
            switch (node->op) {
                case Operator::ADD: return "(" + left + " + " + right + ")";
                case Operator::SUB: return "(" + left + " - " + right + ")";
                case Operator::MUL: return "(" + left + " * " + right + ")";
                case Operator::DIV: return "(" + left + " / " + right + ")";
                case Operator::POW: return "pow(" + left + ", " + right + ")";
            }
            break;
        }
        
        case NodeType::UNARY_OP: {
            std::string operand = generateNodeCode(node->children[0]);
            if (node->op == Operator::SUB) return "(-" + operand + ")";
            break;
        }
        
        case NodeType::FUNCTION: {
            std::string arg = generateNodeCode(node->children[0]);
            switch (node->func) {
                case Function::SIN: return "sin(" + arg + ")";
                case Function::COS: return "cos(" + arg + ")";
                case Function::TAN: return "tan(" + arg + ")";
                case Function::EXP: return "exp(" + arg + ")";
                case Function::LOG: return "log(" + arg + ")";
                case Function::SQRT: return "sqrt(" + arg + ")";
                case Function::ABS: return "abs(" + arg + ")";
            }
            break;
        }
        
        case NodeType::DERIVATIVE: {
            auto derivInfo = DerivativeParser::parseDerivative(node->variable);
            if (derivInfo.variable == "x" && derivInfo.order == 2) {
                return "(grid[((k*ny + j)*nx + i+1)] - 2.0*grid[idx] + grid[((k*ny + j)*nx + i-1)]) / (dx*dx)";
            }
            else if (derivInfo.variable == "y" && derivInfo.order == 2) {
                return "(grid[((k*ny + j+1)*nx + i)] - 2.0*grid[idx] + grid[((k*ny + j-1)*nx + i)]) / (dy*dy)";
            }
            else if (derivInfo.variable == "z" && derivInfo.order == 2) {
                return "(grid[(((k+1)*ny + j)*nx + i)] - 2.0*grid[idx] + grid[(((k-1)*ny + j)*nx + i)]) / (dz*dz)";
            }
            // Add more derivative cases as needed
            return "0.0";
        }
    }
    
    return "0.0";
}

bool EquationParser::validatePDE(const ASTNodePtr& ast) const {
    // Basic validation - should contain derivatives and variables
    auto variables = getVariables(ast);
    bool hasDerivatives = false;
    
    // Check if AST contains derivative nodes
    std::function<void(const ASTNodePtr&)> checkDerivatives = [&](const ASTNodePtr& node) {
        if (!node) return;
        if (node->type == NodeType::DERIVATIVE) {
            hasDerivatives = true;
            return;
        }
        for (const auto& child : node->children) {
            checkDerivatives(child);
        }
    };
    
    checkDerivatives(ast);
    return hasDerivatives && !variables.empty();
}

std::vector<std::string> EquationParser::getVariables(const ASTNodePtr& ast) const {
    std::vector<std::string> vars;
    collectVariables(ast, vars);
    
    // Remove duplicates
    std::sort(vars.begin(), vars.end());
    vars.erase(std::unique(vars.begin(), vars.end()), vars.end());
    
    return vars;
}

void EquationParser::collectVariables(const ASTNodePtr& node, std::vector<std::string>& vars) const {
    if (!node) return;
    
    if (node->type == NodeType::VARIABLE) {
        vars.push_back(node->variable);
    }
    
    for (const auto& child : node->children) {
        collectVariables(child, vars);
    }
}

// DerivativeParser implementation
bool DerivativeParser::isDerivative(const std::string& expr) {
    // Match patterns like: d2u/dx2, du/dx, d2u/dy2, etc.
    std::regex derivRegex(R"(d(\d*)u/d([xyz])(\d*))");
    return std::regex_match(expr, derivRegex);
}

DerivativeParser::DerivativeInfo DerivativeParser::parseDerivative(const std::string& expr) {
    DerivativeInfo info;
    std::regex derivRegex(R"(d(\d*)u/d([xyz])(\d*))");
    std::smatch match;
    
    if (std::regex_match(expr, match, derivRegex)) {
        // Extract order from numerator (default 1)
        std::string orderStr = match[1].str();
        info.order = orderStr.empty() ? 1 : std::stoi(orderStr);
        
        // Extract variable
        info.variable = match[2].str();
        
        // Extract order from denominator (should match numerator)
        std::string denomOrderStr = match[3].str();
        if (!denomOrderStr.empty()) {
            int denomOrder = std::stoi(denomOrderStr);
            if (denomOrder != info.order) {
                throw std::runtime_error("Derivative order mismatch in: " + expr);
            }
        }
        
        info.expression = "u";
    }
    else {
        throw std::runtime_error("Invalid derivative format: " + expr);
    }
    
    return info;
}
