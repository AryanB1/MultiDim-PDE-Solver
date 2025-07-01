#pragma once

#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <string>
#include "parser.h"

class PDESolver {
public:
    PDESolver(int nx, int ny, int nz, double dx, double dy, double dz, double dt);
    void initialize();
    void setInitialConditions();
    void setEquation(const std::string& equation);
    void solve(int timeSteps);
    void saveSliceToFile(const std::string& filename, int zSlice) const;
    const std::vector<double>& getGrid() const;
    std::string getEquationInfo() const;

private:
    int nx_, ny_, nz_;
    double dx_, dy_, dz_, dt_;
    std::vector<double> grid_;
    std::string equation_;
    std::string generatedKernelCode_;
    EquationParser parser_;

    // Helper function to get grid index
    inline int getIndex(int i, int j, int k) const {
        return (k * ny_ + j) * nx_ + i;
    }
    
    // Generate CUDA kernel code from parsed equation
    void generateKernelCode();
};
