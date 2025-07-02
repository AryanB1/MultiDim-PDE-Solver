#include <gtest/gtest.h>
#include <iostream>
#include <cuda_runtime.h>

class CudaEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        // Initialize CUDA context
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        
        if (error != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
            GTEST_SKIP() << "CUDA is not available, skipping CUDA-related tests";
            return;
        }
        
        if (deviceCount == 0) {
            GTEST_SKIP() << "No CUDA devices found, skipping CUDA-related tests";
            return;
        }
        
        // Set device 0 as default
        cudaSetDevice(0);
        
        // Print device information for debugging
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Running tests on CUDA device: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    }
    
    void TearDown() override {
        // Reset CUDA device
        cudaDeviceReset();
    }
};

int main(int argc, char** argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Add global CUDA environment
    ::testing::AddGlobalTestEnvironment(new CudaEnvironment);
    
    // Run all tests
    return RUN_ALL_TESTS();
}
