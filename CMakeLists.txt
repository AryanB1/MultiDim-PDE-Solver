cmake_minimum_required(VERSION 3.18)
project(CudaProjDifferentials LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(GTest)
enable_testing()

# Common CUDA properties function
function(setup_cuda_target target)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "52;61;75;86"
    )
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
    )
    target_compile_definitions(${target} PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
    )
endfunction()

# Core library
add_library(CudaProjCore STATIC src/solver.cu src/parser.cpp)
target_include_directories(CudaProjCore PUBLIC include)
target_link_libraries(CudaProjCore PUBLIC CUDA::cudart)
setup_cuda_target(CudaProjCore)

# Main executable
add_executable(CudaProjDifferentials src/main.cpp)
target_link_libraries(CudaProjDifferentials PRIVATE CudaProjCore)
setup_cuda_target(CudaProjDifferentials)

# Tests
add_subdirectory(tests)
