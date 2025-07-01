# Tests CMakeLists.txt for CUDA PDE Solver

# Enable testing
include(GoogleTest)

# Test executable
add_executable(CudaProjTests
    test_parser.cpp
    test_solver.cpp
    test_derivatives.cpp
    test_integration.cpp
    test_main.cpp
)

# Set CUDA properties for tests
set_property(TARGET CudaProjTests PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET CudaProjTests PROPERTY CUDA_ARCHITECTURES "52;61;75;86")

# Link libraries
target_link_libraries(CudaProjTests 
    PRIVATE
    CudaProjCore
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
)

# Compiler-specific options for tests
target_compile_options(CudaProjTests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
    >
)

# Include directories
target_include_directories(CudaProjTests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Add preprocessor definitions for tests
target_compile_definitions(CudaProjTests PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
    GTEST_HAS_TR1_TUPLE=0
)

# Discover tests for CTest
gtest_discover_tests(CudaProjTests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Add custom target to run tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS CudaProjTests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
