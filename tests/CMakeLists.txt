include(GoogleTest)
find_package(GTest REQUIRED)

add_executable(CudaProjTests
    test_parser.cpp
    test_solver.cpp
    test_derivatives.cpp
    test_integration.cpp
    test_main.cpp
)

target_link_libraries(CudaProjTests 
    PRIVATE
    CudaProjCore
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(CudaProjTests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
