#!/bin/bash

# CUDA PDE Solver Build Tool
# Usage: ./cuda.sh [command]
# Commands:
#   build   - Build the project
#   run     - Run the executable (builds if needed)
#   clean   - Clean build directory
#   test    - Run tests
#   help    - Show this help

set -e  # Exit on any error

PROJECT_NAME="CudaProjDifferentials"
BUILD_DIR="build"
EXE_PATH="$BUILD_DIR/Debug/$PROJECT_NAME.exe"

show_help() {
    echo "CUDA PDE Solver Build Tool"
    echo ""
    echo "Usage: ./cuda.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build   Build the project"
    echo "  run     Run the executable (builds if needed)"
    echo "  clean   Clean build directory"
    echo "  test    Run tests"
    echo "  help    Show this help"
    echo ""
    echo "Examples:"
    echo "  ./cuda.sh build"
    echo "  ./cuda.sh run"
    echo "  ./cuda.sh clean"
}

build_project() {
    echo "🔨 Building CUDA PDE Solver..."
    
    # Create build directory if it doesn't exist
    if [ ! -d "$BUILD_DIR" ]; then
        echo "📁 Creating build directory..."
        mkdir "$BUILD_DIR"
    fi
    
    # Configure and build
    cd "$BUILD_DIR"
    cmake .. -G "Visual Studio 17 2022" -A x64
    cmake --build . --config Debug
    cd ..
    
    echo "✅ Build complete!"
}

run_project() {
    echo "🚀 Running CUDA PDE Solver..."
    
    # Build if executable doesn't exist
    if [ ! -f "$EXE_PATH" ]; then
        echo "📦 Executable not found, building first..."
        build_project
    fi
    
    # Run the executable
    echo "▶️  Starting application..."
    "./$EXE_PATH"
}

clean_project() {
    echo "🧹 Cleaning build directory..."
    
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        echo "✅ Build directory cleaned!"
    else
        echo "ℹ️  No build directory found."
    fi
}

run_tests() {
    echo "🧪 Running tests..."
    
    # Build if needed
    if [ ! -d "$BUILD_DIR" ]; then
        build_project
    fi
    
    cd "$BUILD_DIR"
    ctest --output-on-failure
    cd ..
    
    echo "✅ Tests complete!"
}

# Main script logic
case "${1:-}" in
    "build"|"b")
        build_project
        ;;
    "run"|"r")
        run_project
        ;;
    "clean"|"c")
        clean_project
        ;;
    "test"|"t")
        run_tests
        ;;
    "help"|"h"|"--help"|"-h")
        show_help
        ;;
    "")
        echo "❓ No command specified. Running build and run..."
        build_project
        run_project
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
