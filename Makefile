# Compiler and flags
CXX := cl
NVCC := nvcc

# Windows-specific settings
EXECUTABLE_EXT := .exe
RM_CMD := del /Q /S
MKDIR_CMD := if not exist "$(1)" mkdir "$(1)"

CXXFLAGS := /std:c++17 /I"include" /I"build\deps\googletest-src\googletest\include" /EHsc
NVCCFLAGS := -std=c++17 -Iinclude --expt-relaxed-constexpr --extended-lambda -x cu
LDFLAGS := /LIBPATH:"$(CUDA_PATH)\lib\x64"
LIBS := cudart.lib nvrtc.lib

# Directories
SRC_DIR := src
TEST_SRC_DIR := tests
OBJ_DIR := build/obj
BIN_DIR := build/bin
TEST_OBJ_DIR := build/test_obj
TEST_BIN_DIR := build/test_bin

# Source files (using forward slashes)
CORE_SRCS_CPP := $(wildcard $(SRC_DIR)/*.cpp)
CORE_SRCS_CU := $(wildcard $(SRC_DIR)/*.cu)
MAIN_SRC := $(SRC_DIR)/main.cpp
TEST_SRCS := $(wildcard $(TEST_SRC_DIR)/*.cpp)

# Object files (using Windows paths)
CORE_OBJS_CPP := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.obj,$(filter-out $(MAIN_SRC),$(CORE_SRCS_CPP)))
CORE_OBJS_CU := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.obj,$(CORE_SRCS_CU))
MAIN_OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.obj,$(MAIN_SRC))
TEST_OBJS := $(patsubst $(TEST_SRC_DIR)/%.cpp,$(TEST_OBJ_DIR)/%.obj,$(TEST_SRCS))

# Targets
EXECUTABLE := $(BIN_DIR)\CudaProjDifferentials$(EXECUTABLE_EXT)
TEST_EXECUTABLE := $(TEST_BIN_DIR)\CudaProjTests$(EXECUTABLE_EXT)

.PHONY: all clean run test

all: $(EXECUTABLE)

run: $(EXECUTABLE)
	.\$(EXECUTABLE)

test: $(TEST_EXECUTABLE)
	.\$(TEST_EXECUTABLE)

# Main executable rule
$(EXECUTABLE): $(MAIN_OBJ) $(CORE_OBJS_CPP) $(CORE_OBJS_CU)
	$(call MKDIR_CMD,$(BIN_DIR))
	$(CXX) $(CXXFLAGS) $^ /Fe:$@ $(LDFLAGS) $(LIBS)

# Test executable rule
$(TEST_EXECUTABLE): $(TEST_OBJS) $(CORE_OBJS_CPP) $(CORE_OBJS_CU)
	$(call MKDIR_CMD,$(TEST_BIN_DIR))
	$(CXX) $(CXXFLAGS) $^ /Fe:$@ $(LDFLAGS) $(LIBS) /LIBPATH:"build\lib" gtest.lib gtest_main.lib

# C++ object file rule
$(OBJ_DIR)\%.obj: $(SRC_DIR)\%.cpp
	$(call MKDIR_CMD,$(OBJ_DIR))
	$(CXX) $(CXXFLAGS) /c $< /Fo:$@

# CUDA object file rule
$(OBJ_DIR)\%.obj: $(SRC_DIR)\%.cu
	$(call MKDIR_CMD,$(OBJ_DIR))
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Test object file rule
$(TEST_OBJ_DIR)\%.obj: $(TEST_SRC_DIR)\%.cpp
	$(call MKDIR_CMD,$(TEST_OBJ_DIR))
	$(CXX) $(CXXFLAGS) /c $< /Fo:$@

clean:
	$(RM_CMD) $(OBJ_DIR) $(BIN_DIR) $(TEST_OBJ_DIR) $(TEST_BIN_DIR)
