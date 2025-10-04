cmake_minimum_required(VERSION 3.18)

# Project declaration with C++ and CUDA support
project(AscensionRadar2026)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define the path to TensorRT installation
set(TENSORRT_PATH "/usr/include/aarch64-linux-gnu")  # Update this to the actual path for TensorRT

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Include directories for TensorRT
include_directories(
    include
    "/usr/include/aarch64-linux-gnu"
)

# Define source files (including CUDA sources)
set(SOURCES
    main.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
add_executable(Radar ${SOURCES})

# Specify include directories (modern CMake approach)
target_include_directories(Radar PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_PATH}
)

# Use proper library names (without path or .so extension)
set(TENSORRT_LIBS
    nvinfer
    nvonnxparser
    nvinfer_plugin
)

target_link_libraries(Radar PRIVATE
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)
