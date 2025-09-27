cmake_minimum_required(VERSION 3.10)
project(AscensionRadar2026)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add your source files here
add_executable(main main.cpp)

# Link OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS})
