#include <iostream>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include "functions.h"

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

int testCUDA() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error_id) << std::endl;
        return -1;
    }

    std::cout << "CUDA device count: " << deviceCount << std::endl;
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
    }

    return 0;
}

int testTensorRT() {
    Logger logger;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder!" << std::endl;
        return -1;
    }

    std::cout << "TensorRT builder created successfully!" << std::endl;
    delete builder;
    return 0;
}

int main() {
    std::cout << "=== CUDA Test ===" << std::endl;
    if (testCUDA() != 0) return -1;

    std::cout << "\n=== TensorRT Test ===" << std::endl;
    if (testTensorRT() != 0) return -1;

    std::cout << "\nAll tests passed!" << std::endl;
    printSomething("Hello World!");

    return 0;
}
