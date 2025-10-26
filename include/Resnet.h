#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <opencv2/opencv.hpp>
#include "common.h"
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;
using namespace cv;

class Resnet{
public:
    void init(string modelPath, ILogger& logger);

    void preprocess(Mat& image);

    void infer();

    void postprocess();

private:
    int input_h;

    int input_w;

    int num_classes;

    float* cpu_output_buffer;

    float* gpu_buffers[2];

    const int MAX_IMAGE_SIZE = 4096 * 4096;
    
    cudaStream_t stream;

    IRuntime* runtime;

    ICudaEngine* engine;

    IExecutionContext* context;
    
};