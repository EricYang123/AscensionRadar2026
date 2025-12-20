#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <opencv2/opencv.hpp>
#include "common.h"
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;
using namespace cv;

class Yolo{
public:

    void build(string modelPath, ILogger& logger);

    void init(string modelPath, ILogger& logger);

    void preprocess(Mat& image);

    void infer();

    void postprocess(vector<Detection>& output);

    void display(Mat& image, const vector<Detection>& output);



private:

    float conf_threshold = 0.5f;

    float nms_threshold = 0.8f;

    const int MAX_IMAGE_SIZE = 4096 * 4096;

    IRuntime* runtime;

    ICudaEngine* engine;

    IExecutionContext* context;

    int input_h;

    int input_w;

    int num_detections;

    int detection_attribute_size;

    int num_classes;

    float* cpu_output_buffer;

    float* gpu_buffers[2];
    
    cudaStream_t stream;
};