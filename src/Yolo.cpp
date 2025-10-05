#include "Yolo.h"
#include "cuda.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <memory>
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;
using namespace cv;

struct Detection {
    float conf; 
    int class_id; 
    Rect bbox;
};

struct AffineMatrix {
    float value[6];
};

void Yolo::build(string modelPath, ILogger& logger){
    IBuilder* builder = createInferBuilder(logger);
    INetworkDefinition* network = builder->createNetworkV2(0);
    IParser* parser = createParser(*network, logger);
    bool parsed = parser->parseFromFile(modelPath.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
    for(int32_t i = 0; i < parser->getNbErrors(); ++i){
        cout << parser->getError(i)->desc() << endl;
    }
}

void Yolo::init(string modelPath, ILogger& logger){
    ifstream engineStream(modelPath, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();
    auto input_name = engine->getIOTensorName(0);
    auto output_name = engine->getIOTensorName(1);
    auto input_dims = engine->getTensorShape(input_name);
    auto output_dims = engine->getTensorShape(output_name);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    detection_attribute_size = output_dims.d[1];
    num_detections = output_dims.d[2];
    num_classes = detection_attribute_size - 4;
    cpu_output_buffer = new float[detection_attribute_size * num_detections];
    cudaMalloc((void**)&gpu_buffers[0], 3 * input_w * input_h * sizeof(float));
    cudaMalloc((void**)&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float));
    cuda_preprocess_init(MAX_IMAGE_SIZE);
    cudaStreamCreate(&stream);
}

void Yolo::preprocess(Mat& image){
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    cudaStreamSynchronize(stream);
}