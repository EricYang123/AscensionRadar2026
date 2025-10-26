#include "Resnet.h"
#include "cuda.h"
#include "common.h"
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

void Resnet::init(string modelPath, ILogger& logger){
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
    context->setInputShape(input_name, Dims{4, {1, 3, 224, 224}});
    auto input_dims = engine->getTensorShape(input_name);
    auto output_dims = engine->getTensorShape(output_name);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    num_classes = output_dims.d[1];
    cpu_output_buffer = new float[num_classes];
    cudaMalloc((void**)&gpu_buffers[0], 3 * input_w * input_h * sizeof(float));
    cudaMalloc((void**)&gpu_buffers[1], num_classes * sizeof(float));
    cuda_preprocess_init(MAX_IMAGE_SIZE);
    cudaStreamCreate(&stream);
}

void Resnet::preprocess(Mat& image){
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    cudaStreamSynchronize(stream);
}

void Resnet::infer(){
    const char* input_name = engine->getIOTensorName(0);
    const char* output_name = engine->getIOTensorName(1);

    int nbBindings = engine->getNbIOTensors();

    context->setTensorAddress(input_name, gpu_buffers[0]);
    context->setOutputTensorAddress(output_name, gpu_buffers[1]);

    this->context->enqueueV3(this->stream);
}

void Resnet::postprocess(){
    cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], num_classes * sizeof(float), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float sum_exp = 0.0f;
    std::vector<float> probabilities(num_classes);

    for (int i = 0; i < num_classes; ++i) {
        probabilities[i] = std::exp(cpu_output_buffer[i]);
        sum_exp += probabilities[i];
    }

    int max_class = 0;
    float max_score = cpu_output_buffer[0];

    for(int i = 0; i < num_classes; i++){
        if(cpu_output_buffer[i] > max_score){
            max_score = cpu_output_buffer[i];
            max_class = i;
        }
    }

    cout << "Class ID: " << max_class << " Confidence: " << max_score << endl;

}