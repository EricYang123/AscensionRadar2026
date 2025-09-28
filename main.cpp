#include "functions.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <fstream>
#include <memory>
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;

const int MAX_IMAGE_SIZE = 4096 * 4096;
static uint8_t* img_buffer_host = nullptr;
static uint8_t* img_buffer_device = nullptr;
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

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

void build(string modelPath){
    IBuilder* builder = createInferBuilder(logger);
    INetworkDefinition* network = builder->createNetworkV2(0);
    IParser* parser = createParser(*network, logger);
    bool parsed = parser->parseFromFile(modelPath.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
    for(int32_t i = 0; i < parser->getNbErrors(); ++i){
        cout << parser->getError(i)->desc() << endl;
    }
}

void init(string modelPath, ILogger& logger){
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
    cudaMallocHost((void**)&img_buffer_host, MAX_IMAGE_SIZE * 3);
    cudaMalloc((void**)&img_buffer_device, MAX_IMAGE_SIZE * 3);
}

int main(){
    Logger logger;
    init("yolov12n.engine", logger);
    cout << "eh" << endl;
    return 0;
}