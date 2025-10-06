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

void Yolo::infer(){
    const char* input_name = engine->getIOTensorName(0);
    const char* output_name = engine->getIOTensorName(1);

    int nbBindings = engine->getNbIOTensors();
    for (int i = 0; i < nbBindings; ++i) {
        std::cout << "Binding " << i << ": " << engine->getIOTensorName(i) << std::endl;
    }

    context->setTensorAddress(input_name, gpu_buffers[0]);
    context->setOutputTensorAddress(output_name, gpu_buffers[1]);

    cout << "set addresses" << endl;

    this->context->enqueueV3(this->stream);
    cout << "infered" << endl;
}

void Yolo::postprocess(vector<Detection>& output){
    cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    for(int i = 0; i < det_output.cols; i++){
        const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        Point class_id_point;
        double score;
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if(score > conf_threshold){
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }

    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, conf_threshold,nms_threshold, nms_result);

    for(int i = 0; i < nms_result.size(); i++){
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        output.push_back(result);
    }
}

void Yolo::display(Mat& image, const vector<Detection>& output){
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    for(int i = 0; i < output.size(); i++){
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        Scalar colour = Scalar(COLOURS[class_id][0], COLOURS[class_id][1], COLOURS[class_id][2]);

        if(ratio_h > ratio_w){
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_h;
        } else{
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_h;
        }

        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), colour, 3);

        string class_string = CLASS_NAMES[class_id] + ' '+ to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, colour, FILLED);
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}