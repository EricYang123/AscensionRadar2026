#include "layers.h"
#include "common.h"
#include <iostream>
#include <opencv2/opencv.hpp>

void Layers::showLargestDetection(Mat image, const vector<Detection>& output, Mat& outputImage){
    vector<int> detectionSizes;
    for(int i = 0; i < output.size(); i++){
        auto detection = output[i];
        auto box = detection.bbox;
        int boxSize = box.width * box.height;
        detectionSizes.push_back(boxSize);
    }
    auto maxVal = max_element(detectionSizes.begin(), detectionSizes.end());
    int maxIndex = distance(detectionSizes.begin(), maxVal);
    auto object = output[maxIndex];
    auto box = object.bbox;
    const float ratio_h = model_h1 / (float)image.rows;
    const float ratio_w = model_w1 / (float)image.cols;
    if(ratio_h > ratio_w){
            box.x = box.x / ratio_w;
            box.y = (box.y - (model_h1 - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_h;
        } else{
            box.x = (box.x - (model_w1 - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_h;
        }
    outputImage = image(box);
    cout << object.class_id << " " << CLASS_NAMES.at(object.class_id) << endl;

}