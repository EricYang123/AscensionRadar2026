#include "common.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


class Layers{
public:
    void showLargestDetection(Mat image, const vector<Detection>& output, Mat& outputImage);

private: 
    int model_w1 = 640;
    int model_h1 = 640;

};