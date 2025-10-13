#include "Yolo.h"
#include "layers.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <memory>
#include <chrono>
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;
using namespace cv;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

int main(){
    Logger logger;
    Yolo yolo;
    Layers layers;

    yolo.init("yolo11n.engine", logger);
    VideoCapture cap(0);
    namedWindow("Webcam", WINDOW_AUTOSIZE);
    namedWindow("Cropped Image", WINDOW_AUTOSIZE);

    Mat frame;
    Mat cropped;
    auto start = std::chrono::high_resolution_clock::now();
    cap >> frame;

    yolo.preprocess(frame);

    yolo.infer();

    vector<Detection> detected;
    yolo.postprocess(detected);
    yolo.display(frame, detected);
    layers.showLargestDetection(frame, detected, cropped);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Elapsed time: " << elapsed.count() << endl;

    imshow("Webcam", frame);
    imshow("Cropped Image", cropped);
    waitKey(0);

    cap.release();
    cv::destroyAllWindows();
    return 0;
}