#include "Yolo.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <memory>
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
    yolo.init("yolo11n.engine", logger);
    VideoCapture cap(0);
    namedWindow("Webcam", WINDOW_AUTOSIZE);
    Mat frame;
    cap >> frame;
    yolo.preprocess(frame);
    yolo.infer();
    vector<Detection> detected;
    yolo.postprocess(detected);
    cout << detected.size() << endl;
    yolo.display(frame, detected);
    imshow("Webcam", frame);
    waitKey(0);
    cap.release();
    cv::destroyAllWindows();
    return 0;
}