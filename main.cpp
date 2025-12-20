#include "Yolo.h"
#include "layers.h"
#include "Resnet.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "SORT.h"
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
    // Logger logger2;
    // Logger logger3;
    Yolo yolo;
    // Yolo yolo2;
    // Resnet resnet;
    // Layers layers;
    SORT sort;

    yolo.init("/root/workspace/yolo11n.engine", logger);
    // yolo2.init("models/yolov12n.engine", logger2);
    // resnet.init("models/resnet18.engine", logger3);
    
    VideoCapture cap(0);
    // namedWindow("webcam", WINDOW_NORMAL);
    // namedWindow("Cropped Image", WINDOW_AUTOSIZE);

    Mat frame;
    Mat cropped;
    
    int frames = 0;
    while(true){
        cap >> frame;

        yolo.preprocess(frame);

        yolo.infer();

        vector<Detection> detected;
        yolo.postprocess(detected);
        auto start = std::chrono::high_resolution_clock::now();
        sort.sort(detected);
        auto timeNow = std::chrono::high_resolution_clock::now();
        // cout << "detection vector size: " << detected.size() << endl;
        yolo.display(frame, detected);
        // layers.showLargestDetection(frame, detected, cropped);
        
        imshow("webcam", frame);

        // yolo2.preprocess(cropped);
        // resnet.preprocess(frame);

        // yolo2.infer();
        // resnet.infer();

        // vector<Detection> detected2;
        // yolo2.postprocess(detected2);
        // resnet.postprocess();
        // yolo.display(cropped, detected2);

        // imshow("Cropped Image", cropped);
        
        frames++;
        double elapsed_time = std::chrono::duration<double, std::micro>(timeNow - start).count();
        // cout << "Time in microseconds: " << elapsed_time << endl;
        if(waitKey(1) != -1){
            break;
        }
    }
    

    cap.release();
    cv::destroyAllWindows();
    return 0;
}