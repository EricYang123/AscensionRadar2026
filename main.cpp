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
    Yolo yolo;
    SORT sort;

    yolo.init("/root/workspace/yolo11n.engine", logger);
    
    VideoCapture cap(0);

    Mat frame;
    Mat cropped;
    
    while(true){
        cap >> frame;

        yolo.preprocess(frame);

        yolo.infer();

        vector<Detection> detected;
        yolo.postprocess(detected);
        sort.sort(detected);

        yolo.display(frame, detected);
        
        imshow("webcam", frame);

        if(waitKey(1) != -1){
            break;
        }
    }
    

    cap.release();
    cv::destroyAllWindows();
    return 0;
}