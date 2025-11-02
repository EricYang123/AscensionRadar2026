#include "Yolo.h"
#include "layers.h"
#include "Resnet.h"
#include "SORT.h"
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
    Logger logger2;
    Logger logger3;
    Yolo yolo;
    Yolo yolo2;
    Resnet resnet;
    Layers layers;
    SORT sort;

    vector<Detection> d;
    Mat distanceMat(1,1, CV_32S);
    Detection temp;
    temp.bbox.x = 50;
    temp.bbox.y = 50;
    d.push_back(temp);
    sort.getDistanceMatrix(distanceMat, d);
    for(int i = 0; i < distanceMat.rows; i++){
        for(int j = 0; j < distanceMat.cols; j++){
            cout << distanceMat.at<int>(i, j) << " ";
        }
        cout << endl;
    }

    // yolo.init("models/yolo11n.engine", logger);
    // yolo2.init("models/yolov12n.engine", logger2);
    // resnet.init("models/resnet18.engine", logger3);
    
    // VideoCapture cap(0);
    // namedWindow("Webcam", WINDOW_AUTOSIZE);
    // // namedWindow("Cropped Image", WINDOW_AUTOSIZE);

    // Mat frame;
    // Mat cropped;
    // while(true){
    //     cap >> frame;

    //     // yolo.preprocess(frame);

    //     // yolo.infer();

    //     // vector<Detection> detected;
    //     // yolo.postprocess(detected);
    //     // yolo.display(frame, detected);
    //     // layers.showLargestDetection(frame, detected, cropped);

    //     imshow("Webcam", frame);

    //     // yolo2.preprocess(cropped);
    //     resnet.preprocess(frame);

    //     // yolo2.infer();
    //     resnet.infer();

    //     // vector<Detection> detected2;
    //     // yolo2.postprocess(detected2);
    //     resnet.postprocess();
    //     // yolo.display(cropped, detected2);

    //     // imshow("Cropped Image", cropped);
    //     if(waitKey(10) == 'q'){
    //         break;
    //     }
    // }
    

    // cap.release();
    // cv::destroyAllWindows();
    return 0;
}