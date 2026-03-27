#include "Yolo.h"
#include "layers.h"
#include "Resnet.h"
#include "SORT.h"
#include "laser.h"
#include "stereo.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <memory>
#include <chrono>
#include <cstring>
#include <ratio>
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
    SORT sort2;
    stereoCam stereo;
    // laser serial("/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_3147374A3131-if00", B115200);
    // stereo.calibrate(Size(7, 5), 0.027f, 0.016f, aruco::DICT_6X6_50, "None");
    // yolo.init("../models/yolo11n.engine", logger);
    // resnet.init("models/resnet18.engine", logger3);
    
    VideoCapture capL(0);
    VideoCapture capR(2);

    Mat frameL;
    Mat frameR;
    vector<Point> coords;
    coords.push_back(Point(300, 300));
    // Mat cropped;
    // Mat combinedView;

    capL >> frameL;
    capR >> frameR;
    stereo.initStereo(frameL, frameR);
    // int frames = 0;
    while(true){
	auto start = std::chrono::high_resolution_clock::now();
        capL >> frameL;
	capR >> frameR;
	imshow("left", frameL);
	imshow("right", frameR);

	vector<double> depths;
	depths = stereo.get_depths(coords, frameL, frameR);

	for(int i = 0; i < depths.size(); i++){
	    cout << depths[i] << "\n";
	}
        auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
        // cout << "Time in milliseconds: " << duration.count() << endl;
        if(waitKey(1) != -1){
            break;
        }
    }
    

    capR.release();
    capL.release();
    cv::destroyAllWindows();
    return 0;
}
