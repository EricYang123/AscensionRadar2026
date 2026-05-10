#include "Yolo.h"
#include "layers.h"
#include "Resnet.h"
#include "SORT.h"
#include "laser.h"
#include "stereo.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <opencv2/highgui.hpp>
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
    // stereoCam stereo;
    laser serial("/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_3147374A3131-if00", B115200);
    // stereo.calibrate(Size(7, 5), 0.027f, 0.016f, aruco::DICT_6X6_50, "None");
    yolo.init("../yolo11n.engine", logger);
    // resnet.init("models/resnet18.engine", logger3);
    
    VideoCapture capL(0);
    // VideoCapture capR(2);

    Mat frameL;
    // Mat frameR;
    // vector<Point> coords;
    // Mat cropped;
    // Mat combinedView;

    // capR >> frameR;
    // stereo.initStereo(frameL, frameR);
    // int frames = 0;
    serial.setServoAngle(135, 135);
    int counter = 0;
    while(true){
	capL >> frameL;
	yolo.preprocess(frameL);
	yolo.infer();
	vector<Detection> detections;
	yolo.postprocess(detections);
	counter++;
	for(int i = 0; i < detections.size(); i++){
	    if(detections[i].class_id == 39){
		int xcoord = detections[i].bbox.x + detections[i].bbox.width / 2;
		int ycoord = detections[i].bbox.y + detections[i].bbox.height / 2;
		cout << "Detections: " << xcoord << "," << ycoord << "\n";
		if(counter > 5){
		    serial.aimLaser(xcoord, ycoord);
		    counter = 0;
		}
		break;
	    }
	}
	yolo.display(frameL, detections);
	imshow("Window", frameL);

        if(waitKey(1) != -1){
            break;
        }
    }
    
 //    while(true){
	// int x, y;
	// cin >> x >> y;
	// serial.setServoAngle(x, y);
    // }

    // capR.release();
    capL.release();
    cv::destroyAllWindows();
    return 0;
}
