#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class stereoCam{
public:
    stereoCam(); 
    void calibrate(cv::Size inputBoardSize, float squareSize, float markerSize, cv::aruco::PredefinedDictionaryType arucoDict, 
		   std::string arucoDictFile, bool displayCorners = false, bool useCalibrated=true, bool showRectified=true);

private:
    void initStereo();
    vector<Mat> take_images();
};
