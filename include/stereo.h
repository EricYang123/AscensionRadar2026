#include <opencv2/calib3d.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class stereoCam{
public:
    stereoCam(); 
    void calibrate(cv::Size inputBoardSize, float squareSize, float markerSize, cv::aruco::PredefinedDictionaryType arucoDict, 
		   std::string arucoDictFile, bool displayCorners = false, bool useCalibrated=true, bool showRectified=true);
    vector<double> get_depths(vector<Point> coordinates, Mat imageL, Mat imageR);
    void initStereo(Mat imageL, Mat imageR);

private:
    vector<Mat> take_images();
    Mat getDisparity(Mat imageL, Mat imageR);


    Ptr<StereoSGBM> sgbm;
    bool display = false;

    // Calibration Variables
    Mat M1, D1, M2, D2;
    Mat R, T, R1, P1, R2, P2;
    Mat Q;
    Rect roi1, roi2;

    // Rectify Images variables
    Mat map11, map12, map21, map22;
    Mat img1r, img2r;
    Size img_size;

    // Disparity variables
    int numberOfDisparities;
    int sgbmWinSize;
    int cn;

    // Depth Variables
    float d;
    double f, B;
};
