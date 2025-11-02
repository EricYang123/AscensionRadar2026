#include <vector>
#include <opencv2/opencv.hpp>
#include "common.h"
using namespace std;
using namespace cv;

class SORT{
public:
    int calculateDistance(Rect point1, Rect point2);
    
    void getDistanceMatrix(Mat& distances, vector<Detection> detection);

private:
    vector<Rect> predictions;
};