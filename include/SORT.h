#include <vector>
#include <opencv2/opencv.hpp>
#include "common.h"
using namespace std;
using namespace cv;

class SORT{
public:
    void sort(vector<Detection>& detections);

    int calculateDistance(Rect point1, Rect point2);
    
    Mat getDistanceMatrix(vector<Detection> detection);

private:
    vector<Rect> predictions;

    Mat distanceMatrix;
};