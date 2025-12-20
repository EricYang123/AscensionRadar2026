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

    void assignId(vector<Detection>& detections, vector<Point> starred);

    void reID(vector<Detection> detections);

private:

    struct lostId{
            int object_id = -1;
            int lostFrames = 0;
    };
    vector<Detection> predictions;

    int lostFramesThresh = 5;

    vector<lostId> lostIds;

    
};