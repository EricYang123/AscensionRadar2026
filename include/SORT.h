#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
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

    void initKalman(KalmanFilter& kf, Point initialPoint);

    void updatePredictions(Detection detect, int predictionsIdx);

    void updatePredictions(int predictionsIdx);

    void removeKalId(int object_id);

private:

    struct lostId{
        int object_id = -1;
        int lostFrames = 0;
    };

    struct kals{
        KalmanFilter kf;
        int object_id = -1;
    };

    vector<kals> kalmans;

    vector<Detection> predictions;

    int lostFramesThresh = 10;

    vector<lostId> lostIds;

    
};