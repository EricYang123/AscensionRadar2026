#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "SORT.h"
using namespace std;
using namespace cv;

void SORT::sort(vector<Detection>& detections){
    distanceMatrix = getDistanceMatrix(detections);
}

int SORT::calculateDistance(Rect point1, Rect point2){
    int distance = 0;
    distance = sqrt((point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y));
    return distance;
}

Mat SORT::getDistanceMatrix(vector<Detection> detections){
    Mat distMat(predictions.size(), detections.size(), CV_32SC1);
    for(int i = 0; i < predictions.size(); i++){
        for(int j = 0; j < detections.size(); j++){
            distanceMatrix.at<int>(i, j) = calculateDistance(predictions.at(i), detections.at(j).bbox);
        }
    }
    return distMat;
}