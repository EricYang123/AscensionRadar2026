#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "SORT.h"
using namespace std;
using namespace cv;

int SORT::calculateDistance(Rect point1, Rect point2){
    int distance = 0;
    distance = sqrt((point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y));
    return distance;
}

void SORT::getDistanceMatrix(Mat& distances, vector<Detection> detection){
    Rect temp1;
    temp1.x = 10;
    temp1.y = 10;
    predictions.push_back(temp1);
    for(int i = 0; i < predictions.size(); i++){
        for(int j = 0; j < detection.size(); j++){
            distances.at<int>(i, j) = calculateDistance(predictions.at(i), detection.at(j).bbox);
        }
    }
}