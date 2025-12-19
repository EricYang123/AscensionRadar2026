#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "SORT.h"
#include "hungarian.h"
using namespace std;
using namespace cv;



void SORT::sort(vector<Detection>& detections){
    Hungarian hung;
    Mat distanceMatrix;
    if(predictions.empty()){
        predictions = detections;
        return;
    }
    getDistanceMatrix(detections);
    if(distanceMatrix.empty()){
        return;
    }
    // cout << distanceMatrix << endl;
    vector<Point> starred = hung.hungarian(distanceMatrix);
    predictions = detections;
    // cout << "Starred Spaces:\n" << starred << endl;
    vector<Detection> temp(starred.size());
    cout << "Before temp" << endl;
    cout << "Sizes of vectors " << detections.size() << " " << predictions.size() << " " << starred.size() << endl;
    cout << starred << endl;
    for(int i = 0; i < starred.size(); i++){
        temp.at(starred.at(i).y) = detections.at(starred.at(i).x);
    }
    cout << "After temp" << endl;
}

int SORT::calculateDistance(Rect point1, Rect point2){
    int distance = 0;
    distance = sqrt((point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y));
    return distance;
}

Mat SORT::getDistanceMatrix(vector<Detection> detections){
    Mat distMat(predictions.size(), detections.size(), CV_32SC1);
    if(predictions.size() == 0 || detections.size() == 0){
        return distMat;
    }
    for(int i = 0; i < predictions.size(); i++){
        for(int j = 0; j < detections.size(); j++){
            distMat.at<int>(i, j) = calculateDistance(predictions.at(i).bbox, detections.at(j).bbox);
        }
    }
    return distMat;
}   