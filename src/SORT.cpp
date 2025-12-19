#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "SORT.h"
#include "hungarian.h"
using namespace std;
using namespace cv;



void SORT::sort(vector<Detection>& detections){
    cout << "Detections Before:\n";
    for(int i = 0; i < detections.size(); i++){
        cout << detections.at(i).bbox << endl;
    }
    Hungarian hung;
    Mat distanceMatrix;
    if(predictions.empty()){
        cout << "Predictions Empty" << endl;
        predictions = detections;
        return;
    }
    distanceMatrix = getDistanceMatrix(detections);
    // cout << "Distance Matrix: \n" << distanceMatrix << endl;
    if(distanceMatrix.empty()){
        return;
    }
    vector<Point> starred = hung.hungarian(distanceMatrix);
    // cout << "Detections Size: " << detections.size() << endl;
    cout << "Starred Values: \n" << starred << endl;
    
    vector<Detection> temp(detections.size());
    for(int i = 0; i < starred.size(); i++){
        if(starred.at(i).y >= detections.size() || starred.at(i).x >= detections.size()){
            continue;
        }
        temp.at(starred.at(i).y) = detections.at(starred.at(i).x);
    }
    for(int i = 0; i < detections.size(); i++){
        bool indexAdded = any_of(starred.begin(), starred.end(), [i](const cv::Point& point){ return point.x == i;});
        if(!indexAdded){
            temp.push_back(detections.at(i));
        }
    }

    detections = temp;
    cout << "Detections After:\n";
    for(int i = 0; i < detections.size(); i++){
        cout << detections.at(i).bbox << endl;
    }
    predictions = detections;
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