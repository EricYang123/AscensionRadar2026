#include <vector>
#include <cmath>
#include <algorithm>
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
        for(int i = 0; i < predictions.size(); i++){
            predictions.at(i).object_id = i;
        }
        return;
    }
    distanceMatrix = getDistanceMatrix(detections);
    if(distanceMatrix.empty()){
        reID(detections);
        return;
    }
    vector<Point> starred = hung.hungarian(distanceMatrix);
    assignId(detections, starred);
    reID(detections);
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

void SORT::assignId(vector<Detection>& detections, vector<Point> starred){
    for(int i = 0; i < starred.size(); i++){
        detections.at(starred.at(i).x).object_id = predictions.at(starred.at(i).y).object_id;
    }
    for(int i = 0; i < detections.size(); i++){
        if(detections.at(i).object_id == -1){
            for(int j = 0; j < detections.size() + predictions.size(); j++){
                bool idExistsInDet = any_of(detections.begin(), detections.end(), [j](const Detection& detect){ return detect.object_id == j;});
                bool idExistsInPre = any_of(predictions.begin(), predictions.end(), [j](const Detection& predict){ return predict.object_id == j;});
                if(!idExistsInDet && !idExistsInPre){
                    detections.at(i).object_id = j;
                    break;
                }
            }
        }

    }
}

void SORT::reID(vector<Detection> detections){
    for(int i = 0; i < predictions.size(); i++){
        int predictObjectId = predictions.at(i).object_id;
        auto detectIt = find_if(detections.begin(), detections.end(), [predictObjectId](const Detection& det){return det.object_id == predictObjectId;});
        int detectIdx = distance(detections.begin(), detectIt);
        if(detectIdx < detections.size()){
            predictions.at(i).bbox = detections.at(detectIdx).bbox;
        }else{
            auto lostIt = find_if(lostIds.begin(), lostIds.end(), [predictObjectId](const lostId& l){return l.object_id == predictObjectId;});
            int lostIdx = distance(lostIds.begin(), lostIt);
            if(lostIdx < lostIds.size()){
                if(lostIds.at(lostIdx).lostFrames >= lostFramesThresh){
                    predictions.erase(predictions.begin() + i);
                    lostIds.erase(lostIds.begin() + lostIdx);
                }else{
                    lostIds.at(lostIdx).lostFrames++;

                }
            }else{
                lostId lost;
                lost.object_id = predictions.at(i).object_id;
                lost.lostFrames = 1;
                lostIds.push_back(lost);
            }
        }
    }
    for(int i = 0; i < detections.size(); i++){
        int detectObjectId = detections.at(i).object_id;
        auto predictIt = find_if(predictions.begin(), predictions.end(), [detectObjectId](const Detection& pre){return pre.object_id == detectObjectId;});
        int predictIdx = distance(predictions.begin(), predictIt);
        if(predictIdx >= predictions.size()){
            predictions.push_back(detections.at(i));
        }
    }
}