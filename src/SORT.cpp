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
            kals kal;
            kal.object_id = i;
            initKalman(kal.kf, Point(predictions.at(i).bbox.x, predictions.at(i).bbox.y));
            kalmans.push_back(kal);
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
            // predictions.at(i).bbox = detections.at(detectIdx).bbox;
            updatePredictions(detections.at(detectIdx), i);
        }else{
            auto lostIt = find_if(lostIds.begin(), lostIds.end(), [predictObjectId](const lostId& l){return l.object_id == predictObjectId;});
            int lostIdx = distance(lostIds.begin(), lostIt);
            if(lostIdx < lostIds.size()){
                if(lostIds.at(lostIdx).lostFrames >= lostFramesThresh){
                    predictions.erase(predictions.begin() + i);
                    lostIds.erase(lostIds.begin() + lostIdx);
                    removeKalId(predictObjectId);
                }else{
                    lostIds.at(lostIdx).lostFrames++;
                    updatePredictions(i);
                }
            }else{
                lostId lost;
                lost.object_id = predictions.at(i).object_id;
                lost.lostFrames = 1;
                lostIds.push_back(lost);
                updatePredictions(i);
            }
        }
    }
    for(int i = 0; i < detections.size(); i++){
        int detectObjectId = detections.at(i).object_id;
        auto predictIt = find_if(predictions.begin(), predictions.end(), [detectObjectId](const Detection& pre){return pre.object_id == detectObjectId;});
        int predictIdx = distance(predictions.begin(), predictIt);
        if(predictIdx >= predictions.size()){
            predictions.push_back(detections.at(i));
            kals kal;
            kal.object_id = detections.at(i).object_id;
            initKalman(kal.kf, Point(detections.at(i).bbox.x, detections.at(i).bbox.y));
            kalmans.push_back(kal);
        }
    }
}

void SORT::initKalman(KalmanFilter& kf, Point initialPoint){
    kf.init(4, 2, 0);
    kf.transitionMatrix = (Mat_<float>(4,4) << 
                            1, 0, 1, 0,
                            0, 1, 0, 1,
                            0, 0, 1, 0,
                            0, 0, 0, 1);
    kf.measurementMatrix = (Mat_<float>(2, 4) <<
                            1, 0, 0, 0,
                            0, 1, 0, 0);

    setIdentity(kf.processNoiseCov, Scalar::all(1e-5));
    setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(kf.errorCovPost, Scalar::all(1));

    kf.statePost.at<float>(0) = initialPoint.x;
    kf.statePost.at<float>(1) = initialPoint.y;
    kf.statePost.at<float>(2) = 0;
    kf.statePost.at<float>(3) = 0;

    kf.predict();
}

void SORT::updatePredictions(Detection detect, int predictionsIdx){
    int object_id = detect.object_id;
    auto kalIt = find_if(kalmans.begin(), kalmans.end(), [object_id](const kals& kal){return kal.object_id == object_id;});
    int kalIdx = distance(kalmans.begin(), kalIt);
    Mat meas = (Mat_<float>(2, 1) << detect.bbox.x, detect.bbox.y);
    kalmans.at(kalIdx).kf.correct(meas);
    Mat pred = kalmans.at(kalIdx).kf.predict();
    Point nextPred = Point(static_cast<int>(pred.at<float>(0)), static_cast<int>(pred.at<float>(1)));
    predictions.at(predictionsIdx).bbox.x = nextPred.x;
    predictions.at(predictionsIdx).bbox.y = nextPred.y;
}

void SORT::updatePredictions(int predictionsIdx){
    int object_id = predictions.at(predictionsIdx).object_id;
    auto kalIt = find_if(kalmans.begin(), kalmans.end(), [object_id](const kals& kal){return kal.object_id == object_id;});
    int kalIdx = distance(kalmans.begin(), kalIt);
    Mat pred = kalmans.at(kalIdx).kf.predict();
    Point nextPred = Point(static_cast<int>(pred.at<float>(0)), static_cast<int>(pred.at<float>(1)));
    predictions.at(predictionsIdx).bbox.x = nextPred.x;
    predictions.at(predictionsIdx).bbox.y = nextPred.y;
}

void SORT::removeKalId(int object_id){
    auto kalIt = find_if(kalmans.begin(), kalmans.end(), [object_id](const kals& kal){return kal.object_id == object_id;});
    int kalIdx = distance(kalmans.begin(), kalIt);
    kalmans.erase(kalmans.begin() + kalIdx);
}