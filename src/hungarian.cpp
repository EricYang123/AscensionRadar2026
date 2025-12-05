#include "hungarian.h"
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void Hungarian::hungarian(Mat& matrix){
    Mat initialMatrix = matrix.clone();
    coveredCol.assign(matrix.cols, false);
    coveredRow.assign(matrix.rows, false);
    step1(matrix);
    step2(matrix);
    step3(matrix);
}

void Hungarian::step1(Mat& matrix){
    for(int i = 0; i < matrix.rows; i++){
        int smallest = matrix.at<int>(i, 0);
        for(int j = 0; j < matrix.cols; j++){
            if(matrix.at<int>(i, j) < smallest){
                smallest = matrix.at<int>(i, j);
            }
        }
        for(int j = 0; j < matrix.cols; j++){
            matrix.at<int>(i, j) -= smallest;
        }
    }
}

void Hungarian::step2(Mat& matrix){
    for(int i = 0; i < matrix.cols; i++){
        int smallest = matrix.at<int>(0, i);
        for(int j = 0; j < matrix.rows; j++){
            if(matrix.at<int>(j, i) < smallest){
                smallest = matrix.at<int>(j, i);
            }
        }
        for(int j = 0; j < matrix.rows; j++){
            matrix.at<int>(j, i) -= smallest;
        }
    }
}

void Hungarian::step3(Mat& matrix){
    for(int i = 0; i < matrix.rows; i++){
        for(int j = 0; j < matrix.cols; j++){
            if (matrix.at<int>(i, j) == 0 && !coveredCol.at(j)){
                starred.push_back(Point(j, i));
                coveredCol.at(j) = true;
                break;
            }
        }
    }
    cout << starred << endl;
}

void Hungarian::step4(Mat& matrix){
    for(int c = 0; c < starred.size(); c++){
        coveredCol.at(starred.at(c).x) = true;
    }
    bool allCovered = false;
    while(!allCovered){
        allCovered = true;
        for(int j = 0; j < matrix.cols; j++){
            if(coveredCol.at(j)){
                continue;
            }
            for(int i = 0; i < matrix.rows; i++){
                if(matrix.at<int>(i, j) == 0 && !coveredRow.at(i)){
                    allCovered = false;
                    primed.push_back(Point(j, i));
                    int starredIdx;
                    auto starredIt = find_if(starred.begin(), starred.end(), [i](const Point& p){return p.y == i;});
                    if(starredIt != starred.end()){
                        starredIdx = distance(starred.begin(), starredIt);
                    }else{
                        starredIdx = -1;
                    }

                    if(starredIdx != -1){

                    }else{




                        fill(coveredCol.begin(), coveredCol.end(), false);
                        fill(coveredRow.begin(), coveredRow.end(), false);
                        for(int c = 0; c < starred.size(); c++){
                            coveredCol.at(starred.at(c).x) = true;
                        }
                    }
                }
            }
        }
    }

}

void Hungarian::step5(Mat& matrix){
    
}