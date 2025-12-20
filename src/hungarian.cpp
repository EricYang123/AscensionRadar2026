#include "hungarian.h"
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<Point> Hungarian::hungarian(Mat& matrix){
    
    Mat initialMatrix = matrix.clone();
    coveredCol.assign(matrix.cols, false);
    coveredRow.assign(matrix.rows, false);
    step1(matrix);
    step2(matrix);
    step3(matrix);
    step4(matrix);
    step5(matrix);
    // int cost = 0;
    // for(int i = 0; i < starred.size(); i++){
    //     cost += initialMatrix.at<int>(starred.at(i).y, starred.at(i).x);
    // }
    // cout << "Cost: " << cost << endl;

    return starred;
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
}

void Hungarian::step4(Mat& matrix){
    bool allCovered = false;
    while(!allCovered){
        allCovered = true;
        bool gotoStart = false;
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
                        coveredCol.at(starred.at(starredIdx).x) = false;
                        coveredRow.at(i) = true;
                        gotoStart = true;
                    }else{
                        vector<Point> path;
                        path.push_back(Point(j, i));
                        int latestPrimeCol = j;
                        int latestStarRow;
                        vector<Point> prime2swap;
                        prime2swap.push_back(Point(j, i));
                        vector<Point> star2swap;
                        while(true){
                            //substep 1
                            starredIt = find_if(starred.begin(), starred.end(), [latestPrimeCol](const Point& p){return p.x == latestPrimeCol;});
                            starredIdx = distance(starred.begin(), starredIt);
                            if(starredIdx < starred.size()){
                                latestStarRow = starred.at(starredIdx).y;
                                star2swap.push_back(starred.at(starredIdx));
                            }else{
                                break;
                            }
                            //substep 2
                            auto primedIt = find_if(primed.begin(), primed.end(), [latestStarRow](const Point& p){return p.y == latestStarRow;});
                            int primedIdx = distance(primed.begin(), primedIt);
                            latestPrimeCol = primed.at(primedIdx).x;
                            prime2swap.push_back(primed.at(primedIdx));                   
                        }
                        for(Point point2swap : prime2swap){
                            auto it = find(primed.begin(), primed.end(), point2swap);
                            starred.push_back(*it);
                            primed.erase(it);
                        }
                        for(Point point2swap : star2swap){
                            auto it = find(starred.begin(), starred.end(), point2swap);
                            primed.push_back(*it);
                            starred.erase(it);
                        }
                        fill(coveredCol.begin(), coveredCol.end(), false);
                        fill(coveredRow.begin(), coveredRow.end(), false);
                        primed.clear();
                        for(int c = 0; c < starred.size(); c++){
                            coveredCol.at(starred.at(c).x) = true;
                        }
                        gotoStart = true;
                    }
                }
                if(gotoStart){
                    break;
                }
            }
            if(gotoStart){
                break;
            }
        }
    }
}

void Hungarian::step5(Mat& matrix){
    while(starred.size() < min(matrix.rows, matrix.cols)){
        int minUncovered = INT_MAX;\
        for(int i = 0; i < matrix.rows; i++){
            for(int j = 0; j < matrix.cols; j++){
                if(minUncovered > matrix.at<int>(i, j) && !coveredCol.at(j) && !coveredRow.at(i)){
                    minUncovered = matrix.at<int>(i, j);
                }
            }
        }
        for(int i = 0; i < matrix.rows; i++){
            for(int j = 0; j < matrix.cols; j++){
                if(coveredCol.at(j) && coveredRow.at(i)){
                    matrix.at<int>(i, j) += minUncovered;
                }else if(!coveredCol.at(j) && !coveredRow.at(i)){
                    matrix.at<int>(i, j) -= minUncovered;
                }
            }
        }
        fill(coveredCol.begin(), coveredCol.end(), false);
        fill(coveredRow.begin(), coveredRow.end(), false);
        for(int s = 0; s < starred.size(); s++){
            coveredCol.at(starred.at(s).x) = true;
        }
        step4(matrix);
    }
}