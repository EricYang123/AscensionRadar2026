#include "hungarian.h"
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(){
    Hungarian hungarian;
    // Mat matrix = (Mat_<int>(3, 3) << 108,125,150, 150,135,175, 122,148,250);
    // Mat matrix = (Mat_<int>(4, 4) << 82,83,69,92, 77,37,49,92, 11,69,5,86, 8,9,98,23);
    // Mat matrix = (Mat_<int>(3, 3) << 40,60,15, 25,30,45, 55,30,25);
    // Mat matrix = (Mat_<int>(3, 3) << 30,25,10, 15,10,20, 25,20,15);
    // Mat matrix = (Mat_<int>(4, 4) << 20,22,14,24, 20,19,12,20, 13,10,18,16, 22,23,9,28);
    Mat matrix = (Mat_<int>(7, 7) <<
        14,  9, 12,  7, 11, 15, 10,
        8, 13, 10, 12, 14,  9, 11,
        11,  7, 15, 13,  9, 12, 14,
        12, 10,  8, 14, 13, 11,  9,
        9, 12, 11, 10, 15, 13, 14,
        13, 11,  9, 12, 10, 14,  8,
        10, 14, 13,  9, 12,  7, 11
    );

    cout << matrix << endl;
    auto start = std::chrono::high_resolution_clock::now();
    hungarian.hungarian(matrix);
    auto end = std::chrono::high_resolution_clock::now();
    cout << matrix << endl;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " us" << std::endl;
    return 0;
}