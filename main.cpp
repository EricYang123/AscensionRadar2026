#include "hungarian.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    Hungarian hungarian;
    // Mat matrix = (Mat_<int>(3, 3) << 108,125,150, 150,135,175, 122,148,250);
    Mat matrix = (Mat_<int>(4, 4) << 82,83,69,92, 77,37,49,92, 11,69,5,86, 8,9,98,23);
    cout << matrix << endl;
    hungarian.hungarian(matrix);
    cout << matrix << endl;
    return 0;
}