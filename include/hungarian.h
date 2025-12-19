#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Hungarian{
    public:
        vector<Point> hungarian(Mat& matrix);
        void step1(Mat& matrix);
        void step2(Mat& matrix);
        void step3(Mat& matrix);
        void step4(Mat& matrix);
        void step5(Mat& matrix);
    
    private:
        vector<Point> starred;
        vector<Point> primed;
        vector<bool> coveredCol;
        vector<bool> coveredRow;
};