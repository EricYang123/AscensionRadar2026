#include <opencv2/opencv.hpp>
int main() {
    cv::Mat image = cv::Mat::zeros(300, 300, CV_8UC3);
    cv::imshow("Black Image", image);
    cv::waitKey(0);
    return 0;
}