#include <opencv2/imgproc.hpp>

class MathImg
{
private:
    double standard_deviation = 0.f;
    double mean = 0.f;
    cv::Mat img;
    void calcStandardDeviation();
    void calcMean();
public:
    MathImg(const cv::Mat& img);
    double getStandardDeviation();
    double getMean();
    const cv::Mat& getImg();
};
