#include "MathImg.h"

MathImg::MathImg(const cv::Mat& img)
{
    this->img = img.clone();
    this->calcMean();
    this->calcStandardDeviation();
}

void MathImg::calcStandardDeviation()
{
    double variance = 0.f;
    for (int x = 0; x < img.cols; x++)
    {
        for (int y = 0; y < img.rows; y++)
        {
            variance += abs(img.at<uchar>(y, x) - mean);
        }
    }
    variance /= img.cols * img.rows;
    standard_deviation = sqrt(variance);
}

void MathImg::calcMean()
{
    double sum = 0.f;
    for (int x = 0; x < img.cols; x++)
    {
        for (int y = 0; y < img.rows; y++)
        {
            sum += img.at<uchar>(y, x);
        }
    }
    mean = sum / (img.cols * img.rows);
}

double MathImg::getStandardDeviation() {
    return standard_deviation;
}

double MathImg::getMean() {
    return mean;
}

const cv::Mat& MathImg::getImg()
{
    return img;
}

