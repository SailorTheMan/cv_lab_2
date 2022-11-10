#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "MathImg/MathImg.h"
#include <iostream>

using namespace cv;

double calcCrossCorr(MathImg& f, MathImg& t)
{
    double cross_corr = 0.f;
    const Mat& fImg = f.getImg();
    const Mat& tImg = t.getImg();
    for (int x = 0; x < fImg.cols; x++)
    {
        for (int y = 0; y < fImg.rows; y++)
        {
            double fMinusFMean = fImg.at<uchar>(y, x) - f.getMean();
            double tMinusTMean = tImg.at<uchar>(y, x) - t.getMean();
            cross_corr += (fMinusFMean * tMinusTMean)
                / (f.getStandardDeviation() * t.getStandardDeviation());
        }
    }
    return cross_corr / (fImg.cols * fImg.rows);
}

Rect2i findPattern(const Mat& pattern, const Mat& src)
{
    Rect2i result;
    result.width = pattern.cols;
    result.height = pattern.rows;
    double max_cross_corr = 0.f;
    MathImg pattern_mi(pattern);
    int counter = 0;
    for (int x = 0; x < src.cols - pattern.cols; x++)
    {
        for (int y = 0; y < src.rows - pattern.rows; y++)
        {
            Rect roi_rect(x, y, pattern.cols, pattern.rows);
            Mat roi = src(roi_rect);
            MathImg roi_mi(roi);
            double cross_corr = calcCrossCorr(roi_mi, pattern_mi);
            if (cross_corr > max_cross_corr)
            {
                max_cross_corr = cross_corr;
                result.x = x;
                result.y = y;
            }
        }
    }
    return result;

}

int main(int argc, char const *argv[])
{
    // Mat srcImg = imread("../samples/lena/lena.png");
    // Mat pattern = imread("../samples/lena/lena_pattern.png");

    const Mat srcImg = imread("../samples/box/test1.jpg", 0);
    const Mat pattern = imread("../samples/box/pattern.jpg", 0);
    Rect patternRect = findPattern(pattern, srcImg);
    rectangle(srcImg, patternRect, Scalar(255, 0, 0), 3);
    imshow("src", srcImg);
    imshow("pattern", pattern);
    waitKey(0);
    return 0;
}
