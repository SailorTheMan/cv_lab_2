#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;

Rect2i getPatternRect(std::vector<KeyPoint>& kps1, std::vector<KeyPoint>& kps2, std::vector<DMatch>& matches, Size2i patternSize)
{
    KeyPoint point1 = kps1[matches[0].queryIdx];
    KeyPoint point2 = kps2[matches[0].trainIdx];
    return Rect2i (point1.pt.x - point2.pt.x, point1.pt.y - point2.pt.y, patternSize.width, patternSize.height);
}

int main(int argc, char const *argv[])
{
    const Mat input = imread("../samples/lena.png", 0);
    const Mat pattern = imread("../samples/lena_pattern_3.png", 0);

    Ptr<SIFT> siftPtr = SIFT::create();
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat des1, des2;
    siftPtr->detectAndCompute(input, Mat(),  keypoints1, des1);
    siftPtr->detectAndCompute(pattern, Mat(),  keypoints2, des2);

    Ptr<BFMatcher> bfPtr = BFMatcher::create();
    std::vector< std::vector<DMatch> > knn_matches;
    bfPtr->knnMatch(des1, des2, knn_matches, 2);

    const float ratio_thresh = 0.1f;
    std::vector<DMatch> good_matches;

    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    Rect2i patternRect = getPatternRect(keypoints1, keypoints2, good_matches, Size(pattern.cols, pattern.rows ));
    Mat output = input.clone();
    rectangle(output, patternRect, Scalar(255, 0, 0));

    Mat img_matches;
    drawMatches( input, keypoints1, pattern, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow("Good Matches", img_matches );
    imshow("result", output );
    waitKey(0);
    return 0;
}

