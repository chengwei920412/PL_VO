//
// Created by rain on 17-12-20.
//

#include <iostream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

int main(int argc, char** argv)
{
    std::string in;

    cv::Mat image = cv::imread("../test/test_line_feature/desk.png", cv::IMREAD_GRAYSCALE);

#if 0
    cv::Canny(image, image, 50, 200, 3); // Apply canny edge
#endif

    // Create and LSD detector with standard or no refinement.
#if 1
    cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
#else
    cv::Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_NONE);
#endif
    double start = double(cv::getTickCount());
    vector<cv::Vec4f> lines_std;
    // Detect the lines
    ls->detect(image, lines_std);
    double duration_ms = (double(cv::getTickCount()) - start) * 1000 / cv::getTickFrequency();
    std::cout << "It took " << duration_ms << " ms." << std::endl;
    // Show found lines
    cv::Mat drawnLines(image);
    ls->drawSegments(drawnLines, lines_std);
    cv::imshow("Standard refinement", drawnLines);
    cv::waitKey();
    return 0;
}