//
// Created by rain on 17-12-21.
//

#ifndef PL_VO_LINEFEATURE_H
#define PL_VO_LINEFEATURE_H

#include <vector>
#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <line_descriptor_custom.hpp>
#include <line_descriptor/descriptor_custom.hpp>

#include "Config.h"

namespace PL_VO
{

using namespace std;

class LineFeature
{

public:

    LineFeature();

    void detectLinefeature(const cv::Mat img, vector<cv::line_descriptor::KeyLine> &vkeylines,
                           cv::Mat &linedesc, const double minLinelength);

    void matchLineFeatures(cv::BFMatcher *bfmatcher, cv::Mat linedesc1, cv::Mat linedesc2,
                            vector<vector<cv::DMatch>> &linematches12);


}; // class LineFeature

} // namesapce PL_VO

#endif //PL_VO_LINEFEATURE_H
