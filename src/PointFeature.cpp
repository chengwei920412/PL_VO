//
// Created by rain on 17-12-28.
//

#include "PointFeature.h"

namespace PL_VO
{

void PointFeature::detectPointfeature(const cv::Mat &img, vector<cv::KeyPoint> &vkeypoints, cv::Mat &pointdesc)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ("BruteForce-Hamming");

    detector->detect(img, vkeypoints);
    descriptor->compute(img, vkeypoints, pointdesc);
}

} // namespace PL_VO