//
// Created by rain on 17-12-28.
//

#ifndef PL_VO_POINTFEATURE_H
#define PL_VO_POINTFEATURE_H

#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <glog/logging.h>
#include "Config.h"


using namespace std;

namespace PL_VO
{

class PointFeature
{

public:
    void detectPointfeature(const cv::Mat &img, vector<cv::KeyPoint> &vkeypoints, cv::Mat &pointdesc);

    void matchPointFeatures(const cv::Mat &pointdesc1, const cv::Mat &pointdesc2, vector<cv::DMatch> &vpointmatches12);

    vector<cv::DMatch> refineMatchesWithDistance(const vector<cv::DMatch> &vmatches);

    vector<cv::DMatch> refineMatchesWithFundamental(const vector<cv::KeyPoint>& vKeyPoints1, const vector<cv::KeyPoint>& vKeyPoints2,
                                                    const vector<cv::DMatch> &vmathes);

}; // class PointFeature

} // namepsace PL_VO

#endif //PL_VO_POINTFEATURE_H
