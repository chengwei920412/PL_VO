//
// Created by rain on 17-12-28.
//

#ifndef PL_VO_POINTFEATURE_H
#define PL_VO_POINTFEATURE_H

#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

namespace PL_VO
{

class PointFeature
{
    void detectPointfeature(const cv::Mat &img, vector<cv::KeyPoint> &vkeypoints, cv::Mat &pointdesc);

}; // class PointFeature

} // namepsace PL_VO

#endif //PL_VO_POINTFEATURE_H
