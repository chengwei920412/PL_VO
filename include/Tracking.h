//
// Created by rain on 17-12-29.
//

#ifndef PL_VO_TRACKING_H
#define PL_VO_TRACKING_H


#include <string>
#include <iostream>
#include <mutex>
#include <eigen3/Eigen/Dense>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "Camera.h"
#include "Frame.h"
#include "LineFeature.h"
#include "PointFeature.h"

using namespace std;

namespace PL_VO
{

class Camera;
class Frame;
class LineFeature;
class PointFeature;

class Tracking
{

public:

    Tracking(Camera *pCamera);

    ~Tracking();

    void Track(const cv::Mat &imagegray, const cv::Mat &imD, const double &timeStamps);

    bool TrackRefFrame(const vector<cv::Point3d> vpoints3d, const vector<cv::Point2d> vpoints2d, Eigen::Matrix3d K,
                       Eigen::Quaterniond R, Eigen::Vector3d t);

private:

    Camera *mpcamera;
    Frame *mpcurrentFrame;
    Frame *mplastFrame;
    LineFeature *mplineFeature;
    PointFeature *mppointFeature;

    cv::Mat mimgGray;
    cv::Mat mlastimagergb;
    mutex mMutex;

}; // class Tracking

} // namespce PL_VO


#endif //PL_VO_TRACKING_H
