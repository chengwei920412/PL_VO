//
// Created by rain on 17-12-29.
//

#include "System.h"

namespace PL_VO
{

System::System(const string &strSettingsFile)
{
    mpCamera = new Camera(strSettingsFile);
    mpTracking = new Tracking(mpCamera);
}

System::~System()
{
    delete mpCamera;
}

Eigen::Matrix<double, 7, 1>  System::TrackRGBD(const cv::Mat &imagergb, const cv::Mat &imagedepth, const double &timeStamps)
{
    mpTracking->Track(imagergb, imagedepth, timeStamps);
}

} // namespace PL_VO