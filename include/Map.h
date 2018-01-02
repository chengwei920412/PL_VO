//
// Created by rain on 18-1-2.
//

#ifndef PL_VO_MAP_H
#define PL_VO_MAP_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "Frame.h"

namespace PL_VO
{

class Frame;

class MapPoint
{
public:
    size_t  mid;
    Eigen::Vector3d mPosew;
    cv::Mat mdesc;

    list<Frame*> mlpFrameinvert;

};

class MapLine
{
public:
    size_t mid;
    Eigen::Vector3d mPoseStartw;
    Eigen::Vector3d mPoseEndw;
    cv::Mat mdesc;

    list<Frame*> mlpFrameinvert;

};

class Map
{
public:
    Map();

    list<Frame*> mlFrames;

}; // class Map

} // namespace PL_VO

#endif //PL_VO_MAP_H
