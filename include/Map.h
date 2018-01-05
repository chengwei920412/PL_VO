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
class MapPoint;
class MapLine;
struct PointFeature2D;
struct LineFeature2D;

struct PointFeature2D
{
    PointFeature2D(const Eigen::Vector2d &pixel, const int &level=0, const double &score=0)
                   : mpixel(pixel), mlevel(level), mscore(score) {}

    Eigen::Vector2d mpixel;
    double mdepth = -1;
    int mlevel = -1;
    double mangle = 0;
    cv::Mat desc = cv::Mat(1, 32, CV_8UC1);
    Frame *mpFrame = nullptr;
    MapPoint *mpMapPoint = nullptr;
    bool mbbad = false;
    double mscore = 0;
};

struct LineFeature2D
{
    LineFeature2D(const Eigen::Vector2d &Startpixel, const Eigen::Vector2d &Endpixel, const int &level=0,
                  const double &score=0) : mStartpixel(Startpixel), mEndpixel(Endpixel), mlevel(level), mscore(score){}

    Eigen::Vector2d mStartpixel = Eigen::Vector2d(0, 0);
    Eigen::Vector2d mEndpixel = Eigen::Vector2d(0, 0);
    double mStartdepth = -1;
    double mEnddepth = -1;
    int mlevel = -1;
    double mangle = 0;
    cv::Mat desc = cv::Mat(1, 32, CV_8UC1);
    Frame *mpFrame = nullptr;
    MapLine *pMapLine = nullptr;
    bool mbbad = false;
    double mscore = 0;

};

class MapPoint
{
public:

    MapPoint();

    size_t  mID;
    Eigen::Vector3d mPosew;
    cv::Mat mdesc;
    bool mbbad = false;
    list<Frame*> mlpFrameinvert;
    map<size_t, PointFeature2D*> mmpPointFeature2D;
};

class MapLine
{
public:

    MapLine();

    size_t mID;
    Eigen::Vector3d mPoseStartw;
    Eigen::Vector3d mPoseEndw;
    bool mbbad = false;
    cv::Mat mdesc;
    list<Frame*> mlpFrameinvert;
    map<size_t, LineFeature2D*> mmpLineFeature2D;
};

class Map
{
public:

    Map();

    list<Frame*> mlpFrames;
    vector<MapLine*> mvpMapLine;
    vector<MapPoint*> mvpMapPoint;
}; // class Map

} // namespace PL_VO

#endif //PL_VO_MAP_H
