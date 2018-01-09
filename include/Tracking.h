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
#include "Converter.h"

using namespace std;

namespace PL_VO
{

class Camera;
class Frame;
class LineFeature;
class PointFeature;
class Map;

class Tracking
{

public:

    Tracking(Camera *pCamera);

    ~Tracking();

    void SetMap(Map *pMap);

    void Track(const cv::Mat &imagegray, const cv::Mat &imD, const double &timeStamps);

    bool TrackRefFrame(vector<cv::DMatch> &vpointMatches);

    void UpdateMapLPfeature(const vector<cv::DMatch> &vpointMatches, const vector<cv::DMatch> &vlineMatches);

private:

    void UpdateMapPointfeature(const vector<cv::DMatch> &vpointMatches);

    void UpdateMapLinefeature(const vector<cv::DMatch> &vlineMatches);

    Camera *mpcamera = nullptr;
    Frame *mpcurrentFrame = nullptr;
    Frame *mplastFrame = nullptr;
    LineFeature *mpLineFeature = nullptr;
    PointFeature *mpPointFeature = nullptr;
    Map *mpMap = nullptr;

    cv::Mat mimageGray;
    cv::Mat mimagergb;
    cv::Mat mlastimageGrays;
    cv::Mat mlastimagergb;
    cv::Mat mimageDepth;
    cv::Mat mlastimageDepth;

    size_t countMapPoint;
    size_t countMapLine;

    mutex mMutex;

}; // class Tracking

} // namespce PL_VO


#endif //PL_VO_TRACKING_H
