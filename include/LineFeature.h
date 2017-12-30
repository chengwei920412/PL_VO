//
// Created by rain on 17-12-21.
//

#ifndef PL_VO_LINEFEATURE_H
#define PL_VO_LINEFEATURE_H

#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <line_descriptor_custom.hpp>
#include <line_descriptor/descriptor_custom.hpp>

#include "Config.h"

namespace PL_VO
{

using namespace std;


struct compare_descriptor_by_NN_dist
{
    inline bool operator()(const vector<cv::DMatch>& a, const vector<cv::DMatch>& b){
        return ( a[0].distance < b[0].distance );
    }
};

//
struct compare_descriptor_by_NN12_dist
{
    inline bool operator()(const vector<cv::DMatch>& a, const vector<cv::DMatch>& b){
        return ( a[1].distance - a[0].distance > b[1].distance-b[0].distance );
    }
};

struct sort_descriptor_by_queryIdx
{
    inline bool operator()(const vector<cv::DMatch>& a, const vector<cv::DMatch>& b){
        return ( a[0].queryIdx < b[0].queryIdx );
    }
};

class LineFeature
{

public:

    LineFeature();

    void detectLinefeature(const cv::Mat img, vector<cv::line_descriptor::KeyLine> &vkeylines,
                           cv::Mat &linedesc, const double minLinelength);

    void matchLineFeatures(const cv::Mat &linedesc1, const cv::Mat &linedesc2, vector<cv::DMatch> &vlinematches12);

    vector<cv::DMatch> refineMatchesWithDistance(vector<cv::DMatch> &vlinematches12);

    vector<cv::DMatch> refineMatchesWithKnn(vector<vector<cv::DMatch>> &vlinematches12);

    vector<cv::DMatch> refineMatchesWithFundamental(const vector<cv::line_descriptor::KeyLine> &vqueryKeylines,
                                                    const vector<cv::line_descriptor::KeyLine> &vtrainKeylines,
                                                    const vector<cv::DMatch> &vmathes);
}; // class LineFeature

} // namesapce PL_VO

#endif //PL_VO_LINEFEATURE_H
