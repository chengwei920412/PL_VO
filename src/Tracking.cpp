//
// Created by rain on 17-12-29.
//

#include "Tracking.h"

namespace PL_VO
{

Tracking::Tracking(Camera *pCamera) : mpcamera(pCamera)
{
    mplineFeature = new(LineFeature);
    mppointFeature = new(PointFeature);
}

void Tracking::Track(const cv::Mat &imagergb, const cv::Mat &imD, const double &timeStamps)
{
    mimgGray = imagergb.clone();
    cv::Mat imDepth = imD;

//    bool mbRGB = true;
//    if(mimgGray.channels()==3)
//    {
//        if(mbRGB)
//            cvtColor(mimgGray,mimgGray,CV_RGB2GRAY);
//        else
//            cvtColor(mimgGray,mimgGray,CV_BGR2GRAY);
//    }
//    else if(mimgGray.channels()==4)
//    {
//        if(mbRGB)
//            cvtColor(mimgGray,mimgGray,CV_RGBA2GRAY);
//        else
//            cvtColor(mimgGray,mimgGray,CV_BGRA2GRAY);
//    }

    mpcurrentFrame = new Frame(timeStamps, mpcamera, mplineFeature, mppointFeature);

    mpcurrentFrame->detectFeature(imagergb, imDepth);

    if (!mlastimagergb.empty())
    {
        vector<cv::DMatch> vpointMatches;
        vector<cv::DMatch> vpointRefineMatches;
        vector<cv::DMatch> vlineMatches;
        vector<cv::DMatch> vlineRefineMatches;

        mpcurrentFrame->matchLPFeature(mplastFrame->mpointDesc, mpcurrentFrame->mpointDesc, vpointMatches,
                                       mplastFrame->mlineDesc, mpcurrentFrame->mlineDesc, vlineMatches);

        mpcurrentFrame->refineMatches(mplastFrame->mvKeyPoint, mpcurrentFrame->mvKeyPoint,
                                      mplastFrame->mvKeyLine, mpcurrentFrame->mvKeyLine,
                                      vpointMatches, vpointRefineMatches, vlineMatches, vlineRefineMatches);

        cv::Mat showimg;
        cv::drawMatches(mlastimagergb, mplastFrame->mvKeyPoint, mimgGray, mpcurrentFrame->mvKeyPoint,
                        vpointRefineMatches, showimg);

        std::vector<char> mask(vlineRefineMatches.size(), 1);
        cv::line_descriptor::drawLineMatches(mlastimagergb, mplastFrame->mvKeyLine, mimgGray, mpcurrentFrame->mvKeyLine,
                                             vlineRefineMatches, showimg,  cv::Scalar::all(-1), cv::Scalar::all(-1), mask,
                                             cv::line_descriptor::DrawLinesMatchesFlags::DEFAULT);
        cv::imshow(" ", showimg);
        cv::waitKey(5);
    }

    mplastFrame = new Frame(*mpcurrentFrame);
    mlastimagergb = imagergb.clone();
}

} // namespace PL_VO
