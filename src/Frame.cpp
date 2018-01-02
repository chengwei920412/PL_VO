//
// Created by rain on 17-12-29.
//

#include <future>
#include <opencv2/opencv.hpp>
#include "Frame.h"

namespace PL_VO
{

size_t Frame::gCount = 0;

Frame::Frame(const double &timeStamp, Camera *pCamera, LineFeature *pLineFeature, PointFeature *pPointFeature)
             : mpCamera(pCamera), mpLineFeature(pLineFeature), mpPointFeature(pPointFeature)
{
    mID = gCount;
    gCount++;
    mtimeStamp = timeStamp;
}

Frame::Frame(const Frame &frame)
{
    mID = frame.GetFrameID();
    mtimeStamp = frame.mtimeStamp;
    mpointDesc = frame.mpointDesc;
    mlineDesc = frame.mlineDesc;
    mvKeyPoint.assign(frame.mvKeyPoint.begin(), frame.mvKeyPoint.end());
    mvKeyLine.assign(frame.mvKeyLine.begin(), frame.mvKeyLine.end());

    Tcw = frame.Tcw;
    Twc = frame.Twc;

    mpCamera = frame.mpCamera;
    mpLineFeature = frame.mpLineFeature;
    mpPointFeature = frame.mpPointFeature;
}

size_t Frame::GetFrameID()
{
    return mID;
}

const size_t Frame::GetFrameID() const
{
    return mID;
}

Eigen::Vector3d Frame::getCameraCenter()
{
    return Tcw.inverse().translation();
}

void Frame::detectFeature(const cv::Mat &imagergb, const cv::Mat &imagedepth)
{
    double minLinelength = PL_VO::Config::minLineLength() * min(mpCamera->mImageHeight, mpCamera->mImageWidth);

//    TicToc tictocdetectfeature;

    if (Config::lrInParallel())
    {
        auto linedetect = async(launch::async, &LineFeature::detectLinefeature, mpLineFeature, imagergb,
                                ref(mvKeyLine), ref(mlineDesc), minLinelength);
        auto pointdetect = async(launch::async, &PointFeature::detectPointfeature, mpPointFeature, imagergb,
                                 ref(mvKeyPoint), ref(mpointDesc));

        linedetect.wait();
        pointdetect.wait();
    }
    else
    {
        mpPointFeature->detectPointfeature(imagergb, mvKeyPoint, mpointDesc);
        mpLineFeature->detectLinefeature(imagergb, mvKeyLine, mlineDesc, minLinelength);
    }

//    cout << "the paraller detect the point feature and line feature (ms): " << tictocdetectfeature.toc() << endl;

}

void Frame::matchLPFeature(const cv::Mat &pointdesc1, const cv::Mat &pointdesc2, vector<cv::DMatch> &vpointmatches12,
                           const cv::Mat &linedesc1, const cv::Mat &linedesc2, vector<cv::DMatch> &vlinematches12)
{
    if (Config::lrInParallel())
    {
        auto linematch = async(launch::async, &LineFeature::matchLineFeatures, mpLineFeature, ref(linedesc1),
                               ref(linedesc2), ref(vlinematches12));
        auto pointmatch = async(launch::async, &PointFeature::matchPointFeatures, mpPointFeature, ref(pointdesc1),
                                ref(pointdesc2), ref(vpointmatches12));

        linematch.wait();
        pointmatch.wait();
    }
    else
    {
        mpPointFeature->matchPointFeatures(pointdesc1, pointdesc2, vpointmatches12);
        mpLineFeature->matchLineFeatures(linedesc1, linedesc2, vlinematches12);
    }
}

void Frame::refineLPMatches(const vector<cv::KeyPoint> &mvKeyPoint1, const vector<cv::KeyPoint> &mvKeyPoint2,
                            const vector<cv::line_descriptor::KeyLine> &mvKeyLine1,
                            const vector<cv::line_descriptor::KeyLine> &mvKeyLine2,
                            const vector<cv::DMatch> &vpointMatches12, vector<cv::DMatch> &vpointRefineMatches12,
                            const vector<cv::DMatch> &vlineMatches12, vector<cv::DMatch> &vlineRefineMatches12)
{
    if (Config::lrInParallel())
    {
        auto linematch = async(launch::async, &LineFeature::refineMatchesWithFundamental, mpLineFeature, mvKeyLine1,
                                mvKeyLine2, vlineMatches12);
        auto pointmatch = async(launch::async, &PointFeature::refineMatchesWithFundamental, mpPointFeature, mvKeyPoint1,
                                mvKeyPoint2, vpointMatches12);
        linematch.wait();
        pointmatch.wait();

        vlineRefineMatches12 = linematch.get();
        vpointRefineMatches12 = pointmatch.get();
    }
    else
    {
        vlineRefineMatches12 = mpLineFeature->refineMatchesWithFundamental(mvKeyLine1, mvKeyLine2, vlineMatches12);
        vpointRefineMatches12 = mpPointFeature->refineMatchesWithFundamental(mvKeyPoint1,mvKeyPoint2, vpointMatches12);
    }
}

double Frame::findDepth(const cv::KeyPoint &kp, const cv::Mat &imagedepth)
{
    int x = cvRound(kp.pt.x);
    int y = cvRound(kp.pt.y);

    ushort d = imagedepth.ptr<ushort>(y)[x];

    if (d!=0)
    {
        return double(d)/mpCamera->mdepthscale;
    }
    else
    {
        int dx[4] = {-1,0,1,0};
        int dy[4] = {0,-1,0,1};
        for (int i = 0; i < 4; i++)
        {
            d = imagedepth.ptr<ushort>(y)[x];
            if (d != 0)
            {
                return double(d)/mpCamera->mdepthscale;
            }
        }
    }

    return -1.;
}

} // namespace PL_VO
