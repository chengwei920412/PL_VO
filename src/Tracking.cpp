//
// Created by rain on 17-12-29.
//

#include "Tracking.h"
#include "Converter.h"

namespace PL_VO
{

Tracking::Tracking(Camera *pCamera) : mpcamera(pCamera)
{
    mplineFeature = new(LineFeature);
    mppointFeature = new(PointFeature);
}

Tracking::~Tracking()
{
    delete(mplineFeature);
    delete(mppointFeature);
}

void Tracking::SetMap(Map *pMap)
{
    mpMap = pMap;
}

void Tracking::Track(const cv::Mat &imagergb, const cv::Mat &imD, const double &timeStamps)
{
    mimageGray = imagergb.clone();
    mimagergb = imagergb.clone();
    mimageDepth = imD;

    bool mbRGB = Config::imageRGBForm();
    if(mimageGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mimageGray,mimageGray,CV_RGB2GRAY);
        else
            cvtColor(mimageGray,mimageGray,CV_BGR2GRAY);
    }
    else if(mimageGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mimageGray,mimageGray,CV_RGBA2GRAY);
        else
            cvtColor(mimageGray,mimageGray,CV_BGRA2GRAY);
    }

    mpcurrentFrame = new Frame(timeStamps, mpcamera, mplineFeature, mppointFeature);

    if (mpcurrentFrame->GetFrameID() == 0)
    {
        mpcurrentFrame->Tcw.so3().setQuaternion(Eigen::Quaterniond::Identity());
        mpcurrentFrame->Tcw.translation() = Eigen::Vector3d(0, 0, 0);
    }

    mpcurrentFrame->detectFeature(mimageGray, mimageDepth);

    mpcurrentFrame->UndistortKeyFeature();

    if (!mlastimageGrays.empty())
    {
        vector<cv::DMatch> vpointMatches;
        vector<cv::DMatch> vpointRefineMatches;
        vector<cv::DMatch> vlineMatches;
        vector<cv::DMatch> vlineRefineMatches;

        mpcurrentFrame->matchLPFeature(mplastFrame->mpointDesc, mpcurrentFrame->mpointDesc, vpointMatches,
                                       mplastFrame->mlineDesc, mpcurrentFrame->mlineDesc, vlineMatches);

        mpcurrentFrame->refineLPMatches(mplastFrame->mvKeyPoint, mpcurrentFrame->mvKeyPoint,
                                        mplastFrame->mvKeyLine, mpcurrentFrame->mvKeyLine,
                                        vpointMatches, vpointRefineMatches, vlineMatches, vlineRefineMatches);

        // use the pnp and point match to track the reference frame
        TrackRefFrame(vpointRefineMatches);

        cout << mpcurrentFrame->Tcw << endl;

        cv::Mat showimg;
        cv::drawMatches(mlastimageGrays, mplastFrame->mvKeyPoint, mimageGray, mpcurrentFrame->mvKeyPoint,
                        vpointRefineMatches, showimg);

//        std::vector<char> mask(vlineRefineMatches.size(), 1);
//        cv::line_descriptor::drawLineMatches(mlastimagergb, mplastFrame->mvKeyLine, mimagergb, mpcurrentFrame->mvKeyLine,
//                                             vlineRefineMatches, showimg,  cv::Scalar::all(-1), cv::Scalar::all(-1), mask,
//                                             cv::line_descriptor::DrawLinesMatchesFlags::DEFAULT);
        cv::imshow(" ", showimg);
        cv::waitKey(5);
    }

    mpMap->mlpFrames.push_back(mpcurrentFrame);

    mplastFrame = new Frame(*mpcurrentFrame);
    mlastimageGrays = mimageGray.clone();
    mlastimagergb = mimagergb.clone();
    mlastimageDepth = mimageDepth.clone();
}

bool Tracking::TrackRefFrame(const vector<cv::DMatch> vpointMatches)
{
    vector<cv::Point3d> vpts3d;
    vector<cv::Point2d> vpts2d;

    for (auto match:vpointMatches)
    {
        double d = mplastFrame->FindDepth(mplastFrame->mvKeyPoint[match.queryIdx].pt, mlastimageDepth);
        if (d < 0)
            continue;

        Eigen::Vector3d Point3dw;
        Point3dw = mpcamera->Pixwl2World(Converter::toVector2d(mplastFrame->mvKeyPoint[match.queryIdx].pt),
                                         mplastFrame->Tcw.so3().unit_quaternion(), mplastFrame->Tcw.translation(), d);

        vpts3d.push_back(Converter::toCvPoint3f(Point3dw));
        vpts2d.push_back(mpcurrentFrame->mvKeyPoint[match.trainIdx].pt);

    }

    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac(vpts3d, vpts2d, Converter::toCvMat(mpcamera->GetCameraIntrinsic()), cv::Mat(),
                       rvec, tvec, false, 100, 4.0, 0.99, inliers);


    mpcurrentFrame->Tcw = Sophus::SE3(Sophus::SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)),
                                      Converter::toVector3d(tvec));
}



} // namespace PL_VO
