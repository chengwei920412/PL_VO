//
// Created by rain on 17-12-29.
//

#include "Tracking.h"

namespace PL_VO
{

Tracking::Tracking(Camera *pCamera) : mpcamera(pCamera)
{
    mpLineFeature = new(LineFeature);
    mpPointFeature = new(PointFeature);

    countMapPoint = 0;
    countMapLine = 0;
}

Tracking::~Tracking()
{
    delete(mpLineFeature);
    delete(mpPointFeature);
}

void Tracking::SetMap(Map *pMap)
{
    mpMap = pMap;
}

// just use for the test.
void Tracking::SetCurLastFrame(Frame *pcurFrame, Frame *plastFrame)
{
    mpcurrentFrame = pcurFrame;
    mplastFrame = plastFrame;
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

    mpcurrentFrame = new Frame(timeStamps, mpcamera, mpLineFeature, mpPointFeature);

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
        // use the pnp ransanc to remove the outliers
        TrackRefFrame(vpointRefineMatches);

        mpcurrentFrame->UnprojectStereo(mimageDepth, vpointRefineMatches, vlineRefineMatches, true);

        mplastFrame->UnprojectStereo(mlastimageDepth, vpointRefineMatches, vlineRefineMatches, false);


        UpdateMapLPfeature(vpointRefineMatches, vlineRefineMatches);

        mpcurrentFrame->MapLinePointShow();

        Optimizer::PoseOptimization(mpcurrentFrame);

        cout << mpcurrentFrame->Tcw << endl;

        cv::Mat showimg;
        cv::drawMatches(mlastimagergb, mplastFrame->mvKeyPoint, mimagergb, mpcurrentFrame->mvKeyPoint,
                        vpointRefineMatches, showimg);

        std::vector<char> mask(vlineRefineMatches.size(), 1);
        cv::line_descriptor::drawLineMatches(mlastimagergb, mplastFrame->mvKeyLine, mimagergb, mpcurrentFrame->mvKeyLine,
                                             vlineRefineMatches, showimg,  cv::Scalar::all(-1), cv::Scalar::all(-1), mask,
                                             cv::line_descriptor::DrawLinesMatchesFlags::DEFAULT);
        cv::imshow(" ", showimg);
        cv::waitKey(5);
    }

    mpMap->mlpFrames.push_back(mpcurrentFrame);

    mplastFrame = new Frame(*mpcurrentFrame);
    mlastimageGrays = mimageGray.clone();
    mlastimagergb = mimagergb.clone();
    mlastimageDepth = mimageDepth.clone();
}

bool Tracking::TrackRefFrame(vector<cv::DMatch> &vpointMatches)
{
    vector<cv::Point3d> vpts3d;
    vector<cv::Point2d> vpts2d;
    Sophus::SE3 PoseInc;

    for (auto match:vpointMatches)
    {

        double d = mplastFrame->FindDepth(mplastFrame->mvKeyPoint[match.queryIdx].pt, mlastimageDepth);
        if (d < 0)
            continue;

        Eigen::Vector3d Point3dw;
        Point3dw = mpcamera->Pixwl2World(Converter::toVector2d(mplastFrame->mvKeyPointUn[match.queryIdx].pt),
                                         Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 0, 0), d);

        vpts3d.push_back(Converter::toCvPoint3f(Point3dw));
        vpts2d.push_back(mpcurrentFrame->mvKeyPointUn[match.trainIdx].pt);
    }

    cv::Mat rvec, tvec;
    vector<int> inliers;
    cv::solvePnPRansac(vpts3d, vpts2d, Converter::toCvMat(mpcamera->GetCameraIntrinsic()), cv::Mat(),
                       rvec, tvec, false, 100, 4.0, 0.99, inliers);

    int idx = 0;
    for (auto inlier : inliers)
    {
        vpointMatches[idx] = vpointMatches[inlier];
        idx++;
    }

    vpointMatches.resize(inliers.size());

    PoseInc = Sophus::SE3(Sophus::SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)),
                          Converter::toVector3d(tvec));

    mpcurrentFrame->Tcw = mplastFrame->Tcw*PoseInc;
}

void Tracking::UpdateMapPointfeature(const vector<cv::DMatch> &vpointMatches)
{
    for (auto match : vpointMatches)
    {
        PointFeature2D *pPointFeature = mplastFrame->mvpPointFeature2D[match.queryIdx];

        CHECK_NOTNULL(pPointFeature);

        if (pPointFeature->midxMatch != match.queryIdx)
        {
            LOG(ERROR) << "the idx of the PointFeature 2d is wrong " << endl;
        }

        if (pPointFeature->mpMapPoint == nullptr)
        {
            MapPoint *pMapPoint = new MapPoint;

            pMapPoint->mID = countMapPoint;
            pMapPoint->mPosew = pPointFeature->mPoint3dw;
            pMapPoint->mmpPointFeature2D[mplastFrame->GetFrameID()] = pPointFeature;
            pMapPoint->mlpFrameinvert.push_back(mplastFrame);

            mplastFrame->mvpMapPoint.push_back(pMapPoint);
            pPointFeature->mpMapPoint = pMapPoint;

            countMapPoint++;

            PointFeature2D *pcurPointFeature = mpcurrentFrame->mvpPointFeature2D[match.trainIdx];
            CHECK_NOTNULL(pcurPointFeature);

            if (pcurPointFeature->midxMatch != match.trainIdx)
            {
                LOG(ERROR) << "the idx of the PointFeature 2d is wrong " << endl;
            }

            pMapPoint->mmpPointFeature2D[mpcurrentFrame->GetFrameID()] = pcurPointFeature;
            pMapPoint->mlpFrameinvert.push_back(mpcurrentFrame);

            mpcurrentFrame->mvpMapPoint.push_back(pMapPoint);
            pcurPointFeature->mpMapPoint = pMapPoint;

        }
        else // if (pPointFeature->mpMapPoint == nullptr)
        {
            MapPoint *pMapPoint = pPointFeature->mpMapPoint;

            PointFeature2D *pcurPointFeature = mpcurrentFrame->mvpPointFeature2D[match.trainIdx];
            CHECK_NOTNULL(pcurPointFeature);

            if (pcurPointFeature->midxMatch != match.trainIdx)
            {
                LOG(ERROR) << "the idx of the PointFeature 2d is wrong " << endl;
            }

            pMapPoint->mmpPointFeature2D[mpcurrentFrame->GetFrameID()] = pcurPointFeature;
            pMapPoint->mlpFrameinvert.push_back(mpcurrentFrame);

            if (pMapPoint->mPosew.isZero())
            {
                pMapPoint->mPosew = pcurPointFeature->mPoint3dw;
//                cout << "use the current frame's observation to updae the MapPoint: " << pcurPointFeature->mPoint3dw.transpose() << endl;
            }

            mpcurrentFrame->mvpMapPoint.push_back(pMapPoint);
            pcurPointFeature->mpMapPoint = pMapPoint;

        } // if (pPointFeature->mpMapPoint == nullptr)

    } // for (auto match : vpointMatches)

}

void Tracking::UpdateMapLinefeature(const vector<cv::DMatch> &vlineMatches)
{

    for (auto match : vlineMatches)
    {
        LineFeature2D *pLineFeature = mplastFrame->mvpLineFeature2D[match.queryIdx];

        CHECK_NOTNULL(pLineFeature);

        if (pLineFeature->midxMatch != match.queryIdx)
        {
            LOG(ERROR) << "the idx of the LineFeature 2d is wrong " << endl;
        }

        if (pLineFeature->mpMapLine == nullptr)
        {
            MapLine *pMapLine = new MapLine;
            pMapLine->mID = countMapLine;
            pMapLine->mPoseStartw = pLineFeature->mStartPoint3dw;
            pMapLine->mPoseEndw = pLineFeature->mEndPoint3dw;
            pMapLine->mlpFrameinvert.push_back(mplastFrame);
            pMapLine->mmpLineFeature2D[mplastFrame->GetFrameID()] = pLineFeature;

            mplastFrame->mvpMapLine.push_back(pMapLine);
            pLineFeature->mpMapLine = pMapLine;

            countMapLine++;

            LineFeature2D *pcurLineFeature = mpcurrentFrame->mvpLineFeature2D[match.trainIdx];
            CHECK_NOTNULL(pcurLineFeature);

            if (pcurLineFeature->midxMatch != match.trainIdx)
            {
                LOG(ERROR) << "the idx of the LineFeature 2d is wrong " << endl;
            }

            pMapLine->mmpLineFeature2D[mpcurrentFrame->GetFrameID()] = pcurLineFeature;
            pMapLine->mlpFrameinvert.push_back(mpcurrentFrame);

            mpcurrentFrame->mvpMapLine.push_back(pMapLine);
            pcurLineFeature->mpMapLine = pMapLine;

        }
        else // if (pLineFeature->pMapLine == nullptr)
        {
            MapLine *pMapLine = pLineFeature->mpMapLine;

            LineFeature2D *pcurLineFeature = mpcurrentFrame->mvpLineFeature2D[match.trainIdx];
            CHECK_NOTNULL(pcurLineFeature);

            if (pcurLineFeature->midxMatch != match.trainIdx)
            {
                LOG(ERROR) << "the idx of the LineFeature 2d is wrong " << endl;
            }

            pMapLine->mmpLineFeature2D[mpcurrentFrame->GetFrameID()] = pcurLineFeature;
            pMapLine->mlpFrameinvert.push_back(mpcurrentFrame);

            if (pMapLine->mPoseStartw.isZero() || pMapLine->mPoseEndw.isZero())
            {
//                cout << "use the current frame's observation to updae the MapLine: " << pcurLineFeature->mStartPoint3dw.transpose() << endl;
//                cout << "use the current frame's observation to updae the MapLine: " << pcurLineFeature->mEndPoint3dw.transpose() << endl;

                if (!pcurLineFeature->mStartPoint3dw.isZero())
                {
                    pMapLine->mPoseStartw = pcurLineFeature->mStartPoint3dw;
                }

                if (!pcurLineFeature->mEndPoint3dw.isZero())
                {
                    pMapLine->mPoseEndw = pcurLineFeature->mEndPoint3dw;
                }
            }

            mpcurrentFrame->mvpMapLine.push_back(pMapLine);
            pcurLineFeature->mpMapLine = pMapLine;

        } // if (pLineFeature->pMapLine == nullptr)
    } // for (auto match : vlineMatches)
}

void Tracking::UpdateMapLPfeature(const vector<cv::DMatch> &vpointMatches, const vector<cv::DMatch> &vlineMatches)
{
    if (Config::plInParallel())
    {
        auto updateMapLine = async(launch::async, &Tracking::UpdateMapLinefeature, this, vlineMatches);
        auto updateMapPoint = async(launch::async, &Tracking::UpdateMapPointfeature, this, vpointMatches);
    }
    else
    {
        UpdateMapLinefeature(vlineMatches);
        UpdateMapPointfeature(vpointMatches);
    }
}



} // namespace PL_VO
