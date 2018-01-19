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


        mpcurrentFrame->UnprojectStereo(mimageDepth, vpointRefineMatches, vlineRefineMatches, true);

        mplastFrame->UnprojectStereo(mlastimageDepth, vpointRefineMatches, vlineRefineMatches, false);

        // use the pnp and point match to track the reference frame
        // use the pnp ransanc to remove the outliers
        TrackRefFrame(vpointRefineMatches, vlineRefineMatches);

        UpdateMapLPfeature(vpointRefineMatches, vlineRefineMatches);

//        mpcurrentFrame->MapLinePointShow();

//        Optimizer::PoseOptimization(mpcurrentFrame);

        cout << mpcurrentFrame->Tcw << endl;

        cv::Mat showimg;
        cv::drawMatches(mlastimagergb, mplastFrame->mvKeyPoint, mimagergb, mpcurrentFrame->mvKeyPoint,
                        vpointRefineMatches, showimg);

        std::vector<char> mask(vlineRefineMatches.size(), 1);
        cv::line_descriptor::drawLineMatches(mlastimagergb, mplastFrame->mvKeyLine, mimagergb, mpcurrentFrame->mvKeyLine,
                                             vlineRefineMatches, showimg,  cv::Scalar::all(-1), cv::Scalar::all(-1), mask,
                                             cv::line_descriptor::DrawLinesMatchesFlags::DEFAULT);
        cv::imshow(" ", showimg);
        cv::waitKey(0);
    }

    mpMap->mlpFrames.push_back(mpcurrentFrame);

    mplastFrame = new Frame(*mpcurrentFrame);
    mlastimageGrays = mimageGray.clone();
    mlastimagergb = mimagergb.clone();
    mlastimageDepth = mimageDepth.clone();
}

bool Tracking::TrackRefFrame(const vector<cv::DMatch> &vpointMatches, const vector<cv::DMatch> &vlineMatches)
{
    Sophus::SE3 PoseInc;

    vector<PointFeature2D> vPointFeature2D;
    vector<LineFeature2D *> vpLineFeature2D;
    vector<cv::Point3d> vPoint3d;
    vector<cv::Point2d> vPoint2d;
    vector<cv::Point2d> vLineStart2d;
    vector<cv::Point2d> vLineEnd2d;
    vector<cv::DMatch> vpointMatchesUnzero;

    vector<PointFeature2D *> vpPointFeature2DLast;
    vector<PointFeature2D *> vpPointFeature2DCur;
    vector<LineFeature2D *> vpLineFeature2DLast;
    vector<LineFeature2D *> vpLineFeature2DCur;

    {
//        cv::KeyPoint kp;
//
//        for (auto match : vpointMatches)
//        {
//            kp = mplastFrame->mvKeyPointUn[match.queryIdx];
//
//            Eigen::Vector3d Point3dw;
//            double d = mplastFrame->FindDepth(mplastFrame->mvKeyPoint[match.queryIdx].pt, mlastimageDepth);
//
//            if (d <= 0)
//                continue;
//
//            Point3dw = mpcamera->Pixwl2World(Converter::toVector2d(kp.pt), Eigen::Quaterniond::Identity(),
//                                             Eigen::Vector3d(0, 0, 0), d);
//
//            vPoint3d.emplace_back(Converter::toCvPoint3f(Point3dw));
//            vPoint2d.emplace_back(mpcurrentFrame->mvKeyPointUn[match.trainIdx].pt);
//
//            vpointMatchesUnzero.emplace_back(match);
//
//        } // for (auto match : vpointMatches)
    }

    {
        for (auto match : vpointMatches)
        {
            PointFeature2D *pPointFeature2DLast = mplastFrame->mvpPointFeature2D[match.queryIdx];
            PointFeature2D *pPointFeature2DCur = mpcurrentFrame->mvpPointFeature2D[match.trainIdx];

            CHECK_NOTNULL(pPointFeature2DLast);
            CHECK_NOTNULL(pPointFeature2DCur);

            if (!pPointFeature2DLast->mbinlier)
                continue;

            vPoint3d.emplace_back(Converter::toCvPoint3f(pPointFeature2DLast->mPoint3dw));
            vPoint2d.emplace_back(Converter::toCvPoint2f(pPointFeature2DCur->mpixel));
            vpPointFeature2DLast.emplace_back(pPointFeature2DLast);
            vpPointFeature2DCur.emplace_back(pPointFeature2DCur);
        }
    }

    {
//        for (auto match : vlineMatches)
//        {
//            cv::line_descriptor::KeyLine klUn; // in the last frame
//            cv::line_descriptor::KeyLine kl; // in the last frame
//            cv::line_descriptor::KeyLine kl2Un; // in the current frame
//
//            klUn = mplastFrame->mvKeyLineUn[match.queryIdx];
//            kl = mplastFrame->mvKeyLine[match.queryIdx];
//
//
//            cv::Point2f startPointUn2f;
//            cv::Point2f endPointUn2f;
//            cv::Point2f startPoint2f;
//            cv::Point2f endPoint2f;
//
//            startPointUn2f = cv::Point2f(klUn.startPointX, klUn.startPointY);
//            endPointUn2f = cv::Point2f(klUn.endPointX, klUn.endPointY);
//
//            startPoint2f = cv::Point2f(kl.startPointX, kl.startPointY);
//            endPoint2f = cv::Point2f(kl.endPointX, kl.endPointY);
//
//            // !!!notice: use the distored image
//            double d1 = mplastFrame->FindDepth(startPoint2f, mlastimageDepth);
//            double d2 = mplastFrame->FindDepth(endPoint2f, mlastimageDepth);
//
//            Eigen::Vector3d startPoint3dw;
//            Eigen::Vector3d endPoint3dw;
//
//            if (d1 <= 0 || d2 <= 0)
//                continue;
//
//
//            startPoint3dw = mpcamera->Pixwl2World(Converter::toVector2d(startPointUn2f), Eigen::Quaterniond::Identity(),
//                                                  Eigen::Vector3d(0, 0, 0), d1);
//
//            endPoint3dw = mpcamera->Pixwl2World(Converter::toVector2d(endPointUn2f), Eigen::Quaterniond::Identity(),
//                                                Eigen::Vector3d(0, 0, 0), d2);
//
//            LineFeature2D *plineFeature2D = new LineFeature2D(Converter::toVector2d(startPointUn2f), Converter::toVector2d(endPointUn2f),
//                                                        kl.octave, kl.response, match.queryIdx);
//
//            plineFeature2D->mStartPoint3dw = startPoint3dw;
//            plineFeature2D->mEndPoint3dw = endPoint3dw;
//
//            vpLineFeature2D.push_back(plineFeature2D);
//
//            // the observed keyline in the current frame
//            kl2Un = mpcurrentFrame->mvKeyLineUn[match.trainIdx];
//            vLineStart2d.emplace_back(cv::Point2d((double)kl2Un.startPointX, (double)kl2Un.startPointY));
//            vLineEnd2d.emplace_back(cv::Point2d((double)kl2Un.endPointX, (double)kl2Un.endPointY));
//        }
    }

    {
        for (auto match : vlineMatches)
        {
            LineFeature2D *pLineFeature2DLast = mplastFrame->mvpLineFeature2D[match.queryIdx];
            LineFeature2D *pLineFeature2DCur = mpcurrentFrame->mvpLineFeature2D[match.trainIdx];

            CHECK_NOTNULL(pLineFeature2DLast);
            CHECK_NOTNULL(pLineFeature2DCur);

            if (!pLineFeature2DLast->mbinlier)
                continue;

            vpLineFeature2DLast.emplace_back(pLineFeature2DLast);
            vpLineFeature2DCur.emplace_back(pLineFeature2DCur);
        }
    }

    cv::Mat rvec, tvec;
    vector<int> vinliers;
    cv::solvePnPRansac(vPoint3d, vPoint2d, Converter::toCvMat(mpcamera->GetCameraIntrinsic()), cv::Mat(),
                       rvec, tvec, false, 100, 4.0, 0.99, vinliers);

    {
        int idx = 0;
        for (size_t i = 0; i < vpPointFeature2DLast.size(); i++)
        {
            if (i == vinliers[idx])
                idx++;
            else
                vpPointFeature2DLast[i]->mbinlier = false;
        }
    }

    PoseInc = Sophus::SE3(Sophus::SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)),
                          Converter::toVector3d(tvec));

    cout <<  "PoseInc: " << endl << PoseInc << endl;

    Optimizer::PnPResultOptimization(mpcurrentFrame, PoseInc, vpPointFeature2DLast, vpPointFeature2DCur,
                                     vpLineFeature2DLast, vpLineFeature2DCur);

    mpcurrentFrame->Tcw = mplastFrame->Tcw*PoseInc;

    {
        for (auto it = vpLineFeature2D.begin(); it != vpLineFeature2D.end(); it ++)
            if (NULL != *it)
            {
                delete *it;
                *it = NULL;
            }
        vpLineFeature2D.clear();
    }
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

            auto it = pMapPoint->mmpPointFeature2D.find(mpcurrentFrame->GetFrameID());
            if (it != pMapPoint->mmpPointFeature2D.end())
            {
                LOG(ERROR) << "the point feature2d exist " << endl;
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

            auto it = pMapLine->mmpLineFeature2D.find(mpcurrentFrame->GetFrameID());
            if (it != pMapLine->mmpLineFeature2D.end())
            {
                LOG(ERROR) << "the point feature2d exist " << endl;
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
