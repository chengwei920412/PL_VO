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

void Frame::UndistorKeyFeature()
{
    // TODO there are some keypoints are not used, but also undistor they
    {
        if (mpCamera->GetDistortionPara()[0] == 0.)
        {
            mvKeyPointUn.assign(mvKeyPoint.begin(), mvKeyPoint.end());
            return;
        }

        cv::Mat mat((int)mvKeyPoint.size(), 2, CV_32F);
        for (int i = 0; i < mvKeyPoint.size(); i++)
        {
            mat.at<float>(i, 0) = mvKeyPoint[i].pt.x;
            mat.at<float>(i, 1) = mvKeyPoint[i].pt.y;
        }

        mat = mat.reshape(2);

        cv::Mat DistCoef(4,1,CV_32F);
        DistCoef.at<float>(0) = (float)mpCamera->GetDistortionPara()[0];
        DistCoef.at<float>(1) = (float)mpCamera->GetDistortionPara()[1];
        DistCoef.at<float>(2) = (float)mpCamera->GetDistortionPara()[2];
        DistCoef.at<float>(3) = (float)mpCamera->GetDistortionPara()[3];

        cv::Mat K;
        K = Converter::toCvMat(mpCamera->GetCameraIntrinsic());

        cv::undistortPoints(mat, mat, K,  DistCoef, cv::Mat(), K);

        mat = mat.reshape(1);

        mvKeyPointUn.resize(mvKeyPoint.size());

        for (int i = 0; i < mvKeyPoint.size(); i++)
        {
            cv::KeyPoint kp = mvKeyPoint[i];
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeyPointUn[i] = kp;
        }
    }

    {
        // Line feature
        if (mpCamera->GetDistortionPara()[0] == 0)
        {
            mvKeyLineUn.assign(mvKeyLine.begin(), mvKeyLine.end());
            return ;
        }

        cv::Mat mat2(2*(int)mvKeyLine.size(), 2, CV_32F);
        for (int i = 0; i < mvKeyLine.size(); i++)
        {
            mat2.at<float>(2*i, 0) = mvKeyLine[i].startPointX;
            mat2.at<float>(2*i, 1) = mvKeyLine[i].startPointY;
            mat2.at<float>(2*i+1, 0) = mvKeyLine[i].endPointX;
            mat2.at<float>(2*i+1, 1) = mvKeyLine[i].endPointY;
        }

        mat2 = mat2.reshape(2);

        cv::Mat DistCoef2(4,1,CV_32F);
        DistCoef2.at<float>(0) = (float)mpCamera->GetDistortionPara()[0];
        DistCoef2.at<float>(1) = (float)mpCamera->GetDistortionPara()[1];
        DistCoef2.at<float>(2) = (float)mpCamera->GetDistortionPara()[2];
        DistCoef2.at<float>(3) = (float)mpCamera->GetDistortionPara()[3];

        cv::Mat K2;
        K2 = Converter::toCvMat(mpCamera->GetCameraIntrinsic());

        cv::undistortPoints(mat2, mat2, K2,  DistCoef2, cv::Mat(), K2);

        mat2 = mat2.reshape(1);

        mvKeyLineUn.resize(mvKeyLine.size());

        for (int i = 0; i < mvKeyLine.size(); i++)
        {
            cv::line_descriptor::KeyLine kl = mvKeyLine[i];

            if (min(mat2.at<float>(2*i, 0), mat2.at<float>(2*i+1, 0)) <= 0)
            {
                cout << min(mat2.at<float>(2*i, 0), mat2.at<float>(2*i+1, 0)) << endl;
                continue;
            }

            if (min(mat2.at<float>(2*i, 1), mat2.at<float>(2*i+1, 1)) <= 0)
            {
                cout << min(mat2.at<float>(2*i, 1), mat2.at<float>(2*i+1, 1)) << endl;
                continue;
            }

            if (max(mat2.at<float>(2*i, 0), mat2.at<float>(2*i+1, 0)) >= mpCamera->mImageWidth)
            {
                cout << max(mat2.at<float>(2*i, 0), mat2.at<float>(2*i+1, 0) ) << endl;
                continue;
            }

            if (max(mat2.at<float>(2*i, 1), mat2.at<float>(2*i+1, 1)) >= mpCamera->mImageHeight)
            {
                cout << max(mat2.at<float>(2*i, 1), mat2.at<float>(2*i+1, 1) ) << endl;
                continue;
            }

            kl.startPointX = mat2.at<float>(2*i, 0);
            kl.startPointY = mat2.at<float>(2*i, 1);
            kl.endPointX = mat2.at<float>(2*i+1, 0);
            kl.endPointY = mat2.at<float>(2*i+1, 1);
            mvKeyLineUn[i] = kl;
        }
    }

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

double Frame::FindDepth(const cv::KeyPoint &kp, const cv::Mat &imagedepth)
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
            d = imagedepth.ptr<ushort>(y+dy[i])[x+dx[i]];
            if (d != 0)
            {
                return double(d)/mpCamera->mdepthscale;
            }
        }
    }

    return -1.;
}

void Frame::AddMapPoint(const cv::Mat &imageDepth, const vector<cv::DMatch> vpointMatches,
                        const vector<cv::DMatch> vlineMatches)
{
    for (auto match : vpointMatches)
    {
        double d = FindDepth(mvKeyPoint[match.queryIdx], imageDepth);
        if (d < 0)
            continue;

        Eigen::Vector3d Point3dw;
        Point3dw = mpCamera->Pixwl2World(Converter::toVector2d(mvKeyPoint[match.queryIdx].pt),
                                         Tcw.so3().unit_quaternion(), Tcw.translation(), d);
    }
}

} // namespace PL_VO
