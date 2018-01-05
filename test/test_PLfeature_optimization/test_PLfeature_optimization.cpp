//
// Created by rain on 18-1-2.
//

#include <Camera.h>
#include <Converter.h>
#include "Frame.h"
#include "Optimizer.h"


int main(int argc, char* argv[])
{
    PL_VO::Camera *pcamera;
    PL_VO::PointFeature *ppointFeature;
    PL_VO::LineFeature *plineFeature;
    PL_VO::Frame *pcurrentFrame;
    PL_VO::Frame *plastFrame;

    plineFeature = new(PL_VO::LineFeature);
    ppointFeature = new(PL_VO::PointFeature);
    pcamera = new PL_VO::Camera("../Example/TUM2.yaml");

    const string strimg1FilePath = "../test/test_line_match/1.png";
    const string strimg2FilePath = "../test/test_line_match/2.png";
    const string strimg1depthFilePath = "../test/test_line_match/1_depth.png";
    const string strimg2depthFilePath = "../test/test_line_match/2_depth.png";

    cv::Mat img1 = cv::imread(strimg1FilePath, CV_LOAD_IMAGE_COLOR);
    if (img1.empty())
    {
        cout << "can not load the image " << strimg1FilePath << endl;
        return 0;
    }
    cv::Mat img2 = cv::imread(strimg2FilePath, CV_LOAD_IMAGE_COLOR);
    cv::Mat img1depth = cv::imread(strimg1depthFilePath, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat img2depth = cv::imread(strimg2depthFilePath, CV_LOAD_IMAGE_UNCHANGED);

    cv::Mat imagepointshow;
    cv::Mat imagepointsunhow;
    cv::Mat imagelineshow;
    cv::Mat imagelineunshow;
    cv::Mat img1un;
    cv::Mat img2un;

    pcurrentFrame = new PL_VO::Frame(1.0, pcamera, plineFeature, ppointFeature);
    plastFrame = new PL_VO::Frame(2.0, pcamera, plineFeature, ppointFeature);

    pcurrentFrame->detectFeature(img2, img2depth);
    plastFrame->detectFeature(img1, img1depth);

    vector<cv::DMatch> vpointMatches;
    vector<cv::DMatch> vpointRefineMatches;
    vector<cv::DMatch> vlineMatches;
    vector<cv::DMatch> vlineRefineMatches;

    pcurrentFrame->matchLPFeature(plastFrame->mpointDesc, pcurrentFrame->mpointDesc, vpointMatches,
                                  plastFrame->mlineDesc, pcurrentFrame->mlineDesc, vlineMatches);

    pcurrentFrame->refineLPMatches(plastFrame->mvKeyPoint, pcurrentFrame->mvKeyPoint,
                                   plastFrame->mvKeyLine, pcurrentFrame->mvKeyLine,
                                   vpointMatches, vpointRefineMatches, vlineMatches, vlineRefineMatches);

    pcurrentFrame->UndistortKeyFeature();
    plastFrame->UndistortKeyFeature();

    for (size_t i = 0; i < pcurrentFrame->mvKeyPoint.size(); i++)
    {
        cv::circle(imagepointshow, pcurrentFrame->mvKeyPoint[i].pt, 2, cv::Scalar(255, 0, 0), 2);
    }
    for (size_t i = 0; i < pcurrentFrame->mvKeyPointUn.size(); i++)
    {
        cv::circle(imagepointsunhow, pcurrentFrame->mvKeyPointUn[i].pt, 2, cv::Scalar(255, 0, 0), 2);
    }

    cv::line_descriptor::drawKeylines(img2, pcurrentFrame->mvKeyLine, imagelineshow);
    cv::line_descriptor::drawKeylines(img2, pcurrentFrame->mvKeyLineUn, imagelineunshow);

    cv::imshow(" image lineun show ", imagelineunshow);
    cv::imshow(" image line show ", imagelineshow);

    {
        vector<cv::Point3d> vpts3d;
        vector<cv::Point2d> vpts2d;

        plastFrame->Tcw.so3().setQuaternion(Eigen::Quaterniond::Identity());
        plastFrame->Tcw.translation() = Eigen::Vector3d(0, 0, 0);

        cout << "matches size: " << vpointRefineMatches.size() << endl;
        for (auto match:vpointRefineMatches)
        {
            double d = plastFrame->FindDepth(plastFrame->mvKeyPointUn[match.queryIdx].pt, img1depth);
            if (d < 0)
                continue;

            Eigen::Vector3d Point3dw;
            Point3dw = pcamera->Pixwl2World(PL_VO::Converter::toVector2d(plastFrame->mvKeyPointUn[match.queryIdx].pt),
                                            plastFrame->Tcw.so3().unit_quaternion(), plastFrame->Tcw.translation(), d);

            vpts3d.push_back(PL_VO::Converter::toCvPoint3f(Point3dw));
            vpts2d.push_back(pcurrentFrame->mvKeyPointUn[match.trainIdx].pt);

        }

        cv::Mat rvec, tvec, inliers;
        cv::solvePnPRansac(vpts3d, vpts2d, PL_VO::Converter::toCvMat(pcamera->GetCameraIntrinsic()), cv::Mat(),
                           rvec, tvec, false, 100, 4.0, 0.99, inliers);

        pcurrentFrame->Tcw = Sophus::SE3(Sophus::SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)),
                                         PL_VO::Converter::toVector3d(tvec));

        cout << pcurrentFrame->Tcw << endl;
    } // -0.12910648864000002  -0.011553516588 0.056891304389999994


    pcurrentFrame->UnprojectStereo(img2depth, vpointRefineMatches, vlineRefineMatches);

    PL_VO::Optimizer::PoseOptimization(pcurrentFrame);

    cv::Mat showimg;
//    cv::drawMatches(img1, plastFrame->mvKeyPointUn, img2, pcurrentFrame->mvKeyPointUn, vpointRefineMatches, showimg);

    std::vector<char> mask(vlineRefineMatches.size(), 1);
    cv::line_descriptor::drawLineMatches(img1, plastFrame->mvKeyLineUn, img2, pcurrentFrame->mvKeyLineUn,
                                         vlineRefineMatches, showimg,  cv::Scalar::all(-1), cv::Scalar::all(-1), mask,
                                         cv::line_descriptor::DrawLinesMatchesFlags::DEFAULT);
    cv::imshow(" ", showimg);

    cv::waitKey(0);

    return 0;
}