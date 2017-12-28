//
// Created by rain on 17-10-19.
//

#include "Converter.h"
#include <glog/logging.h>

namespace PL_VO
{

cv::Mat Converter::toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_64F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<double>(i,j) = (double)m(i,j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 3, 3> &m)
{
    cv::Mat cvMat(3,3,CV_64F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.at<double>(i,j) = (double)m(i,j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 3, 1> &m)
{
    cv::Mat cvMat(3,1,CV_64F);
    for(int i=0;i<3;i++)
        cvMat.at<double>(i) = (double)m(i);

    return cvMat.clone();
}

cv::Mat Converter::toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t)
{
    cv::Mat cvMat = cv::Mat::eye(4,4,CV_64F);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            cvMat.at<double>(i,j) = (double)R(i,j);
        }
    }
    for(int i=0;i<3;i++)
    {
        cvMat.at<double>(i,3) = (double)t(i);
    }

    return cvMat.clone();
}

Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.at<double>(0), cvVector.at<double>(1), cvVector.at<double>(2);

    return v;
}

Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::Point3f &cvPoint)
{
    Eigen::Matrix<double,3,1> v;
    v << cvPoint.x, cvPoint.y, cvPoint.z;

    return v;
}

Eigen::Matrix<double,3,3> Converter::toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<double>(0,0), cvMat3.at<double>(0,1), cvMat3.at<double>(0,2),
         cvMat3.at<double>(1,0), cvMat3.at<double>(1,1), cvMat3.at<double>(1,2),
         cvMat3.at<double>(2,0), cvMat3.at<double>(2,1), cvMat3.at<double>(2,2);

    return M;
}

std::vector<double> Converter::toQuaternion(const cv::Mat &m)
{
    Eigen::Matrix<double,3,3> eigMat = toMatrix3d(m);
    Eigen::Quaterniond q(eigMat);

    std::vector<double> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}

Eigen::Matrix<double, 2, 1> Converter::toVector2d(const cv::Point2f &cvPoint)
{
    Eigen::Matrix<double, 2, 1> v;
    v << cvPoint.x, cvPoint.y;

    return v;
}

cv::Point2f Converter::toCvPoint2f(const Eigen::Vector2d &v)
{
    cv::Point2f cvPoint2f;
    cvPoint2f.x = (float)v[0];
    cvPoint2f.y = (float)v[1];

    return cvPoint2f;
}

cv::Point3f Converter::toCvPoint3f(const Eigen::Vector3d &v)
{
    cv::Point3f cvPoint3f;
    cvPoint3f.x = (float)v[0];
    cvPoint3f.y = (float)v[1];
    cvPoint3f.z = (float)v[2];

    return cvPoint3f;
}

/**
 * @brief euler angle； yaw, pitch, roll
 * @param q
 * @return
 */
Eigen::Vector3d Converter::toEuler(const Eigen::Quaterniond &q)
{
    double yaw, pitch, roll;
    double r1 = 2*(q.w()*q.x() + q.y()*q.z());
    double r2 = 1 - 2*(q.x()*q.x() + q.y()*q.y());
    double r3 = 2*(q.w()*q.y() - q.z()*q.x());
    double r4 = 2*(q.w()*q.z() + q.x()*q.y());
    double r5 = 1 - 2*(q.y()*q.y() + q.z()*q.z());

    roll = atan2(r1, r2);
    pitch = asin(r3);
    yaw = atan2(r4, r5);

    Eigen::Vector3d euler(yaw,pitch,roll);

    return euler;
}

Eigen::Vector3d Converter::toEuler(const Eigen::Matrix3d &R)
{
    double yaw, pitch, roll;

    CHECK(R(0,0) != 0) << "converter matrix to euler is wrong " << std::endl;
    CHECK(R(2,2) != 0) << "converter matrix to euler is wrong " << std::endl;

    roll = atan2(R(2,1), R(2,2));
    pitch = asin(-R(2, 0));
    yaw = atan2(R(1,0), R(0,0));

    Eigen::Vector3d euler(yaw,pitch,roll);

    return euler;
}

Eigen::Matrix3d Converter::skew(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m.setZero();

    m <<    0, -v(2),  v(1),
         v(2),     0, -v(0),
        -v(1),  v(0),     0;

    return m;
}

Eigen::Matrix<double, 4, 4> Converter::quatLeftproduct(const Eigen::Quaterniond &q0)
{
    Eigen::Matrix<double, 4, 4> qL;

    qL << q0.w(), -q0.x(), -q0.y(), -q0.z(),
          q0.x(),  q0.w(), -q0.z(),  q0.y(),
          q0.y(),  q0.z(),  q0.w(), -q0.x(),
          q0.z(), -q0.y(),  q0.x(),  q0.w();

    return qL;
}

Eigen::Matrix<double, 4, 4> Converter::quatRightproduct(const Eigen::Quaterniond &q0)
{
    Eigen::Matrix<double, 4, 4> qR;

    qR << q0.w(), -q0.x(), -q0.y(), -q0.z(),
          q0.x(),  q0.w(),  q0.z(), -q0.y(),
          q0.y(), -q0.z(),  q0.w(),  q0.x(),
          q0.z(),  q0.y(), -q0.x(),  q0.w();

    return qR;
}

} // namesapce RAIN_VIO