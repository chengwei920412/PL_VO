//
// Created by rain on 17-10-19.
//

#ifndef PL_VO_CONVERTER_H
#define PL_VO_CONVERTER_H

#include <iostream>
#include <vector>
#include <list>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace PL_VO
{

class Converter
{
public:
    static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);
    static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 3> &m);
    static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m);
    static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t);


    static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat &cvVector);
    static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f &cvPoint);
    static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat &cvMat3);

    static std::vector<double> toQuaternion(const cv::Mat &m);

    static Eigen::Matrix<double, 2, 1> toVector2d(const cv::Point2f &cvPoint);
    static cv::Point2f toCvPoint2f(const Eigen::Vector2d &v);
    static cv::Point3f toCvPoint3f(const Eigen::Vector3d &v);

    static Eigen::Vector3d toEuler(const Eigen::Quaterniond &q);
    static Eigen::Vector3d toEuler(const Eigen::Matrix3d &R);

    static Eigen::Matrix3d skew(const Eigen::Vector3d &v);

    static Eigen::Matrix<double, 4, 4> quatLeftproduct(const Eigen::Quaterniond &q0);
    static Eigen::Matrix<double, 4, 4> quatRightproduct(const Eigen::Quaterniond &q0);

}; // class Converter

} // namesapce PL_VO

#endif //PL_VO_CONVERTER_H
