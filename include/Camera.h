//
// Created by rain on 17-12-19.
//

#ifndef PL_VO_CAMERA_H
#define PL_VO_CAMERA_H

#include <string>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include "Converter.h"

namespace PL_VO
{

using namespace std;

class Camera
{

protected:

    double mfx;
    double mfy;
    double mcx;
    double mcy;
    double mk1;
    double mk2;
    double mp1;
    double mp2;
    double mk3;

public:

    int mImageHeight;
    int mImageWidth;
    int mnumFeatures;
    int minDist;
    int mFeatureShow;
    double mImageGridHeight;
    double mImageGridWidth;
    double mdepthscale;

public:

    Camera(const string &strSettingsFile);

    Camera();

    inline double Getfx() const {return mfx;}

    inline double Getfy() const {return mfy;}

    inline double Getcx() const {return mcx;}

    inline double Getcy() const {return mcy;}

    inline double Getfocal() const {return (mfx+mfy)/2;}

    inline Eigen::Vector4d GetDistortionPara() const {return Eigen::Vector4d(mk1, mk2, mp1, mp2);}

    inline Eigen::Matrix<double, 5, 1> GetDistortionPara2() const;

    inline Eigen::Matrix3d GetCameraIntrinsic() const;

    inline void SetCameraIntrinsic(Eigen::Matrix3d intrinsic);

    inline Eigen::Vector3d World2Camera(const Eigen::Vector3d &point3dw, const Eigen::Quaterniond &Rcw, const Eigen::Vector3d &tcw);

    inline Eigen::Vector3d Camera2World(const Eigen::Vector3d &point3dw, const Eigen::Quaterniond &Rcw, const Eigen::Vector3d &tcw);

    inline Eigen::Vector3d Pixel2Camera(const Eigen::Vector2d &point, double depth=1);

    inline Eigen::Vector2d Camera2Pixel(const Eigen::Vector3d &point3dc);

    inline Eigen::Vector2d Pixel2Camera2D(const Eigen::Vector2d &point2d);

    inline Eigen::Vector3d Pixwl2World(const Eigen::Vector2d &point2d, const Eigen::Quaterniond &Rcw, const Eigen::Vector3d &tcw,
                                       double depth=1);

    inline Eigen::Vector2d World2Pixel(const Eigen::Vector3d &point3dw, const Eigen::Quaterniond &Rcw, const Eigen::Vector3d &tcw);

    void Distortion(const Eigen::Vector2d &p, Eigen::Vector2d &du);

    void LiftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P);

    void InitUndistortRectifyMap(cv::Mat &undistmap1, cv::Mat &undistmap2);

}; // class Camera

inline Eigen::Matrix<double, 5, 1> Camera::GetDistortionPara2() const
{
    Eigen::Matrix<double, 5, 1> undistortmatrix;
    undistortmatrix << mk1, mk2, mp1, mp2, mk3;

    return undistortmatrix;
}


inline Eigen::Matrix3d Camera::GetCameraIntrinsic() const
{
    Eigen::Matrix3d intrinsic;
    intrinsic << mfx,  0., mcx,
                  0., mfy, mcy,
                  0.,  0.,  1.;

    return intrinsic;
}

inline void Camera::SetCameraIntrinsic(Eigen::Matrix3d intrinsic)
{
    mfx = intrinsic(0, 0);
    mfy = intrinsic(1, 1);
    mcx = intrinsic(0, 2);
    mcy = intrinsic(1, 2);
}

inline Eigen::Vector3d Camera::World2Camera(const Eigen::Vector3d &point3dw, const Eigen::Quaterniond &Rcw,
                                            const Eigen::Vector3d &tcw)
{
    return (Rcw*point3dw + tcw);
}

inline Eigen::Vector3d Camera::Camera2World(const Eigen::Vector3d &point3dw, const Eigen::Quaterniond &Rcw,
                                            const Eigen::Vector3d &tcw)
{
    Eigen::Quaterniond Rwc = Rcw.inverse();
    Eigen::Vector3d twc = Rwc*tcw*-1;

    return (Rwc*point3dw + twc);
}

inline Eigen::Vector2d Camera::Camera2Pixel(const Eigen::Vector3d &point3dc)
{
    return Eigen::Vector2d(mfx*point3dc(0)/point3dc(2) + mcx,
                           mfy*point3dc(1)/point3dc(2) + mcy);
}

inline Eigen::Vector3d Camera::Pixel2Camera(const Eigen::Vector2d &point, double depth)
{
    return Eigen::Vector3d((point[0] - mcx)*depth/mfx,
                           (point[1] - mcy)*depth/mfy,
                           depth);
}

inline Eigen::Vector2d Camera::Pixel2Camera2D(const Eigen::Vector2d &point2d)
{
    return Eigen::Vector2d((point2d[0] - mcx)/mfx,
                           (point2d[1] - mcy)/mfy);
}

inline Eigen::Vector3d Camera::Pixwl2World(const Eigen::Vector2d &point2d, const Eigen::Quaterniond &Rcw, const Eigen::Vector3d &tcw,
                                           double depth)
{
    return Camera2World(Pixel2Camera(point2d, depth), Rcw, tcw);
}

inline Eigen::Vector2d Camera::World2Pixel(const Eigen::Vector3d &point3dw, const Eigen::Quaterniond &Rcw,
                                           const Eigen::Vector3d &tcw)
{
    return Camera2Pixel(World2Camera(point3dw, Rcw, tcw));
}

} // namespace PL_VO

#endif //PL_VO_CAMERA_H
