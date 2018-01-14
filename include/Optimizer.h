//
// Created by rain on 18-1-3.
//

#ifndef PL_VO_OPTIMIZER_H
#define PL_VO_OPTIMIZER_H

#include <math.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <sophus/se3.h>
#include <sophus/so3.h>
#include "Converter.h"
#include "Frame.h"

namespace PL_VO
{

class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;

    virtual bool ComputeJacobian(const double *x, double *jacobian) const;

    virtual int GlobalSize() const { return 7; };

    virtual int LocalSize() const { return 6; };

}; // class PoseLocalParameterization

/**
 * @brief the ceres::SizedCostFuntion<2, 7, 3>
 *        the 2 is number of the residuals
 *        the 7 is number of the camera pose parameters
 *        the 3 is number of the MapPoint pose parameters
 */
class ReprojectionErrorSE3 : public ceres::SizedCostFunction<2, 7, 3>
{

private:

    double observedx;
    double observedy;

public:

    double fx;
    double fy;
    double cx;
    double cy;

    ReprojectionErrorSE3(double fx_, double fy_, double cx_, double cy_, double observedx_, double observedy_):
            fx(fx_), fy(fy_), cx(cx_), cy(cy_), observedx(observedx_), observedy(observedy_) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

//    void check(double **parameters);

}; // class ReprojectionErrorSE3

/**
 * @brief the ceres::SizedCostFuntion<2, 7, 6>
 *        the 2 is number of the residuals
 *        the 7 is number of the camera pose parameters
 *        the 6 is number of the MapLine pose parameters and  the end point and the start point of the line
 */
class ReprojectionLineErrorSE3 : public ceres::SizedCostFunction<2, 7, 3, 3>
{
private:

    Eigen::Vector2d Startpixel = Eigen::Vector2d(0, 0);
    Eigen::Vector2d Endpixel = Eigen::Vector2d(0, 0);
    Eigen::Vector3d lineCoef = Eigen::Vector3d(0, 0, 0);

public:

    double fx;
    double fy;
    double cx;
    double cy;

    ReprojectionLineErrorSE3(double fx_, double fy_, double cx_, double cy_,
                             const Eigen::Vector2d &Startpixel_, const Eigen::Vector2d &Endpixel_,
                             const Eigen::Vector3d &lineCoef_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_),
                             Startpixel(Startpixel_), Endpixel(Endpixel_), lineCoef(lineCoef_){}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

};

class Optimizer
{
public:
    static void PoseOptimization(Frame *pFrame);

    static Eigen::Vector2d ReprojectionError(const ceres::Problem& problem, ceres::ResidualBlockId id);

    static std::vector<double> GetReprojectionErrorNorms(const ceres::Problem& problem);

    static void RemoveOutliers(ceres::Problem& problem, double threshold);
}; // class Optimizer

} //namespace PL_VO

#endif //PL_VO_OPTIMIZER_H
