//
// Created by rain on 18-1-3.
//

#ifndef PL_VO_OPTIMIZER_H
#define PL_VO_OPTIMIZER_H

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

class Optimizer
{
    void static PoseOptimization(Frame *pFrame);
}; // class Optimizer

} //namespace PL_VO

#endif //PL_VO_OPTIMIZER_H
