//
// Created by rain on 18-1-3.
//

#include "Optimizer.h"

namespace PL_VO
{

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *xplusdelta) const
{
    // x is the seven dimensional space
    Eigen::Map<const Eigen::Vector3d> trans(x + 4);
    Eigen::Map<const Eigen::Quaterniond> quaterd(x);

    // delta is the parameter in the parameter space
    // delta is the six dimensional space
    Sophus::SE3 se3delta = Sophus::SE3::exp(Eigen::Map<const Eigen::Matrix<double, 6, 1, Eigen::ColMajor>>(delta));

    Eigen::Map<Eigen::Quaterniond> quaterdplus(xplusdelta);
    Eigen::Map<Eigen::Vector3d> transplus(xplusdelta + 4);

    quaterdplus = se3delta.so3().matrix() * quaterd;
    transplus = se3delta.so3().matrix() * trans + se3delta.translation();

    return true;
}


bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > J(jacobian);
    J.setZero();
    J.block<6,6>(0, 0).setIdentity();
    return true;
}

/**
 * @brief rotation(quaternion), translation(vector3d), point3d(vector3d)
 * @param parameters
 * @param residuals
 * @param jacobians
 * @return
 */

bool ReprojectionErrorSE3::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Map<const Eigen::Quaterniond> quaterd(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> trans(parameters[0] + 4);
    Eigen::Map<const Eigen::Vector3d> point(parameters[1]);

    Eigen::Vector3d p = quaterd * point + trans;

    residuals[0] = fx*p[0]/p[2] + cx - observedx;
    residuals[1] = fy*p[1]/p[2] + cy - observedy;

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> jacobian;

    jacobian << fx/p[2],  0, -fx*p[0]/p[2]/p[2],
            0, fy/p[2], -fy*p[1]/p[2]/p[2];

    if(jacobians != nullptr)
    {
        if(jacobians[0] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > Jse3(jacobians[0]);
            Jse3.setZero();

            // very important! the form of the se3 is the rotation in the front and the transformation in the back
            Jse3.block<2,3>(0,0) = -jacobian*Converter::skew(p);
            Jse3.block<2,3>(0,3) = jacobian;
        }
        if(jacobians[1] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > Jpoint(jacobians[1]);
            Jpoint = jacobian * quaterd.toRotationMatrix();
        }
    }

    return true;
}

void Optimizer::PoseOptimization(Frame *pFrame)
{
//    Eigen::Matrix3d K;
//    K = pFrame->mpCamera->GetCameraIntrinsic();
//
//    cv::Mat extrinsic(7, 1, CV_64FC1);
//
//    {
//        extrinsic.ptr<double>()[0] = pFrame->Tcw.unit_quaternion().x();
//        extrinsic.ptr<double>()[1] = pFrame->Tcw.unit_quaternion().y();
//        extrinsic.ptr<double>()[2] = pFrame->Tcw.unit_quaternion().z();
//        extrinsic.ptr<double>()[3] = pFrame->Tcw.unit_quaternion().w();
//        pFrame->Tcw.translation().copyTo(extrinsic.rowRange(4, 7));
//    }
//
//    cout << "extrinsic: " << endl << extrinsic.t() << endl;
//
//    ceres::Problem problem;
//
//    problem.AddParameterBlock(extrinsic.ptr<double>(), 7, new PoseLocalParameterization());
//
//
//    ceres::LossFunction* lossfunction = new ceres::HuberLoss(4);   // loss function make bundle adjustment robuster.
//
//    for (int i = 0; i < vPoints3d.size(); i++)
//    {
//        cv::Point2d observed = vPoints2d[i];
//
//        ceres::CostFunction* costfunction = new ReprojectionErrorSE3(K(0, 0), K(1, 1),
//                                                                     K(0, 2), K(1, 2),
//                                                                     observed.x, observed.y);
//
//        problem.AddResidualBlock(
//                costfunction, lossfunction,
//                extrinsic.ptr<double>(), &vPoints3d[i].x);
//
//        problem.AddParameterBlock(&vPoints3d[i].x, 3);
//    }
//
//    ceres::Solver::Options options;
//    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.minimizer_progress_to_stdout = true;
//    options.max_solver_time_in_seconds = 0.3;
//
//    ceres::Solver::Summary summary;
//    ceres::Solve(options, &problem, &summary);
//
//    if (!summary.IsSolutionUsable())
//    {
//        cout << "Bundle Adjustment failed." << std::endl;
//    }
//    else
//    {
//        // Display statistics about the minimization
//        cout << summary.BriefReport() << endl
//             << " residuals number: " << summary.num_residuals << endl
//             << " Initial RMSE: " << sqrt(summary.initial_cost / summary.num_residuals) << endl
//             << " Final RMSE: " << sqrt(summary.final_cost / summary.num_residuals) << endl
//             << " Time (s): " << summary.total_time_in_seconds << endl;
//
//        cout << extrinsic.t() << endl;
//    }
}

} // namespace PL_VO