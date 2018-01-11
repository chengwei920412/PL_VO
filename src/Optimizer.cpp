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
        if(jacobians[0] != nullptr) // the jacobian for the MapPoint pose optimization
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > Jse3(jacobians[0]);
            Jse3.setZero();

            // very important! the form of the se3 is the rotation in the front and the transformation in the back
            Jse3.block<2,3>(0,0) = jacobian;
            Jse3.block<2,3>(0,3) = -jacobian*Converter::skew(p);
        }
        if(jacobians[1] != nullptr) // the jacobian for the camera pose optimization
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > Jpoint(jacobians[1]);
            Jpoint = jacobian * quaterd.toRotationMatrix();
        }
    }

    return true;
}

bool ReprojectionLineErrorSE3::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Map<const Eigen::Quaterniond> quaterd(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> trans(parameters[0] + 4);
    Eigen::Map<const Eigen::Vector3d> Startpoint3d(parameters[1]);
    Eigen::Map<const Eigen::Vector3d> Endpoint3d(parameters[1]+3);

    Eigen::Vector3d Startpoint3dC;
    Eigen::Vector3d Endpoint3dC;
    Eigen::Vector2d Startpoint2d;
    Eigen::Vector2d Endpoint2d;
    Eigen::Vector2d err;

    // the MapLine in the camera coordinate
    Startpoint3dC = quaterd*Startpoint3d + trans;
    Endpoint3dC = quaterd*Endpoint3d + trans;

    Startpoint2d[0] = fx*Startpoint3dC[0]/Startpoint3dC[2] + cx;
    Startpoint2d[1] = fy*Startpoint3dC[1]/Startpoint3dC[2] + cy;

    Endpoint2d[0] = fx*Endpoint3dC[0]/Endpoint3dC[2] + cx;
    Endpoint2d[1] = fy*Endpoint3dC[1]/Endpoint3dC[2] + cy;

    err[0] = lineCoef[0]*Startpoint2d[0] + lineCoef[1]*Startpoint2d[1] + lineCoef[2];
    err[1] = lineCoef[0]*Endpoint2d[0] + lineCoef[1]*Endpoint2d[1] + lineCoef[2];

    err[0] = err[0]/err.norm();
    err[1] = err[1]/err.norm();

    residuals[0] = err[0];
    residuals[1] = err[1];

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> jacobianStart;
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> jacobianEnd;
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> jacobian;

    jacobianStart << fx/Startpoint3dC[2],  0, -fx*Startpoint3dC[0]/Startpoint3dC[2]/Startpoint3dC[2],
                      0, fy/Startpoint3dC[2], -fy*Startpoint3dC[1]/Startpoint3dC[2]/Startpoint3dC[2];

    jacobianEnd << fx/Endpoint3dC[2],  0, -fx*Endpoint3dC[0]/Endpoint3dC[2]/Endpoint3dC[2],
                    0, fy/Endpoint3dC[2], -fy*Endpoint3dC[1]/Endpoint3dC[2]/Endpoint3dC[2];

    if (jacobians != nullptr)
    {
        if(jacobians[0] != nullptr) // the jacobian for the MapPoint pose optimization
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > Jse3(jacobians[0]);
            Jse3.setZero();

            // very important! the form of the se3 is the rotation in the front and the transformation in the back
            jacobian <<  lineCoef[0]*jacobianStart(0, 0) + lineCoef[1]*jacobianStart(1, 0),
                         lineCoef[0]*jacobianStart(0, 1) + lineCoef[1]*jacobianStart(1, 1),
                         lineCoef[0]*jacobianStart(0, 2) + lineCoef[1]*jacobianStart(1, 2),
                         lineCoef[0]*jacobianEnd(0, 0) + lineCoef[1]*jacobianEnd(1, 0),
                         lineCoef[0]*jacobianEnd(0, 1) + lineCoef[1]*jacobianEnd(1, 1),
                         lineCoef[0]*jacobianEnd(0, 2) + lineCoef[1]*jacobianEnd(1, 2);

            Jse3.block<2,3>(0,0) = jacobian;
            Jse3.block<2,3>(0,3) = -jacobian*(Converter::skew(Startpoint3dC) + Converter::skew(Endpoint3dC));
        }
        if(jacobians[1] != nullptr) // the jacobian for the camera pose optimization
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > Jpoint(jacobians[1]);
            Jpoint = jacobian * quaterd.toRotationMatrix();
        }

    }

}


Eigen::Vector2d Optimizer::ReprojectionError(const ceres::Problem& problem, ceres::ResidualBlockId id)
{
    auto cost = problem.GetCostFunctionForResidualBlock(id);

    std::vector<double*> parameterBlocks;
    problem.GetParameterBlocksForResidualBlock(id, &parameterBlocks);

    Eigen::Vector2d residual;
    cost->Evaluate(parameterBlocks.data(), residual.data(), nullptr);

    return residual;
}

std::vector<double> Optimizer::GetReprojectionErrorNorms(const ceres::Problem& problem)
{
    std::vector<double> result;
    std::vector<ceres::ResidualBlockId> ids;

    problem.GetResidualBlocks(&ids);

    for (auto& id : ids)
    {
        result.push_back(ReprojectionError(problem, id).norm());
    }

    return result;
}

void Optimizer::RemoveOutliers(ceres::Problem& problem, double threshold)
{
    std::vector<ceres::ResidualBlockId> ids;
    problem.GetResidualBlocks(&ids);

    int count = 0;
    for (auto & id: ids)
    {
        if (ReprojectionError(problem, id).norm() > threshold)
        {
            problem.RemoveResidualBlock(id);
            count++;
        }
    }

    LOG_IF(ERROR, (count/ids.size()) > 0.5) << " too much outliers: " << (count/ids.size()) << endl;
}

void Optimizer::PoseOptimization(Frame *pFrame)
{
    Eigen::Matrix3d K;
    K = pFrame->mpCamera->GetCameraIntrinsic();

    cv::Mat extrinsic(7, 1, CV_64FC1);

    {
        extrinsic.ptr<double>()[0] = pFrame->Tcw.unit_quaternion().x();
        extrinsic.ptr<double>()[1] = pFrame->Tcw.unit_quaternion().y();
        extrinsic.ptr<double>()[2] = pFrame->Tcw.unit_quaternion().z();
        extrinsic.ptr<double>()[3] = pFrame->Tcw.unit_quaternion().w();
        extrinsic.ptr<double>()[4] = pFrame->Tcw.translation()[0];
        extrinsic.ptr<double>()[5] = pFrame->Tcw.translation()[1];
        extrinsic.ptr<double>()[6] = pFrame->Tcw.translation()[2];
    }

//    cout << "extrinsic: " <<extrinsic.t() << endl;

    cout << pFrame->Tcw << endl;

    ceres::Problem problem;

    problem.AddParameterBlock(extrinsic.ptr<double>(), 7, new PoseLocalParameterization());

    ceres::LossFunction* lossfunction = new ceres::HuberLoss(1);   // loss function make bundle adjustment robuster.

    for (int i = 0; i < pFrame->mvpMapPoint.size(); i++)
    {
        if (pFrame->mvpMapPoint[i]->mPosew.isZero())
            continue;

        Eigen::Vector2d observed = pFrame->mvpMapPoint[i]->mmpPointFeature2D[pFrame->GetFrameID()]->mpixel;

        ceres::CostFunction* costfunction = new ReprojectionErrorSE3(K(0, 0), K(1, 1),
                                                                     K(0, 2), K(1, 2),
                                                                     observed[0], observed[1]);

        problem.AddResidualBlock(
                costfunction, lossfunction,
                extrinsic.ptr<double>(), &pFrame->mvpMapPoint[i]->mPosew.x());

        problem.AddParameterBlock(&pFrame->mvpMapPoint[i]->mPosew.x(), 3);
    }

    RemoveOutliers(problem, 10);

//    vector<double> vresiduals;
//    vresiduals = GetReprojectionErrorNorms(problem);
//
//    for (auto residual : vresiduals)
//    {
//        cout << residual << endl;
//    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_solver_time_in_seconds = 0.1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

//    cout << "extrinsic: " <<extrinsic.t() << endl;

    {
        // it is very important that the scale data is the first position in the quaternion data type in the eigen
        // but if you use pointer to use the data, the scale date is the last position.
        pFrame->Tcw.setQuaternion(Eigen::Quaterniond(extrinsic.ptr<double>()[3], extrinsic.ptr<double>()[0],
                                                     extrinsic.ptr<double>()[1],extrinsic.ptr<double>()[2]));

        pFrame->Tcw.so3().unit_quaternion().norm();
        pFrame->Tcw.translation()[0] = extrinsic.ptr<double>()[4];
        pFrame->Tcw.translation()[1] = extrinsic.ptr<double>()[5];
        pFrame->Tcw.translation()[2] = extrinsic.ptr<double>()[6];
    }

    cout << pFrame->Tcw << endl;

    if (!summary.IsSolutionUsable())
    {
        cout << "Bundle Adjustment failed." << std::endl;
    }
    else
    {
        // Display statistics about the minimization
        cout << summary.BriefReport() << endl
             << " residuals number: " << summary.num_residuals << endl
             << " Initial RMSE: " << sqrt(summary.initial_cost / summary.num_residuals) << endl
             << " Final RMSE: " << sqrt(summary.final_cost / summary.num_residuals) << endl
             << " Time (s): " << summary.total_time_in_seconds << endl;
    }

//    vresiduals = GetReprojectionErrorNorms(problem);
//
//    for (auto residual : vresiduals)
//    {
//        cout << residual << endl;
//    }
}

} // namespace PL_VO