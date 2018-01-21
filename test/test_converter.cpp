//
// Created by rain on 17-12-27.
//

#include <iostream>

#include <eigen3/Eigen/Dense>
#include <Converter.h>
#include <sophus/se3.h>
#include <sophus/so3.h>


using namespace std;

int main(int argv, char* argc[])
{
    Eigen::Quaterniond q1(0.2, 0.3, 0.4 ,0.5);
    Eigen::Quaterniond q2(0.3, 0.5, 0.6 ,0.7);

    q1.normalize();
    q2.normalize();

    cout << q1.coeffs().transpose() << endl;
    cout << q2.coeffs().transpose() << endl;

    cout << "quaternion left (quaternion1 product quaternion2)  " << endl;
    cout << PL_VO::Converter::quatLeftproduct(q1*q2) << endl;
    cout << "(quaternion left quaternion1) product (quaternion left quaternion2) "<< endl;
    cout << PL_VO::Converter::quatLeftproduct(q1)*PL_VO::Converter::quatLeftproduct(q2) << endl;

    cout << "quaternion right product quaternion left" << endl;
    cout << PL_VO::Converter::quatLeftproduct(q1)*PL_VO::Converter::quatRightproduct(q2) << endl;
    cout << "quaternion left product quaternion right" << endl;
    cout << PL_VO::Converter::quatRightproduct(q2)*PL_VO::Converter::quatLeftproduct(q1)<< endl;

    Sophus::SO3 so31(-0.0207925, 0.0495638, 0.0489707);
    Sophus::SO3 so32(-0.0772475, -0.0758137, -3.11846);

    cout << PL_VO::Converter::toEuler(so31.unit_quaternion()).transpose() << endl;
    cout << PL_VO::Converter::toEuler(so32.unit_quaternion()).transpose() << endl;


    Sophus::SE3 Tcw(Eigen::Quaterniond(0.99942977681163814, -0.0028868419120917322, -0.016111434490270738, -0.029533185481412594),
                    Eigen::Vector3d(-0.012884787233118483, -0.0069446675501963423, 0.028263296041360034));

    Eigen::Vector3d Point(0.5257240294266079, -0.034655717506164807, 1.1037999999999999);

    cout << "Tcw*Point3d: "  << endl;
    cout << (Tcw.inverse()*Point).transpose() << endl;
    cout << (Tcw.rotation_matrix()*Point + Tcw.translation()) << endl;
}