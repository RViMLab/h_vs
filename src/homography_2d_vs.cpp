#include <h_vs/homography_2d_vs.h>


Homography2DVisualServo::Homography2DVisualServo(Eigen::Matrix3d& K, Eigen::Vector3d& lambda_v, Eigen::Vector3d& lambda_w)
    : _K(K), _lambda_v(lambda_v), _lambda_w(lambda_w) {   };


// eq. 15 and 16, see paper
Eigen::VectorXd Homography2DVisualServo::computeFeedback(Eigen::Matrix3d& G, Eigen::Vector3d& p_star) {

    Eigen::Matrix3d H = _K.inverse()*G*_K;
    Eigen::Vector3d m_star = _K.inverse()*p_star;

    Eigen::VectorXd dtwist(6);
    dtwist << _lambda_v.asDiagonal()*(H, m_star), _lambda_w.asDiagonal()*_computeEw(H);

    return dtwist;
}


// eq. 15 and 16, see paper
Eigen::VectorXd Homography2DVisualServo::computeFeedback(Eigen::Matrix3d& G) {

    Eigen::Vector3d p_star(_K(0,2), _K(1,2), 1.);  // principal point per default

    Eigen::Matrix3d H = _K.inverse()*G*_K;
    Eigen::Vector3d m_star = _K.inverse()*p_star;

    Eigen::VectorXd dtwist(6);
    dtwist << _lambda_v.asDiagonal()*(H, m_star), _lambda_w.asDiagonal()*_computeEw(H);

    return dtwist;
}



// eq. 15, see paper
Eigen::Vector3d Homography2DVisualServo::_computeEv(Eigen::Matrix3d& H, Eigen::Vector3d& m_star) {
    return (H - Eigen::Matrix3d::Identity())*m_star;
}


// eq. 16, see paper
Eigen::Vector3d Homography2DVisualServo::_computeEw(Eigen::Matrix3d& H) {

    auto H_skew = H - H.transpose();

    Eigen::Vector3d ew(
        H_skew(2, 1),
        H_skew(0, 2),
        H_skew(1, 0)
    );

    return ew;
}
