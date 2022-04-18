#pragma once

#include <Eigen/Core>
#include <Eigen/LU>

class HVs {

    public:

        /**
         * @brief Implements 'Homography-based 2D Visual Servoing'
         *        https://ieeexplore.ieee.org/document/1642061
         * 
         * @param K intrinsic camera parameters (Eigen::Matrix3d)
         * @param lambda_v velocity gain, see eq. 20 paper (Eigen::Vector3d)
         * @param lambda_w velocity gain, see eq. 20 paper (Eigen::Vector3d)
        **/
        HVs(Eigen::Matrix3d& K, Eigen::Vector3d& lambda_v, Eigen::Vector3d& lambda_w);
        HVs() = default;

        /**
         * @brief eq. 15 and 16, see paper
         * 
         * @param G homography matrix, see eq. 13 (Eigen::Matrix3d)
         * @param p_star control point in normalized coordinates, see eq. 2, defaults to principal point (Eigen::Vector3d)
        **/
        Eigen::VectorXd computeFeedback(Eigen::Matrix3d& G, Eigen::Vector3d& p_star);
        Eigen::VectorXd computeFeedback(Eigen::Matrix3d& G);

        Eigen::MatrixXd K() const { return _K; };  // value-oriented getter and setter
        void K(Eigen::Matrix3d& K) { _K = std::move(K); };

        Eigen::Vector3d lambdaV() const { return _lambda_v; };
        void lambdaV(Eigen::Vector3d& lambda_v) { _lambda_v = std::move(lambda_v); };

        Eigen::Vector3d lambdaW() const { return _lambda_w; };
        void lambdaW(Eigen::Vector3d& lambda_w) { _lambda_w = std::move(lambda_w); };

    private:

        // Camera intrinsics, see eq. 3 paper
        Eigen::Matrix3d _K;

        // Gains, see eq. 20 paper
        Eigen::Vector3d _lambda_v;
        Eigen::Vector3d _lambda_w;

        /**
         * @brief eq. 15, see paper
         * 
         * @param H euclidean Homography matrix, see eq. 11 (Eigen::Matrix3d)
         * @param m_star control point in normalized coordinates, see eq. 1 (Eigen::Vector3d)
        **/
        Eigen::Vector3d _computeEv(Eigen::Matrix3d& H, Eigen::Vector3d& m_star);

        /**
         * @brief eq. 16, see paper
         * 
         * @param H euclidean Homography matrix, see eq. 11 (Eigen::Matrix3d)
        **/
        Eigen::Vector3d _computeEw(Eigen::Matrix3d& H);
};
