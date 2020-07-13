#pragma once

#include <Eigen/Core>

class Homography2DVisualServo {

    public:

        /**
         * @brief Implements 'Homography-based 2D Visual Servoing'
         *        https://ieeexplore.ieee.org/document/1642061
         * 
         * @param K intrinsic camera parameters (Eigen::Matrix3d)
        **/
        Homography2DVisualServo(Eigen::Matrix3d& K);

        /**
         * @brief eq. 15 and 16, see paper
         * 
         * @param G homography matrix, see eq. 13 (Eigen::Matrix3d)
         * @param m_star control point in normalized coordinates, see eq. 1 (Eigen::Vector3d)
        **/
        Eigen::VectorXd computeFeedback(Eigen::Matrix3d& G, Eigen::Vector3d& m_star);

        void K(Eigen::Matrix3d& K) { _K = std::move(K); };

    private:

        // Camera intrinsics, see eq. 3 paper
        Eigen::Matrix3d _K;

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
