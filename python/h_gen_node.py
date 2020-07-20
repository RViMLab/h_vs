#!/usr/bin/python3

import rospy
import cv2
import torch
import numpy as np
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from rospy.numpy_msg import numpy_msg

import homography_generators.calibration_pattern_homography_generator as cphg

# write node that 
#   subscribes camera images, possibly holds
#   read cam params
#   undistort img
#   computes projective homography

# TODO: https://gitter.im/RoboStack/Lobby, try to add a conda env once more ros packages integrated

if __name__ == '__main__':
    print('hello world!')

    # ones = torch.ones(2).cuda()

    # print(ones)

    # hg = cphg.CalibrationPatternHomographyGenerator(1, np.array([1.]), 10.)
    # print(hg._K)

    rospy.init_node("test")
    pub = rospy.Publisher("visual_servo/G", numpy_msg(Float64MultiArray), queue_size=1)

    while not rospy.is_shutdown():
        
        G = np.array([
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ], np.float64)


        layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(label='rows', size=G.shape[0]),
                MultiArrayDimension(label='cols', size=G.shape[1])
            ],
            data_offset=0
        )


        msg = Float64MultiArray(layout=layout, data=G.flatten().tolist())

        pub.publish(msg)
        rospy.sleep(1.)
