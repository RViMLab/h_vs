#!/usr/bin/python3

from PIL import Image, ImageTk
import tkinter
from tkinter import messagebox
import numpy as np
import pandas as pd
from enum import Enum
import os
from cv_bridge import CvBridge
import sys
from actionlib.simple_action_client import SimpleActionClient

sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')


import rospy
from std_srvs.srv import Empty, EmptyRequest
from geometry_msgs.msg import Twist

from h_vs.srv import capture, captureRequest
from h_vs.msg import h_vsActionGoal, h_vsAction
# from sensor_msgs.msg import Image

# Switch Control Service
#   - on button, change keyboard to automatic control and vice versa
#   - h_vs: G -> ...
#   - h_vs: twist -> ... either of these publishers, client?

# Capture Image Service
#   - on button, send capture to server
#   - add immage and id to buffer, return to this

# Remove Image Service
#   - on button, remove image for client and server

# Execute Service
#   - on button, send id to servo


class ControlMode(Enum):
    MANUAL = 0
    AUTOMATIC = 1


class ControlGalleryGUI():
    def __init__(self, master: tkinter.Tk):
        self._master = master

        # GUI
        self._label = tkinter.Label(self._master, compound=tkinter.TOP)
        self._label.pack()

        self._frame = tkinter.Frame(self._master)
        self._frame.focus_set()
        self._frame.pack()

        self._master.bind("<KeyPress>", self._keydown)
        self._master.bind("<KeyRelease>", self._keyup) 

        tkinter.Button(self._frame, text='Previous', command=lambda: self._next_image(-1)).pack(side=tkinter.LEFT)
        tkinter.Button(self._frame, text='Next', command=lambda: self._next_image(+1)).pack(side=tkinter.LEFT)
        tkinter.Button(self._frame, text='Switch', command=lambda: self._switch_control()).pack(side=tkinter.LEFT)
        tkinter.Button(self._frame, text='Execute', command=lambda: self._execute_control()).pack(side=tkinter.LEFT)
        tkinter.Button(self._frame, text='Capture', command=lambda: self._capture_image()).pack(side=tkinter.LEFT)
        tkinter.Button(self._frame, text='Quit', command=self._master.quit).pack(side=tkinter.LEFT)

        # Member
        self._img_df = pd.DataFrame(columns=['img', 'id'])
        self._current_id = -1
        self._control_mode = ControlMode(ControlMode.MANUAL)
        self.repeat(False)
        self._twist = Twist()

        # ROS bindings, services: Capture, execute
        self._twist_pub = rospy.Publisher('visual_servo/twist', Twist, queue_size=1)
        self._cv_bridge = CvBridge()
        self._cap_client = rospy.ServiceProxy('visual_servo/capture', capture)


        # self._cap_client.wait_for_service()
        # self._execute_client.wait_for_service()

    def repeat(self, r: bool=True):
        if r:
            os.system('xset r on') # https://stackoverflow.com/questions/27215326/tkinter-keypress-keyrelease-events
        else:
            os.system('xset r off')

    def _keydown(self, event: tkinter.Event):
        if self._control_mode is not ControlMode.MANUAL:
            messagebox.showinfo('Control', 'Switch to manual control first.')
            return

        if event.keysym == 'a':
            self._twist.linear.x  = -0.1
        if event.keysym == 'd':
            self._twist.linear.x  =  0.1
        if event.keysym == 'w':
            self._twist.linear.y  = -0.1
        if event.keysym == 's':
            self._twist.linear.y  =  0.1
        if event.keysym == 'Left':
            self._twist.angular.z = -0.1
        if event.keysym == 'Right':
            self._twist.angular.z =  0.1
        if event.keysym == 'Up':
            self._twist.linear.z  =  0.1
        if event.keysym == 'Down':
            self._twist.linear.z  = -0.1

        self._twist_pub.publish(self._twist)

    def _keyup(self, event: tkinter.Event):
        if event.keysym == 'a' or event.keysym == 'd':
            self._twist.linear.x  = 0.0
        if event.keysym == 'w' or event.keysym == 's':
            self._twist.linear.y  = 0.0
        if event.keysym == 'Left' or event.keysym == 'Right':
            self._twist.angular.z = 0.0
        if event.keysym == 'Up' or event.keysym == 'Down':
            self._twist.linear.z  = 0.0

        self._twist_pub.publish(self._twist)

    def _next_image(self, delta: int):
        if not (0 <= self._current_id + delta < len(self._img_df)):
            messagebox.showinfo('End', 'No more images.')
            return
        self._current_id += delta

        img = self._img_df.iloc[self._current_id].img
        img = Image.fromarray(img)

        img = ImageTk.PhotoImage(img)
        self._label['image'] = img
        self._label['text'] = 'Image {}/{}'.format(self._current_id+1, len(self._img_df))
        self._label.photo = img

    def _switch_image(self, idx: int):
        img = self._img_df.iloc[idx].img
        img = Image.fromarray(img)

        self._current_id = idx

        img = ImageTk.PhotoImage(img)
        self._label['image'] = img
        self._label['text'] = 'Image {}/{}'.format(self._current_id+1, len(self._img_df))
        self._label.photo = img

    def _switch_control(self):
        if self._control_mode is ControlMode.MANUAL:
            self.repeat(True)
            self._control_mode = ControlMode.AUTOMATIC
        elif self._control_mode is ControlMode.AUTOMATIC:
            self.repeat(False)
            self._control_mode = ControlMode.MANUAL
        else:
            raise ValueError('Unknown control mode.')

    def _capture_image(self):
        # send empty request
        req = captureRequest()
        res = self._cap_client(req)

        img = self._cv_bridge.imgmsg_to_cv2(res.capture)
        id = res.id.data

        self._img_df = self._img_df.append({'img': img, 'id': id}, ignore_index=True)
        self._switch_image(id)

    # def _remove_image(self):
    #     if self._current_id == -1:
    #         messagebox.showinfo('Empty', 'No further images to remove.')
    #         return

    #     self._img_df = self._img_df.drop(self._current_id)
    #     self._current_id -= 1

    #     if self._current_id == -1:
    #         self._label.config(image='', text='Image 0/0')
    #     else:
    #         self._switch_image(self._current_id)

    def _execute_control(self):
        pass


if __name__ == '__main__':
    rospy.init_node('control_gallery_node')

    root = tkinter.Tk()
    gui = ControlGalleryGUI(root)
    root.mainloop()
    gui.repeat(True)  # clear sys
