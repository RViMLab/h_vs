#!/usr/bin/python3

import os
import sys
import tkinter
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
from enum import Enum
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from actionlib.simple_action_client import SimpleActionClient

from h_vs.srv import capture, captureRequest
from h_vs.msg import h_vsGoal, h_vsAction


class ControlMode(Enum):
    MANUAL = 0
    AUTOMATIC = 1


class ControlGalleryGUI():
    def __init__(self, master: tkinter.Tk, repeat=False):
        self._master = master
        self._repeat = repeat

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
        self.repeat(self._repeat)
        self._twist = Twist()

        # ROS bindings, services: Capture, execute
        self._twist_pub = rospy.Publisher('visual_servo/twist', Twist, queue_size=1)
        self._cv_bridge = CvBridge()
        self._cap_client = rospy.ServiceProxy('visual_servo/capture', capture)
        self._execute_client = SimpleActionClient('visual_servo/execute', h_vsAction)

        self._cap_client.wait_for_service()

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
            self.repeat(self._repeat)
            self._control_mode = ControlMode.MANUAL
        else:
            raise ValueError('Unknown control mode.')

    def _capture_image(self):
        # send empty request
        req = captureRequest()
        res = self._cap_client(req)
        if not res.success.data:
            return

        img = self._cv_bridge.imgmsg_to_cv2(res.capture)
        id = res.id.data

        self._img_df = self._img_df.append({'img': img, 'id': id}, ignore_index=True)
        self._switch_image(id)

    def _execute_control(self):
        goal = h_vsGoal()
        goal.id.data = self._current_id
        self._execute_client.send_goal(goal)


if __name__ == '__main__':
    rospy.init_node('control_gallery_node')

    repeat = rospy.get_param('control_gallery_node/repeat')

    root = tkinter.Tk()
    gui = ControlGalleryGUI(root, repeat)
    root.mainloop()
    gui.repeat(True)  # clear sys
