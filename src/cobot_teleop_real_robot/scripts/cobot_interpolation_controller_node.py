#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import json
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from dh_gripper_msgs.msg import GripperState
import message_filters
import cv2
from cv_bridge import CvBridge
import os
import json
from datetime import datetime
from replay_buffer import ReplayBuffer
from rm_msgs.msg import MoveJ
import numpy as np
from geometry_msgs.msg import Pose
from collections import deque
from rm_msgs.msg import CartePos
from collections import deque
from queue import Empty
from pose_trajectory_interpolator import PoseTrajectoryInterpolator
import scipy.spatial.transform as st


class SafeDeque(deque):
    def safe_pop(self, default=None):
        try:
            return self.pop()
        except IndexError:
            return default
            
    def safe_popleft(self, default=None):
        try:
            return self.popleft()
        except IndexError:
            return default
    
    def get_all(self):
        """Get all elements and clear the deque"""
        items = list(self)
        # self.clear()
        return items


class CobotInterpolationControllerNode():
    def __init__(self, verbose=True):
        rospy.init_node('cobot_interpolation_controller_node', anonymous=True)
        self.bridge = CvBridge()
        self.home_joints = [-0.495929, -0.191444, -1.430638, 0.002077, -1.499967, 0.019300]
        self.command_queue = SafeDeque(maxlen=256)
        self.pose_msg_queue = SafeDeque(maxlen=256)
        self.command_sub = rospy.Subscriber('/cobot_interpolation_controller/command', String, self.command_callback, queue_size=10)
        self.pose_sub = rospy.Subscriber('/rm_driver/Pose_State', Pose, self.pose_callback, queue_size=10)
        self.pose_pub = rospy.Publisher('/rm_driver/MoveP_Fd_Cmd', CartePos, queue_size=10)
        self.movej_pub = rospy.Publisher('/rm_driver/MoveJ_Cmd', MoveJ, queue_size=10)

        self.max_pos_speed = 0.25 # 0.25
        self.max_rot_speed = 0.6 # 0.6
        self.verbose = verbose
        self.frequency = 50 # Hz
        self.dt = 1. / self.frequency
        rospy.loginfo("等待获取机械臂当前位姿...")
        while self.pose_msg_queue.safe_pop() is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("初始化完成")
    
    def pose_callback(self, pose_msg):
        self.pose_msg_queue.append(pose_msg)
    
    def joint_callback(self, joint_msg):
        self.joint_msg_queue.append(joint_msg)
    
    def command_callback(self, command_msg):
        self.command_queue.append(command_msg)
         
    def move_to_home(self):
        """Move the robot to home position and wait for completion"""
        rospy.loginfo("Moving to home position...")
        movej_msg = MoveJ()
        movej_msg.joint = self.home_joints
        movej_msg.speed = 0.3
        # Send move command
        self.movej_pub.publish(movej_msg)
        rospy.sleep(3.0)
    
    def pose_msg_to_numpy(self, pose_msg):
        pose_array =  np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w])
        new_pose_array = np.zeros(6)
        new_pose_array[:3] = pose_array[:3]
        new_pose_array[3:] = st.Rotation.from_quat(pose_array[3:]).as_rotvec()
        return new_pose_array

    def array_to_pose_msg(self, array):
        pose_rot_vec = st.Rotation.from_rotvec(array[3:])
        pose_rot_quat = pose_rot_vec.as_quat()
        pose_array = np.array([array[0], array[1], array[2], pose_rot_quat[0], pose_rot_quat[1], pose_rot_quat[2], pose_rot_quat[3]])
        pose = Pose()
        pose.position.x = pose_array[0]
        pose.position.y = pose_array[1]
        pose.position.z = pose_array[2]
        pose.orientation.x = pose_array[3]
        pose.orientation.y = pose_array[4]
        pose.orientation.z = pose_array[5]
        pose.orientation.w = pose_array[6]
        return pose
    
    def servoL(self, pose_command):
        target_pose_msg = self.array_to_pose_msg(pose_command)
        msg = CartePos()
        msg.Pose = target_pose_msg
        self.pose_pub.publish(msg)

    def run(self):

        # main loop
        pose_msg = self.pose_msg_queue.safe_pop()
        if pose_msg is not None:
            curr_pose = self.pose_msg_to_numpy(pose_msg)
        else:
            rospy.logerr("No pose message received, please launch the robot driver node first")
            return
        # use monotonic time to make sure the control loop never go backward
        curr_t = time.monotonic()
        last_waypoint_time = curr_t
        pose_interp = PoseTrajectoryInterpolator(
            times=[curr_t],
            poses=[curr_pose]
        )
        
        dt = 1. / self.frequency
        rate = rospy.Rate(self.frequency)
        iter_idx = 0
        while not rospy.is_shutdown():
            t_start = time.perf_counter()
            # send command to robot
            t_now = time.monotonic()
            pose_command = pose_interp(t_now)
            rospy.loginfo(f"pose_command: {pose_command}")
            self.servoL(pose_command)
            
            # fetch command from queue
            try:
                commands = self.command_queue.get_all()
                n_cmd = len(commands)
            except IndexError:
                n_cmd = 0
                rospy.logerr("[CobotInterpolationControllerNode.run] command_queue is empty!")
                continue
            # execute commands
            for i in range(n_cmd):
                command_dict = json.loads(commands[i].data)
                cmd = command_dict['cmd']
                if cmd == "STOP":   
                    # stop ROS node
                    rospy.signal_shutdown("Stopping ROS node")
                    # stop immediately, ignore later commands
                    break
                elif cmd == "SERVOL":
                    # since curr_pose always lag behind curr_target_pose
                    # if we start the next interpolation with curr_pose
                    # the command robot receive will have discontinouity 
                    # and cause jittery robot behavior.
                    target_pose = command_dict['target_pose']
                    duration = float(command_dict['duration'])
                    curr_time = t_now + dt
                    t_insert = curr_time + duration
                    pose_interp = pose_interp.drive_to_waypoint(
                        pose=target_pose,
                        time=t_insert,
                        curr_time=curr_time,
                        max_pos_speed=self.max_pos_speed,
                        max_rot_speed=self.max_rot_speed
                    )
                    last_waypoint_time = t_insert
                    if self.verbose:
                        rospy.loginfo("[Controller] New pose target:{} duration:{}s".format(
                            target_pose, duration))
                elif cmd == "SCHEDULE_WAYPOINT":
                    target_pose = command_dict['target_pose']
                    target_time = float(command_dict['target_time'])
                    # translate global time to monotonic time
                    target_time = time.monotonic() - time.time() + target_time
                    curr_time = t_now + dt
                    pose_interp = pose_interp.schedule_waypoint(
                        pose=target_pose,
                        time=target_time,
                        max_pos_speed=self.max_pos_speed,
                        max_rot_speed=self.max_rot_speed,
                        curr_time=curr_time,
                        last_waypoint_time=last_waypoint_time
                    )
                    last_waypoint_time = target_time
                else:
                    rospy.logerr("Invalid command: {}".format(cmd))
                    break
            rate.sleep()
            iter_idx += 1
            if self.verbose:
                rospy.loginfo(f"Actual frequency {1/(time.perf_counter() - t_start)}")


if __name__ == '__main__':
    node = CobotInterpolationControllerNode(verbose=False)
    node.run()