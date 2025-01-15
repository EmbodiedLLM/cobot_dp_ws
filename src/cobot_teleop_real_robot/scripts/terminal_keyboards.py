#!/usr/bin/env python

import os
import json
import time
import datetime
import rospy
import numpy as np
import sys
import tty
import termios
import click
import scipy.spatial.transform as st
from cobot_teleop_real_robot.srv import RosbagRecord
from cobot_real_env import CobotEnv

def getKey():
    """获取键盘按键"""
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def call_rosbag_service(command, session_name, bag_path=''):
    """调用rosbag录制服务"""
    try:
        rospy.wait_for_service('/cobot_rosbag_recorder')
        service = rospy.ServiceProxy('/cobot_rosbag_recorder', RosbagRecord)
        response = service(command=command, 
                         session_name=session_name,
                         bag_save_full_path=bag_path)
        rospy.loginfo(f"Service call {command} successful")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('terminal_keyboard_control')
    
    # Initialize environment
    env = CobotEnv(rosbag_save_root_dir="/home/agilex/cobot_teleop_data")
    
    try:
        while not rospy.is_shutdown():
            key = getKey()
            
            if key == '\x03':  # ctrl-c
                break
            
            if key == 'e': #紧急停止
                # 停止所有ROS Node
                os.system("rosnode kill -a")
                env.drop_episode()
                break
            elif key.lower() == 'c':  # Start recording
                if not env.is_recording:
                    rospy.loginfo("Starting recording...")
                    current_time = time.time()
                    env.start_episode(
                        episode_start_time=current_time,
                        task_name="keyboard_teleop"
                    )
                else:
                    rospy.loginfo("Already recording!")
                
            elif key.lower() == 's':  # Stop recording
                if env.is_recording:
                    rospy.loginfo("Stopping recording...")
                    env.end_episode()
                else:
                    rospy.loginfo("Not recording!")

            elif key == '\x7f':  # backspace
                if env.is_recording:
                    # Delete the most recent recorded episode
                    if click.confirm('Are you sure to drop an episode?'):
                        env.drop_epiwsode()
                else:
                    rospy.loginfo("Not recording!")
                
    except Exception as e:
        rospy.logerr(e)
        
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
