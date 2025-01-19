#!/usr/bin/env python3
import rospy
import rosbag
import os
import message_filters
from sensor_msgs.msg import JointState, CompressedImage, Image
from geometry_msgs.msg import PoseStamped
from cobot_teleop_real_robot.msg import PoseStampedWithGripper
import subprocess
from replay_buffer import ReplayBuffer
from glob import glob
from natsort import natsorted
from datetime import datetime
import cv2
from cv_bridge import CvBridge
import scipy.spatial.transform as st
import numpy as np


class RosbagConverter:
    def __init__(self, depth_compressed=True):
        self.count = 0
        self.depth_compressed = depth_compressed
        
        # 获取参数
        self.bag_path = rospy.get_param('~bag_path', 
            '/home/ubuntu/cobot_push_cube/2025-01-16-jls/')
        self.all_bag_files = natsorted(glob(os.path.join(self.bag_path, '*.bag')))
        self.output_dir = rospy.get_param('~output_dir',
            '/home/ubuntu/cobot_push_cube/training_data')
        output_full_path = os.path.join(self.output_dir, f'push_cube_2025-01-16-jls')

        # 创建订阅者
        self.pose_sub = message_filters.Subscriber('/cobot/obs/pose', PoseStamped)
        if self.depth_compressed:
            self.hand_depth_sub = message_filters.Subscriber('/cobot/obs/hand_cam_depth/compressed', CompressedImage)
            self.main_depth_sub = message_filters.Subscriber('/cobot/obs/main_cam_depth/compressed', CompressedImage)
        else:
            self.hand_depth_sub = message_filters.Subscriber('/cobot/obs/hand_cam_depth', Image)
            self.main_depth_sub = message_filters.Subscriber('/cobot/obs/main_cam_depth', Image)
        self.joint_states_sub = message_filters.Subscriber('/cobot/obs/joint_states', JointState)
        self.main_cam_sub = message_filters.Subscriber('/cobot/obs/main_cam/compressed', CompressedImage)
        self.hand_cam_sub = message_filters.Subscriber('/cobot/obs/hand_cam/compressed', CompressedImage)
        self.action_sub = message_filters.Subscriber('/cobot/actions', PoseStampedWithGripper)

        # self.synced_main_cam_pub = rospy.Publisher('/cobot/synced_main_cam/compressed', CompressedImage, queue_size=100)
        # self.synced_hand_cam_pub = rospy.Publisher('/cobot/synced_hand_cam/compressed', CompressedImage, queue_size=100)

        self.all_synced_msgs = []
        self.bridge = CvBridge()

        # 创建同步器
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.pose_sub, self.hand_depth_sub, self.main_depth_sub, 
             self.joint_states_sub, self.main_cam_sub, self.hand_cam_sub, 
             self.action_sub],
            queue_size=100,
            slop=0.05
        )
        
        # 注册回调函数
        self.sync.registerCallback(self.callback)
        self.replay_buffer = ReplayBuffer.create_from_path(output_full_path, mode='a')
        self.episode_id = 0
        self.episode_msgs = {}
        # self.all_episode_data = {}
        self.episode_data = {
            "agent_view_image": [],
            "hand_view_image": [],
            "robot_eef_pose": [],
            "actions": [],
            "action_2d": []
        }

        self.all_bag_finished = False


    def callback(self, pose_msg, hand_depth_msg, main_depth_msg, 
                joint_states_msg, main_cam_msg, hand_cam_msg, action_msg):
        """
        同步消息的回调函数
        """
        self.count += 1
        # 在这里处理同步后的消息
        rospy.loginfo(f"Received synchronized messages: {self.count}")
        # --- Get Observation ---
        # Convert images
        convert_func = (self.bridge.compressed_imgmsg_to_cv2 
                                      if self.depth_compressed else self.bridge.imgmsg_to_cv2)
        main_cam_cv2 = self.bridge.compressed_imgmsg_to_cv2(main_cam_msg)
        hand_cam_cv2 = self.bridge.compressed_imgmsg_to_cv2(hand_cam_msg)
        main_depth_cv2 = convert_func(main_depth_msg)
        hand_depth_cv2 = convert_func(hand_depth_msg)

        # Process pose
        pose_array = np.zeros(6, dtype=np.float32)
        pose_msg = pose_msg
        pose_array[:3] = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
        pose_array[3:] = st.Rotation.from_quat([
            pose_msg.pose.orientation.x, pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z, pose_msg.pose.orientation.w
        ]).as_rotvec()
        # --- End of Get Observation ---
        # --- Get Action ---
        action_msg = action_msg
        actions_array = np.zeros(7, dtype=np.float32)
        actions_array[:3] = [action_msg.pose.position.x, action_msg.pose.position.y, action_msg.pose.position.z]
        actions_array[3:6] = st.Rotation.from_quat([
            action_msg.pose.orientation.x, action_msg.pose.orientation.y,
            action_msg.pose.orientation.z, action_msg.pose.orientation.w
        ]).as_rotvec()
        actions_array[6] = float(action_msg.gripper_state)
        action_2d = np.zeros(2, dtype=np.float32)
        action_2d[0] = actions_array[0]
        action_2d[1] = actions_array[1]
        # --- End of Get Action ---
        self.episode_data["agent_view_image"].append(main_cam_cv2)
        self.episode_data["hand_view_image"].append(hand_cam_cv2)
        self.episode_data["robot_eef_pose"].append(pose_array)
        self.episode_data["actions"].append(actions_array)
        self.episode_data["action_2d"].append(action_2d)
    def run(self):
        rospy.spin()
        
def main():
    rospy.set_param('/use_sim_time', True)
    
    rospy.init_node('rosbag_converter', anonymous=True)
    
    converter = RosbagConverter()
    
    # all_episode_ends = []
    # Add bag playback loop
    for idx, bag_file in enumerate(converter.all_bag_files):
        if idx <= converter.replay_buffer.n_episodes-1:
            rospy.loginfo(f"Skipping bag file: {bag_file}, as it has already been processed")
            continue
        rospy.loginfo(f"Playing bag file: {bag_file}")

        # Start rosbag play in a separate process
        play_command = f"rosbag play --clock {bag_file} -r 2.5 > /dev/null 2>&1"
        process = subprocess.Popen(play_command, shell=True)

        # Wait for the process to complete
        process.wait()
        
        rospy.loginfo(f"Finished processing bag file: {bag_file}")
        rospy.loginfo(f"Saving episode {converter.replay_buffer.n_episodes} to replay buffer")
        # to numpy
        converter.episode_data = {
            key: np.array(value) for key, value in converter.episode_data.items()
        }
        for key, value in converter.episode_data.items():
            rospy.loginfo(f"{key}: {value.shape}")
        # svae to replay buffer
        converter.replay_buffer.add_episode(converter.episode_data, compressors='disk')
        converter.episode_data ={
            "agent_view_image": [],
            "hand_view_image": [],
            "robot_eef_pose": [],
            "actions": [],
            "action_2d": []
        }
        import gc
        gc.collect()
    # self.all_bag_finished = True
    # for episode_id in self.all_episode_data.keys():
        # self.replay_buffer.add_episode(self.all_episode_data[episode_id], compressors='disk')
    rospy.loginfo("All bag files have been processed")
    converter.run()

if __name__ == '__main__':
    main()
