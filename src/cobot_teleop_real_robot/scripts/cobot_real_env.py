import os
import json
import time
import datetime
import gymnasium as gym
from gymnasium import spaces
import rospy
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, Image
from rm_msgs.msg import CartePos
from geometry_msgs.msg import PoseStamped, Pose
from cobot_teleop_real_robot.msg import PoseStampedWithGripper
from dh_gripper_msgs.msg import GripperCtrl, GripperState
from sensor_msgs.msg import Image, JointState
import message_filters
from cv_bridge import CvBridge
from rm_msgs.msg import MoveJ
from collections import deque
from typing import Optional
import scipy.spatial.transform as st
from cobot_teleop_real_robot.srv import RosbagRecord


class CobotEnv(gym.Env):
    def __init__(self, node_name=None, rosbag_save_root_dir=None):
        super(CobotEnv, self).__init__()
        
        # Define topic names
        self.TOPICS = {
            'joint_states': '/joint_states',
            'main_depth': '/main_cam/aligned_depth_to_color/image_raw',
            'hand_depth': '/hand_cam/aligned_depth_to_color/image_raw',
            'main_color': '/main_cam/color/image_raw/compressed',
            'hand_color': '/hand_cam/color/image_raw/compressed',
            'gripper_states': '/gripper/states'
        }

        self.gripper_state = None # 1 or 0 -> Open or Close
        self.gripper_real_state = None
        # Initialize state variables
        self.current_pose = np.zeros(6, dtype=np.float32)
        self.pose_msg_queue = deque(maxlen=256)
        self.all_sync_msg_queue = deque(maxlen=256)
        self.current_observation = None
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            dtype=np.float32
        )
        
        # Define observation space (now includes images)
        self.observation_space = spaces.Dict({
            'timestamp': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'joint_positions': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            'main_color': spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            'main_depth': spaces.Box(low=0, high=65535, shape=(480, 640), dtype=np.uint16),
            'hand_color': spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            'hand_depth': spaces.Box(low=0, high=65535, shape=(480, 640), dtype=np.uint16)
        })
        
        # Initialize ROS node only if not already initialized
        if node_name and not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)
            
        # Initialize ROS communications
        self.pose_pub = rospy.Publisher('/rm_driver/MoveP_Fd_Cmd', CartePos, queue_size=10)
        self.gripper_pub = rospy.Publisher('/gripper/ctrl', GripperCtrl, queue_size=10)
        self.pose_sub = rospy.Subscriber('/rm_driver/Pose_State', Pose, self.pose_callback)
        self.gripper_sub = rospy.Subscriber('/gripper/states', GripperState, self.gripper_callback)
        self.movej_pub = rospy.Publisher('/rm_driver/MoveJ_Cmd', MoveJ, queue_size=10)
        self.command_publisher = rospy.Publisher('/cobot_interpolation_controller/command', String, queue_size=10)

        self.cobot_main_cam_pub = rospy.Publisher('/cobot/obs/main_cam/compressed', CompressedImage, queue_size=10)
        self.cobot_main_cam_depth_pub = rospy.Publisher('/cobot/obs/main_cam_depth', Image, queue_size=10)
        self.cobot_hand_cam_pub = rospy.Publisher('/cobot/obs/hand_cam/compressed', CompressedImage, queue_size=10)
        self.cobot_hand_cam_depth_pub = rospy.Publisher('/cobot/obs/hand_cam_depth', Image, queue_size=10)
        self.cobot_pose_pub = rospy.Publisher('/cobot/obs/pose', PoseStamped, queue_size=10)
        self.cobot_joint_pub = rospy.Publisher('/cobot/obs/joint_states', JointState, queue_size=10)    

        self.cobot_action_pub = rospy.Publisher('/cobot/actions', PoseStampedWithGripper, queue_size=10)
        
        # Initialize ROS subscribers with message filters
        self.bridge = CvBridge()
        self.joint_sub = message_filters.Subscriber(self.TOPICS['joint_states'], JointState)
        self.main_depth_sub = message_filters.Subscriber(self.TOPICS['main_depth'], Image)
        self.hand_depth_sub = message_filters.Subscriber(self.TOPICS['hand_depth'], Image)
        self.main_color_sub = message_filters.Subscriber(self.TOPICS['main_color'], CompressedImage)
        self.hand_color_sub = message_filters.Subscriber(self.TOPICS['hand_color'], CompressedImage)
        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.main_depth_sub, self.hand_depth_sub,
             self.main_color_sub, self.hand_color_sub,
             self.joint_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        # Add home position definition
        self.home_joints = [-0.495929, -0.191444, -1.430638, 0.002077, -1.499967, 0.019300]
        if rosbag_save_root_dir is None:
            self.rosbag_save_root_dir = "/home/agilex/cobot_teleop_data"
        else:
            self.rosbag_save_root_dir = rosbag_save_root_dir
        self.episode_start_time = time.time()
        self.episode_id = -1
        self.episode_count = 0
        self.episode_rosbag_name = None
        self.episode_rosbag_path = None
        self.is_recording = False
        rospy.loginfo("Waiting for rosbag recorder service...")
        rospy.wait_for_service('cobot_rosbag_recorder')
        rospy.loginfo("Rosbag recorder service found!")
        self.rosbag_service = rospy.ServiceProxy('cobot_rosbag_recorder', RosbagRecord)
        self.max_pos_speed = 0.1
        self.max_rot_speed = 0.6

    
    def start_episode(self, episode_start_time, task_name):
        self.episode_id+=1
        self.episode_count+=1
        self.episode_start_time = episode_start_time
        self.episode_rosbag_name = f"{task_name}_episode_{self.episode_id}_{self.episode_start_time}.bag"
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.episode_rosbag_path = os.path.join(self.rosbag_save_root_dir, current_date, self.episode_rosbag_name)
        os.makedirs(os.path.dirname(self.episode_rosbag_path), exist_ok=True)
        self.start_rosbag_record(session_name=task_name, bag_save_full_path=self.episode_rosbag_path)
        self.is_recording = True
    
    def end_episode(self):
        self.stop_rosbag_record()
        self.is_recording = False
    
    def drop_episode(self):
        if self.is_recording:
            self.episode_id-=1
            self.episode_count-=1
            self.drop_rosbag_record()
            self.is_recording = False
        else:
            rospy.logerr("[CobotEnv.drop_episode] Not recording!")
        
    def pose_callback(self, msg):
        self.pose_msg_queue.append(msg)
    
    def gripper_callback(self, msg):
        self.gripper_real_state = msg.position > 0.0
        if self.gripper_state is None:
            self.gripper_state = self.gripper_real_state
    
    def toggle_gripper(self):
        self.gripper_state = not self.gripper_state
        self._control_gripper(self.gripper_state)
    
    def gripper_open(self):
        self.gripper_state = True
        self._control_gripper(self.gripper_state)
    
    def gripper_close(self):
        self.gripper_state = False
        self._control_gripper(self.gripper_state)
   
    def array_to_pose_msg(self, array):
        pose = Pose()
        pose.position.x = array[0]
        pose.position.y = array[1]
        pose.position.z = array[2]
        pose.orientation.x = array[3]
        pose.orientation.y = array[4]
        pose.orientation.z = array[5]
        pose.orientation.w = array[6]
        return pose

    def update_current_pose(self):
        try:
            self.pose_msg = self.pose_msg_queue.pop()
        except IndexError:
            self.pose_msg = None
            rospy.logerr("[CobotEnv.update_current_pose] pose_msg_queue is empty!")
            return
        pose_array = np.array([self.pose_msg.position.x, self.pose_msg.position.y, self.pose_msg.position.z, 
                               self.pose_msg.orientation.x, self.pose_msg.orientation.y, self.pose_msg.orientation.z, 
                               self.pose_msg.orientation.w])
        pose_rot_vec = st.Rotation.from_quat(pose_array[3:]).as_rotvec()
        self.current_pose = np.concatenate([pose_array[:3], pose_rot_vec])


    def sync_callback(self, main_depth_msg, hand_depth_msg, 
                     main_color_msg, hand_color_msg, joint_msg):
        """Process synchronized camera images and joint states"""
        try:
            self.all_sync_msg_queue.append((main_depth_msg, hand_depth_msg, 
                     main_color_msg, hand_color_msg, joint_msg))
            timestamp = joint_msg.header.stamp
            self._publish_pose_for_record(timestamp=timestamp)
            self._publish_obs_for_record(timestamp=timestamp,
                                         main_color_msg=main_color_msg, hand_color_msg=hand_color_msg, 
                                         main_depth_msg=main_depth_msg, hand_depth_msg=hand_depth_msg, 
                                         joint_msg=joint_msg)
        except Exception as e:
            rospy.logerr(f"Error in sync_callback: {str(e)}")

    def reset(self):
        """Reset the environment and return initial observation"""
        # Move to home position first
        self.move_to_home()
        return self.get_observation()
    
    def _publish_obs_for_record(self, timestamp ,main_color_msg, hand_color_msg, main_depth_msg, hand_depth_msg, joint_msg):
        main_color_msg.header.stamp = timestamp
        main_depth_msg.header.stamp = timestamp
        hand_color_msg.header.stamp = timestamp
        hand_depth_msg.header.stamp = timestamp
        joint_msg.header.stamp = timestamp
        self.cobot_main_cam_pub.publish(main_color_msg)
        self.cobot_main_cam_depth_pub.publish(main_depth_msg)
        self.cobot_hand_cam_pub.publish(hand_color_msg)
        self.cobot_hand_cam_depth_pub.publish(hand_depth_msg)
        self.cobot_joint_pub.publish(joint_msg)
    
    def get_observation(self):
        """Return the current observation"""
        if len(self.all_sync_msg_queue) == 0:
            return None
        try:
            latest_sync_msg = self.all_sync_msg_queue.pop()
        except IndexError:
            latest_sync_msg = None
            rospy.logerr("[CobotEnv.get_observation] all_sync_msg_queue is empty!")
            return None
        main_depth_msg, hand_depth_msg, main_color_msg, hand_color_msg, joint_msg = latest_sync_msg
        self.update_current_pose()
        self.current_observation = {
            'timestamp': np.array([joint_msg.header.stamp.to_sec()], dtype=np.float32),
            'joint_positions': np.array(list(joint_msg.position), dtype=np.float32),
            'main_color': self.bridge.compressed_imgmsg_to_cv2(main_color_msg, "bgr8"),
            'main_depth': self.bridge.compressed_imgmsg_to_cv2(main_depth_msg, "passthrough"), 
            'hand_color': self.bridge.compressed_imgmsg_to_cv2(hand_color_msg, "bgr8"),
            'hand_depth': self.bridge.compressed_imgmsg_to_cv2(hand_depth_msg, "passthrough"),
            'eef_pose': self.current_pose,
        }
        return self.current_observation
    

    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            stage: int = 0
        ):
        # assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        # schedule waypoints
        for i in range(len(new_actions)):
            message = {
                'cmd': "SCHEDULE_WAYPOINT",
                'target_pose': new_actions[i].tolist(),
                'target_time': new_timestamps[i],
            }
            msg = String()
            msg.data = json.dumps(message)
            self.command_publisher.publish(msg)
            self._publish_action_for_record(new_actions[i], stage)
    
    def _publish_action_for_record(self, new_pose, stage):
        pose_msg = PoseStampedWithGripper()
        new_pose_rot = st.Rotation.from_rotvec(new_pose[3:]).as_quat().tolist()
        new_pose_array = np.concatenate([new_pose[:3], new_pose_rot])
        pose_msg.pose.position.x = new_pose_array[0]
        pose_msg.pose.position.y = new_pose_array[1]
        pose_msg.pose.position.z = new_pose_array[2]
        pose_msg.pose.orientation.x = new_pose_array[3]
        pose_msg.pose.orientation.y = new_pose_array[4]
        pose_msg.pose.orientation.z = new_pose_array[5]
        pose_msg.pose.orientation.w = new_pose_array[6]
        pose_msg.gripper_state = self.gripper_state
        pose_msg.stage = stage
        current_timestamp = rospy.Time.now()
        pose_msg.header.stamp = current_timestamp
        self.cobot_action_pub.publish(pose_msg)

    def _publish_pose_for_record(self, timestamp):
        current_timestamp = timestamp
        current_pose_msg = self.pose_msg_queue.pop()
        new_current_pose_msg = PoseStamped()
        new_current_pose_msg.pose.position.x = current_pose_msg.position.x
        new_current_pose_msg.pose.position.y = current_pose_msg.position.y
        new_current_pose_msg.pose.position.z = current_pose_msg.position.z
        new_current_pose_msg.pose.orientation.x = current_pose_msg.orientation.x
        new_current_pose_msg.pose.orientation.y = current_pose_msg.orientation.y
        new_current_pose_msg.pose.orientation.z = current_pose_msg.orientation.z
        new_current_pose_msg.pose.orientation.w = current_pose_msg.orientation.w
        new_current_pose_msg.header.stamp = current_timestamp
        self.cobot_pose_pub.publish(new_current_pose_msg)

    def _control_gripper(self, open_gripper):
        """Control gripper state"""
        msg = GripperCtrl()
        msg.initialize = False
        msg.speed = 0.0
        if open_gripper:
            msg.position = 1000.0
            msg.force = 0.0
            self.gripper_state = True
        else:
            msg.position = 0.0
            msg.force = 100.0
            self.gripper_state = False
        self.gripper_pub.publish(msg)

    def _calculate_reward(self):
        """Calculate reward based on task completion"""
        # Implement your reward function here
        return 0.0

    def move_to_home(self):
        """Move the robot to home position and wait for completion"""
        rospy.loginfo("Moving to home position...")
        movej_msg = MoveJ()
        movej_msg.joint = self.home_joints
        movej_msg.speed = 0.5
        # Send move command
        self.movej_pub.publish(movej_msg)
        rospy.sleep(3.0)
        rospy.loginfo("Home position reached")

    def start_rosbag_record(self, session_name, bag_save_full_path):
        """Use ROS service to start rosbag recording"""
        try:
            self.rosbag_service('start', session_name, bag_save_full_path)
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def stop_rosbag_record(self):
        """Use ROS service to stop rosbag recording"""
        try:
            self.rosbag_service('stop', self.episode_rosbag_name, '')
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
        
    
    def drop_rosbag_record(self):
        if self.is_recording:
            self.rosbag_service('drop', self.episode_rosbag_name, self.episode_rosbag_path)
            rospy.loginfo("Dropped rosbag record!")
        else:
            rospy.logerr("Not recording!")