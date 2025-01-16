#!/usr/bin/env python3
import os
import datetime
import subprocess
import rospy
import rosbag
from cobot_teleop_real_robot.srv import RosbagRecord, RosbagRecordResponse

# Add a global dictionary to store processes
running_processes = {}

# Add global variable for throttle processes
throttle_processes = []

def start_throttle_nodes():
    global throttle_processes
    # Start throttle nodes
    throttle_commands = [
        ["rosrun", "topic_tools", "throttle", "messages", "/main_cam/color/image_raw/compressed", "10.0"],
        ["rosrun", "topic_tools", "throttle", "messages", "/main_cam/aligned_depth_to_color/image_raw/compressed", "10.0"],
        ["rosrun", "topic_tools", "throttle", "messages", "/hand_cam/color/image_raw/compressed", "10.0"],
        ["rosrun", "topic_tools", "throttle", "messages", "/hand_cam/aligned_depth_to_color/image_raw/compressed", "10.0"]
    ]

    for cmd in throttle_commands:
        process = subprocess.Popen(cmd)
        throttle_processes.append(process)
        rospy.sleep(1)

def start_rosbag_record(session_name="temp_session", bag_save_full_path=None):
    # Create directory path with task name and date
    if bag_save_full_path is None:
        base_dir = "/home/agilex/cobot_teleop_data"
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(base_dir, f"{current_date}", f"{session_name}")
        os.makedirs(save_dir, exist_ok=True)
        bag_save_full_path = os.path.join(save_dir, f"{session_name}.bag")

    print("\n=== Starting ROSBag Recording ===")
    
    # # Topics to record
    # throttled_topics = [
    #     "/main_cam/color/image_raw/compressed_throttle",
    #     "/main_cam/aligned_depth_to_color/image_raw/compressed_throttle", 
    #     "/hand_cam/color/image_raw/compressed_throttle",
    #     "/hand_cam/aligned_depth_to_color/image_raw/compressed_throttle",
    #     "/cobot/pose", # 30Hz
    #     "/gripper/states",
    #     # "/rm_driver/Pose_State", # raw pose, 50Hz
    #     "/joint_states",
    #     "/cobot/actions", # 10Hz
    # ]
    to_record_topics = [
        "/cobot/obs/main_cam/compressed",
        "/cobot/obs/main_cam_depth",
        "/cobot/obs/hand_cam/compressed",
        "/cobot/obs/hand_cam_depth",
        "/cobot/obs/pose",
        "/cobot/obs/joint_states",
        "/cobot/actions",
    ]
    # Use rosbag command line instead of Python interface
    cmd = ["rosbag", "record", "-O", bag_save_full_path] + to_record_topics
    process = subprocess.Popen(cmd)
    running_processes[session_name] = process

def stop_rosbag_record(session_name="temp_session"):
    os.system("killall rosbag")

def drop_rosbag_record(session_name="temp_session", bag_save_full_path=None):
    stop_rosbag_record(session_name)
    if bag_save_full_path is None:
        base_dir = "/home/agilex/cobot_teleop_data"
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(base_dir, f"{current_date}", f"{session_name}")
        bag_save_full_path = os.path.join(save_dir, f"{session_name}.bag")
    os.system(f"rm -rf {bag_save_full_path}.active")
    os.system(f"rm -rf {bag_save_full_path}")

def handle_rosbag_service(req):
    """ROS service handler for rosbag recording commands"""
    response = RosbagRecordResponse()
    print(f"Received command: {req.command} for session: {req.session_name}")
    try:
        if req.command.lower() == 'start':
            start_rosbag_record(req.session_name, req.bag_save_full_path)
            response.success = True
            response.message = "Started recording successfully"
        elif req.command.lower() == 'stop':
            stop_rosbag_record(req.session_name)
            response.success = True 
            response.message = "Stopped recording successfully"
        elif req.command.lower() == 'drop':
            drop_rosbag_record(req.session_name, req.bag_save_full_path)
            response.success = True
            response.message = "Dropped recording successfully"
        else:
            response.success = False
            response.message = f"Invalid command: {req.command}. Use 'start' or 'stop' or 'drop'"
    except Exception as e:
        response.success = False
        response.message = f"Error: {str(e)}"
    
    return response

def main():
    """Main function to initialize and run the ROS node"""
    rospy.init_node('cobot_rosbag_recorder_node')
    # Start throttle nodes when the node initializes
    # start_throttle_nodes()
    service = rospy.Service('cobot_rosbag_recorder', RosbagRecord, handle_rosbag_service)
    rospy.loginfo("Rosbag record service started. Waiting for commands...")
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
