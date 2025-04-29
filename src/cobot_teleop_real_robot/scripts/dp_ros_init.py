#!/usr/bin/env python3
import os
import pexpect
import subprocess
from tmux_utils import create_tmux_session,check_tmux_installed, check_and_kill_session, send_tmux_command, split_tmux_pane, adjust_panes_layout

def main():
    if not check_tmux_installed():
        return

    session_name = "dp_session"
    print("\n=== Starting DP Session Setup ===")

    # 检查并关闭已存在的会话
    check_and_kill_session(session_name)

    # 创建新的tmux会话
    if not create_tmux_session(session_name):
        print("Failed to create tmux session. Exiting...")
        return
    
    print("\n=== Running Commands ===")
    # 清理ROS日志
    send_tmux_command(session_name, "rosclean purge -y", pane_index=0)
    os.system("sleep 3")

    # 启动手部相机
    send_tmux_command(session_name, "roslaunch cobot_teleop_real_robot hand_camera.launch", pane_index=0)
    split_tmux_pane(session_name)
    os.system("sleep 3")

    # # 设置手部相机参数
    # send_tmux_command(session_name, "rosrun dynamic_reconfigure dynparam set /hand_cam/color/image_raw/compressed jpeg_quality 80", pane_index=0)
    # send_tmux_command(session_name, "rosrun dynamic_reconfigure dynparam set /hand_cam/color fps 15", pane_index=0)
    # os.system("sleep 1")

    # 启动头部相机
    send_tmux_command(session_name, "roslaunch cobot_teleop_real_robot main_camera.launch", pane_index=1)
    split_tmux_pane(session_name)
    os.system("sleep 3")

    # # 设置头部相机参数
    # send_tmux_command(session_name, "rosrun dynamic_reconfigure dynparam set /head_cam/color/image_raw/compressed jpeg_quality 80", pane_index=1)
    # send_tmux_command(session_name, "rosrun dynamic_reconfigure dynparam set /head_cam/color fps 15", pane_index=1)
    # os.system("sleep 1")

    # 启动机械臂
    send_tmux_command(session_name, "roslaunch rm_driver rm_75_driver.launch", pane_index=2)
    split_tmux_pane(session_name)
    os.system("sleep 3")

    # 启动cobot_controller_node
    send_tmux_command(session_name, "rosrun cobot_teleop_real_robot cobot_interpolation_controller_node.py", pane_index=4)
    split_tmux_pane(session_name)
    os.system("sleep 3")

    # cobot_rosbag_recorder_node
    send_tmux_command(session_name, "rosrun cobot_teleop_real_robot cobot_rosbag_recorder_node.py", pane_index=5)
    split_tmux_pane(session_name)
    os.system("sleep 3")    

    # 启动rviz
    send_tmux_command(session_name, "roslaunch cobot_teleop_real_robot visualize.launch", pane_index=6)
    split_tmux_pane(session_name)
    os.system("sleep 3")

    print("\n=== Setup Complete ===")
    print(f"To attach to the session, run: tmux attach -t {session_name}")

if __name__ == "__main__":
    main()