import os
import pexpect
import subprocess
import rospy
import json
from std_msgs.msg import String

def check_tmux_installed():
    """检查tmux是否安装"""
    try:
        subprocess.run(['tmux', '-V'], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: tmux is not installed. Please install it with: sudo apt-get install tmux")
        return False

def check_and_kill_session(session_name):
    """Check if a tmux session exists and if so, kill it"""
    try:
        existing_session = os.system(f"tmux has-session -t {session_name} 2>/dev/null")
        if existing_session == 0:  # session exists
            print(f"Killing existing session: {session_name}")
            os.system(f"tmux kill-session -t {session_name}")
            os.system("killall rviz 2>/dev/null")
    except Exception as e:
        print(f"Error checking/killing session: {e}")

def create_tmux_session(session_name):
    """创建新的tmux会话"""
    try:
        result = os.system(f"tmux new-session -d -s {session_name}")
        if result != 0:
            print(f"Error creating tmux session: {session_name}")
            return False
        print(f"Successfully created session: {session_name}")
        return True
    except Exception as e:
        print(f"Exception creating session: {e}")
        return False

def send_tmux_command(session_name, command, pane_index):
    """发送命令到tmux窗格"""
    try:
        result = os.system(f"tmux send-keys -t {session_name}:0.{pane_index} '{command}' C-m")
        if result != 0:
            print(f"Error sending command: {command} to pane {pane_index}")
        else:
            print(f"Successfully sent command: {command}")
    except Exception as e:
        print(f"Exception sending command: {e}")

def split_tmux_pane(session_name, adjust=True):
    """分割tmux窗格"""
    try:
        result = os.system(f"tmux split-window -t {session_name}")
        if result != 0:
            print("Error splitting pane")
        else:
            print("Successfully split pane")
        if adjust:
            adjust_panes_layout(session_name, "tiled")
    except Exception as e:
        print(f"Exception splitting pane: {e}")

def adjust_panes_layout(session_name, layout="tiled"):
    """调整窗格布局"""
    try:
        command = f"tmux select-layout -t {session_name} {layout}"
        result = os.system(command)
        if result != 0:
            print(f"Error adjusting layout to {layout}")
        else:
            print(f"Successfully adjusted layout to {layout}")
    except Exception as e:
        print(f"Exception adjusting layout: {e}")