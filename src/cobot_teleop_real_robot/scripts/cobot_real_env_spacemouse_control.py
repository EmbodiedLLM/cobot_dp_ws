#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import rospy
import scipy.spatial.transform as st
from spacemouse_shared_memory import Spacemouse
from multiprocessing.managers import SharedMemoryManager
import time
import click
import sys
from cobot_real_env import CobotEnv
from keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
import uuid
random_id_str = str(uuid.uuid4())[:8]

def main():
    # Initialize ROS node first
    rospy.init_node('cobot_spacemouse_control', anonymous=True)
    rospy.loginfo("正在初始化...")
    # Initialize the CobotEnv without node initialization
    env = CobotEnv(node_name="cobot_env")
    task_name = f"cobot_push_cube_{random_id_str}"
    # 等待环境初始化完成
    rospy.loginfo("等待获取机械臂当前位姿...")
    while len(env.pose_msg_queue) == 0 and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("初始化完成!")
    t_start = time.monotonic()
    rospy.loginfo(f"开始时间: {t_start}")
    """主循环"""
    rospy.loginfo("""
    控制说明:
    ---------------------------
    移动控制:
        推动手柄控制XYZ轴移动
        
    按键功能:
        按住按钮0: Unlock Rotation
        按住按钮1: Unlock Z-axis
        C: 开始录制
        S: 停止录制
        E: 紧急停止
        I: 记忆当前位置
        R: 重启控制器
        空格键: 开关夹爪
        ESC: 退出程序
    ---------------------------
    开始控制!
     """)
    stop = False
    memorized_pose = None
    # Start keyboard listener thread
    frequency = 10
    rate = rospy.Rate(frequency)  # 10Hz控制频率
    dt = 1/frequency
    iter_idx = 0
    env.update_current_pose()
    target_pose = env.current_pose
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            Spacemouse(shm_manager=shm_manager, deadzone=0.3) as sm:
            while not rospy.is_shutdown() and not stop:
            # while not stop:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_command_target = t_cycle_end + dt
                time_start = time.time()
                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    # 夹爪
                    if key_stroke == Key.space:
                        print("夹爪开关...")
                        if env.gripper_state:
                            env.gripper_close()
                        else:
                            env.gripper_open()
                    if key_stroke == Key.esc:
                        rospy.loginfo("程序退出...")
                        stop = True
                        os.system("rosnode kill -a")
                        env.end_episode()
                        break
                    if key_stroke == KeyCode(char='e'): #紧急停止
                        # 停止所有ROS Node
                        os.system("rosnode kill -a")
                        env.drop_episode()
                        stop = True
                        break
                    if key_stroke == KeyCode(char='i'):
                        memorized_pose = target_pose
                        rospy.loginfo(f"Current pose: {target_pose} memorized!")
                    if key_stroke == KeyCode(char='r'):
                        pass
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                        os.system("rosnode kill -a")
                        env.end_episode()
                        break
                    elif key_stroke == KeyCode(char='c'):
                        if not env.is_recording:    
                            # Start recording
                            env.start_episode(
                                task_name=task_name,
                                episode_start_time=t_start + (iter_idx + 2) * dt - time.monotonic() + time.time()
                            )
                            key_counter.clear()
                            print('Recording!')
                        else:
                            print('Already recording!')
                    elif key_stroke == KeyCode(char='s'):
                        if env.is_recording:
                            # Stop recording
                            env.end_episode()
                            key_counter.clear()
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        if env.is_recording:
                            # Delete the most recent recorded episode
                            # if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                        else:
                            print('Not recording!')

                if not sm.is_button_pressed(0):
                    drot_xyz[:] = 0
                else:
                    # translation mode
                    dpos[:] = 0

                if not sm.is_button_pressed(1):
                    dpos[2] = 0    
                drot = st.Rotation.from_euler('xyz', drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(
                    target_pose[3:])).as_rotvec()
                rospy.loginfo(f"Target pose: {target_pose}")
                env.exec_actions(
                    actions=[target_pose], 
                    timestamps=[t_command_target-time.monotonic()+time.time()]
                    )
                iter_idx += 1
                rate.sleep()
                time_end = time.time()
                rospy.loginfo(f"cycle time: {time_end - time_start}, actual frequency: {1/(time_end - time_start)}")
                rospy.loginfo(f"t_cycle_end: {t_cycle_end}, t_command_target: {t_command_target}")
if __name__ == '__main__':
    main()