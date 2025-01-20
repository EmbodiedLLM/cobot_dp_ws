#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
import rospy
from multiprocessing.managers import SharedMemoryManager
import cv2
import numpy as np
import torch
import dill
import hydra
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from spacemouse_shared_memory import Spacemouse
from cobot_real_env import CobotEnv
from keystroke_counter import (
    KeystrokeCounter,
)
import torchvision.transforms.functional as TF


OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ckpt_path = "/home/agilex/cobot_diffusion_policy/data/outputs/2025.01.19/18.07.10_train_diffusion_unet_image_cobot_real_image/checkpoints/latest.ckpt"
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    print(f"cfg._target_: {cfg._target_}")
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    shape_meta = cfg.task.shape_meta
    # shape_meta example: {'obs': {'agent_view_image': {'shape': [3, 240, 320], 'type': 'rgb'}, 'hand_view_image': {'shape': [3, 240, 320], 'type': 'rgb'}, 'robot_eef_pose': {'shape': [6], 'type': 'low_dim'}}, 'action': {'shape': [2]}}
    c,h,w = shape_meta['obs']['agent_view_image']['shape']
    target_img_size = (w,h)
    # assert target_img_size == (320, 240), f"target_img_size: {target_img_size}"
    task_name = "cobot_push_cube_eval"

    action_offset = 0
    delta_action = False

    # diffusion model
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.eval().to(device)

    # set inference params
    policy.num_inference_steps = 16 # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    steps_per_inference = policy.n_action_steps


    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    env = CobotEnv(node_name="cobot_real_env_ros1", rosbag_save_root_dir="/home/agilex/cobot_eval_data", is_eval=True)
    # wait for env to be ready
    rospy.loginfo("Waiting for env to be ready...")
    while env.current_observation is None and not rospy.is_shutdown():
        env.get_observation()
        rospy.sleep(0.1)
    rospy.loginfo("Env is ready!")
    obs = env.reset(target_size=target_img_size)
    frequency = 10
    max_duration = 60 # max duration of policy control
    rate = rospy.Rate(frequency)  # 10Hz控制频率
    dt = 1/frequency
    iter_idx = 0
    target_pose = env.current_pose
    env.gripper_state = env.gripper_real_state
    stop = False
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager, deadzone=0.3) as sm:
            while not rospy.is_shutdown() and not stop:
                # Should be the same as demo
                rospy.loginfo("Warming up policy inference")
                with torch.no_grad():
                    policy.reset()
                    obs_dict = env.get_obs(target_size=target_img_size)
                    obs_dict = {k: v.to(device) for k, v in obs_dict.items()}
                    # print all shape
                    for k, v in obs_dict.items():
                        print(f"{k}: {v.shape}")
                    result = policy.predict_action(obs_dict)
                    action = result['action'][0].detach().to('cpu').numpy()
                    assert action.shape[-1] == 2
                    del result

                rospy.loginfo('Ready!')
                rospy.loginfo("""
                    ================ Human in control ==============
                    Robot movement:
                    Move your SpaceMouse to move the robot EEF (locked in xy plane).
                    Press SpaceMouse right button to unlock z axis.
                    Press SpaceMouse left button to enable rotation axes.

                    Recording control:
                    Click the opencv window (make sure it's in focus).
                    Press "C" to start evaluation (hand control over to policy).
                    Press "Q" to exit program.

                    ================ Policy in control ==============
                    Make sure you can hit the robot hardware emergency-stop button quickly! 

                    Recording control:
                    Press "S" to stop evaluation and gain control back.
                    """)
                while True:
                    # ========= human control loop ==========
                    rospy.loginfo("Human in control!")
                    state = env.get_observation()
                    target_pose = state['eef_pose']
                    t_start = time.monotonic()
                    iter_idx = 0
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + 1) * dt
                        t_command_target = t_cycle_end + dt

                        # pump obs
                        obs = env.get_obs()

                        # visualize
                        ## episode_id = env.replay_buffer.n_episodes
                        episode_id = 0
                        vis_img_tensor = obs[f'agent_view_image'][0][-1] # Shape(3, H, W)
                        vis_img = vis_img_tensor.permute(1,2,0).cpu().numpy() # Shape(H, W, 3)
                        vis_img = np.array(TF.to_pil_image(vis_img))
                        # TODO: 适配可选功能, 可视化一个数据集的参考图, 手动调整到和数据集的初始状态匹配
                        # match_episode_id = episode_id
                        # if match_episode is not None:
                        #     match_episode_id = match_episode
                        # if match_episode_id in episode_first_frame_map:
                        #     match_img = episode_first_frame_map[match_episode_id]
                        #     ih, iw, _ = match_img.shape
                        #     oh, ow, _ = vis_img.shape
                        #     tf = get_image_transform(
                        #         input_res=(iw, ih), 
                        #         output_res=(ow, oh), 
                        #         bgr_to_rgb=False)
                        #     match_img = tf(match_img).astype(np.float32) / 255
                        #     vis_img = np.minimum(vis_img, match_img)

                        text = f'Episode: {episode_id}'
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])
                                        # handle key presses
                        enter_policy_control = False
                        key_stroke = cv2.waitKey(1)
                        if key_stroke == ord('e'):
                            env.end_episode()
                            rospy.loginfo("Emergency stop!")
                            import os
                            # Exit program
                            os.system("rosnode kill -a")
                            # kill all python process
                            os.system("killall python3")
                            os.system("killall /usr/bin/python3")
                            stop = True
                            exit(0)
                        elif key_stroke == ord('q'):
                            env.end_episode()
                            stop = True
                            exit(0)
                        elif key_stroke == ord('c'):
                            # Exit human control loop
                            # hand control over to the policy
                            enter_policy_control = True
                            break
                        if enter_policy_control:
                            enter_policy_control = False
                            rospy.loginfo("Exit human control loop, and start policy control now!")
                            break
                        # get teleop command
                        sm_state = sm.get_motion_state_transformed()
                        # print(sm_state)
                        dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                        drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)
    
                        if not sm.is_button_pressed(0):
                            # translation mode
                            drot_xyz[:] = 0
                        else:
                            dpos[:] = 0
                        if not sm.is_button_pressed(1):
                            # 2D translation mode
                            dpos[2] = 0    

                        drot = st.Rotation.from_euler('xyz', drot_xyz)
                        target_pose[:3] += dpos
                        target_pose[3:] = (drot * st.Rotation.from_rotvec(
                            target_pose[3:])).as_rotvec()

                        # TODO: modify to fit our env's clip range 
                        # optinal: clip target pose
                        # target_pose[:2] = np.clip(target_pose[:2], [0.25, -0.45], [0.77, 0.40])
                        rospy.loginfo(f"Target_Time: {t_command_target-time.monotonic()+time.time()}, Current_Time: {time.time()}, target_pose: {target_pose}")
                        # execute teleop command
                        env.exec_actions(
                            actions=[target_pose], 
                            timestamps=[t_command_target-time.monotonic()+time.time()])
                        rate.sleep()
                        iter_idx += 1
                    
                    # ========== policy control loop ==============
                    try:
                        # start episode
                        policy.reset()
                        start_delay = 1.0
                        eval_t_start = time.time() + start_delay
                        t_start = time.monotonic() + start_delay
                        env.start_episode(eval_t_start, task_name=task_name)
                        rospy.loginfo("Policy Control Started!")
                        iter_idx = 0
                        term_area_start_timestamp = float('inf')
                        perv_target_pose = None
                        while True:
                            # calculate timing
                            t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                            # get obs
                            rospy.loginfo('get_obs')
                            obs = env.get_obs()
                            obs_timestamps = env.current_observation['timestamp']
                            rospy.loginfo(f'Obs latency {time.time() - obs_timestamps[-1]}')

                            # run inference
                            rospy.loginfo('run inference')
                            with torch.no_grad():
                                s = time.time()
                                result = policy.predict_action(obs)
                                # this action starts from the first obs step
                                action = result['action'][0].detach().to('cpu').numpy()
                                rospy.loginfo(f"Predicted Action: {action}")
                                rospy.loginfo(f'Inference latency: {time.time() - s}')
                            # target pose (1, 6), action (T=8, 2)
                            assert len(action.shape) == 2 and len(target_pose.shape) == 1, f"action.shape: {action.shape}, target_pose.shape: {target_pose.shape}"
                            this_target_poses = np.zeros((action.shape[0], target_pose.shape[0])) # (T, 6)
                            this_target_poses[:] = target_pose # (6,) -> (T, 6)
                            this_target_poses[:,0] = action[:,0] # (T, 2) -> (T, 6)
                            this_target_poses[:,1] = action[:,1] # (T, 2) -> (T, 6)

                            assert this_target_poses.shape == (action.shape[0], 6), f"this_target_poses.shape: {this_target_poses.shape}, action.shape: {action.shape}"

                            # deal with timing
                            # the same step actions are always the target for
                            action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                                ) * dt + obs_timestamps[-1]
                            action_exec_latency = 0.01
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + action_exec_latency)
                            if np.sum(is_new) == 0:
                                # exceeded time budget, still do something
                                this_target_poses = this_target_poses[[-1]]
                                # schedule on next available step
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + (next_step_idx) * dt
                                rospy.loginfo(f'Over budget, action_timestamp - curr_time: {action_timestamp - curr_time}')
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]

                            # # clip actions
                            # this_target_poses[:,:2] = np.clip(
                            #     this_target_poses[:,:2], [0.25, -0.45], [0.77, 0.40])

                            # execute actions
                            env.exec_actions(
                                actions=this_target_poses,
                                timestamps=action_timestamps
                            )
                            rospy.loginfo(f"Submitted {len(this_target_poses)} steps of actions.")

                            # visualize
                            ## episode_id = env.replay_buffer.n_episodes
                            episode_id = 0
                            vis_img_tensor = obs[f'agent_view_image'][0][-1]
                            vis_img = vis_img_tensor.permute(1,2,0).cpu().numpy()
                            vis_img = np.array(TF.to_pil_image(vis_img))
                            text = 'Episode: {}, Time: {:.1f}'.format(
                                episode_id, time.monotonic() - t_start
                            )
                            cv2.putText(
                                vis_img,
                                text,
                                (10,20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                thickness=1,
                                color=(255,255,255)
                            )
                            cv2.imshow('default', vis_img[...,::-1])
                            # key_stroke = cv2.pollKey()
                            key_stroke = cv2.waitKey(1)
                            if key_stroke == ord('s'):
                                # Stop episode
                                # Hand control back to human
                                env.end_episode()
                                print('Stopped.')
                                break
                            # auto termination
                            terminate = False
                            if time.monotonic() - t_start > max_duration:
                                terminate = True
                                rospy.loginfo('Terminated by the timeout!')

                            # TODO: 标定一个停止区域(起始位置周围), 以便完成任务后自动停止policy
                            # Define target end pose for termination
                            # term_pose = np.array([ 3.40948500e-01,  2.17721816e-01,  4.59076878e-02,  2.22014183e+00, -2.22184883e+00, -4.07186655e-04])
                            # curr_pose = env.current_pose
                            # dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)
                            # if dist < 0.03:
                            #     # in termination area
                            #     curr_timestamp = obs['timestamp'][-1]
                            #     if term_area_start_timestamp > curr_timestamp:
                            #         # First time entering target area, record timestamp
                            #         term_area_start_timestamp = curr_timestamp
                            #     else:
                            #         term_area_time = curr_timestamp - term_area_start_timestamp
                            #         if term_area_time > 0.5: # If stayed for more than 0.5 seconds
                            #             terminate = True
                            #             print('Terminated by the policy!')
                            # else:
                            #     # out of the area
                            #     term_area_start_timestamp = float('inf')

                            if terminate:
                                env.end_episode()
                                stop=True
                                rospy.loginfo('Terminated by the policy!')
                                break

                            rate.sleep()
                            iter_idx += steps_per_inference

                    except KeyboardInterrupt:
                        print("Interrupted!")
                        # stop robot.
                        env.end_episode()

                    
                    print("Stopped.")

# %%
if __name__ == '__main__':
    main()
