# %% 载入训练的policy
import dill
import torch
import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ckpt_path = "/home/agilex/cobot_diffusion_policy/data/outputs/2025.01.19/18.07.10_train_diffusion_unet_image_cobot_real_image/checkpoints/latest.ckpt"
payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
cfg = payload['cfg']
with hydra.initialize('./diffusion_policy/config'):
    # cfg = hydra.compose('train_diffusion_unet_cobot_real_image_workspace')
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model
    policy.to(device)
   
# %% 获取当前图像
obs_dict = {
    "agent_view_image": torch.randn(1,2, 3, 240, 320).to(device),
    "hand_view_image": torch.randn(1,2, 3, 240, 320).to(device),
    "robot_eef_pose": torch.randn(1,2, 6).to(device),
}

predicted_action = policy.predict_action(obs_dict)
print(f"predicted_action: {predicted_action}")

# %%
from cobot_real_env import CobotEnv
env = CobotEnv("cobot_env", is_eval=True)
# %%
obs_dict = env.reset(device=None)
obs = env.get_observation()
# %%
# print obs_dict all shape
for key, value in obs_dict.items():
    print(f"{key}: {value.shape}")
# %%
import cv2
import numpy as np
import torchvision.transforms.functional as TF
episode_id = 0
vis_img_tensor = obs_dict[f'agent_view_image'][0][-1] # Shape(3, H, W)
vis_img = vis_img_tensor.permute(1,2,0).cpu().numpy() # Shape(H, W, 3)
vis_img = np.array(TF.to_pil_image(vis_img))
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


# %%
import cv2
from collections import deque
main_cam = obs['main_color']
agent_view_image = cv2.resize(main_cam, (320, 240))
agent_view_image = torch.from_numpy(agent_view_image).permute(2, 0, 1).unsqueeze(0).to(device)

hand_view_image = cv2.resize(obs['hand_color'], (320, 240))
hand_view_image = torch.from_numpy(hand_view_image).permute(2, 0, 1).unsqueeze(0).to(device)
robot_eef_pose = torch.from_numpy(obs['eef_pose']).unsqueeze(0).to(device)  # Add unsqueeze to match the required shape
# %%plot 
import matplotlib.pyplot as plt
plt.title("agent_view_image")
agent_view_image_rgb = cv2.cvtColor(agent_view_image.squeeze(0).permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)  
plt.imshow(agent_view_image_rgb)
plt.show()
plt.title("hand_view_image")
hand_view_image_rgb = cv2.cvtColor(hand_view_image.squeeze(0).permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)
plt.imshow(hand_view_image_rgb)
plt.show()
#%%

obs_que = deque(maxlen=2)
single_step_obs_dict = {
    "agent_view_image": agent_view_image,
    "hand_view_image": hand_view_image,
    "robot_eef_pose": robot_eef_pose,
}
obs_que.append(single_step_obs_dict)
obs_que.append(single_step_obs_dict)
# image: (B, T, C, H, W)
# lowdiw: (B,T,D)
obs_dict = {
    "agent_view_image": torch.stack([obs_que[0]["agent_view_image"], obs_que[1]["agent_view_image"]], dim=1),
    "hand_view_image": torch.stack([obs_que[0]["hand_view_image"], obs_que[1]["hand_view_image"]], dim=1),
    "robot_eef_pose": torch.stack([obs_que[0]["robot_eef_pose"], obs_que[1]["robot_eef_pose"]], dim=1),
}
# print all shapes
for key, value in obs_dict.items():
    print(f"{key}: {value.shape}")

predicted_action = policy.predict_action(obs_dict)
print(f"predicted_action: {predicted_action}")


# %%
import time
current_pose = obs_que[0]["robot_eef_pose"]
print(f"current_pose: {current_pose}")
target_pose = current_pose.clone()
target_pose[0][0] = predicted_action['action'][0][7][0]
target_pose[0][1] = predicted_action['action'][0][7][1]
print(f"target_pose: {target_pose}")
env.exec_actions(target_pose.detach().cpu().numpy(), [time.time()+0.1], stage=0)
# %%
