# import os
# import torch
# import imageio
# import numpy as np
# from robosuite import make
# from robosuite.wrappers import GymWrapper
# from tqdm import trange
# from model import PolicyNetwork 

# def evaluate_and_record(episodes=5, save_path="video.mp4"):
#     env = make(
#         env_name="Lift",
#         robots="Panda",
#         use_camera_obs=False,  # 학습과 동일
#         has_renderer=False,    # 헤드리스이므로 화면 없음
#         has_offscreen_renderer=True,
#         render_camera="agentview",
#         control_freq=20,
#         reward_shaping=True,
#     )

#     env = GymWrapper(env, render_obs=True)  # 이게 핵심

#     ob_dim = env.observation_space.shape[0]
#     ac_dim = env.action_space.shape[0]

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     policy = PolicyNetwork(ob_dim, ac_dim).to(device)
#     policy.load_state_dict(torch.load("./checkpoints/model_state_dict_sac_Panda_Lift_3.pth", map_location=device))
#     policy.eval()


#     for ep in range(episodes):
#         obs = env.reset()
#         done = False
#         episode_return = 0
#         frames = []

#         while not done:
#             # 상태 벡터만 뽑기 (이미지는 뒤에 붙음)
#             obs_vec = obs["observation"][:- (width * height * 3)]
#             obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)

#             with torch.no_grad():
#                 mu, _ = policy(obs_tensor)
#             action = torch.tanh(mu).cpu().numpy()[0]
#             obs, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             # 이미지 추출
#             frame = obs["observation"][-(width * height * 3):]
#             frame = frame.reshape(height, width, 3)
#             frames.append(frame)

#             episode_return += reward

#         print(f"Episode {ep+1} return: {episode_return:.2f}")

#     # 영상 저장
#     imageio.mimsave(save_path, frames, fps=20)
#     print(f"Saved video to {save_path}")


# if __name__ == "__main__":
#     evaluate_and_record(episodes=5)


import imageio
import torch
import numpy as np
from robosuite import make
from robosuite.wrappers import GymWrapper
from model import PolicyNetwork
import os

os.environ["MUJOCO_GL"] = "egl"

def evaluate_and_record(env_name="Lift", episodes=3, filename="Panda_Lift_3.mp4", max_episode_steps=1000):
    # 1. 환경 생성
    raw_env = make(
        env_name=env_name,
        robots="Panda",
        use_camera_obs=False,  
        has_renderer=False,     
        has_offscreen_renderer=True,  
        reward_shaping=True,
        control_freq=20,
    )
    env = GymWrapper(raw_env)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNetwork(ob_dim, ac_dim).to(device)
    policy.load_state_dict(torch.load("./checkpoints/model_state_dict_sac_Panda_Lift_3.pth", map_location=device))
    policy.eval()


    frames = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done and step < max_episode_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = policy(obs_tensor)
            action = torch.tanh(mu).cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

   
            frame = raw_env.sim.render(width=640, height=480, camera_name="frontview")
            frame = np.flipud(frame)
            frames.append(frame)

        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")
    imageio.mimsave("./result_videos/"+filename, frames, fps=20)
    print(f"Saved video to {filename}")

if __name__ == "__main__":
    evaluate_and_record()