import robosuite as suite
from robosuite.wrappers import GymWrapper
import gym
import gymnasium as gym
import numpy as np
import torch
from agent import SAC


def main():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,        
        use_camera_obs=False,      
        reward_shaping=True,
        control_freq=20,
    )
    env = GymWrapper(env)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SAC(env, device, gamma=0.99)

    agent.update(total_update_steps=3000000)

    env.close()

if __name__ == "__main__":
    main()