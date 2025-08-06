import torch
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from model import PolicyNetwork  

def evaluate_policy(episodes=10, render=False, seed=123):

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,        
        use_camera_obs=False,      
        reward_shaping=True,
        control_freq=20,
    )
    env = GymWrapper(env)

    # env.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNetwork(ob_dim, ac_dim).to(device)
    policy.load_state_dict(torch.load("./checkpoints/model_state_dict_sac_Panda_Lift_3.pth", map_location=device))
    policy.eval()

    returns = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = policy(obs_tensor)
            action = torch.tanh(mu).cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if render:
                env.render()

        returns.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")

    print(f"\nAverage Reward over {episodes} episodes: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    env.close()
    # return returns


if __name__ == "__main__":
    evaluate_policy(
        episodes=10,
        render=False 
    )