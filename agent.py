import numpy as np
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils as nn_utils

from model import PolicyNetwork
from model import QNetwork
from utils import ReplayBuffer

from torch.distributions import Normal



writer = SummaryWriter(log_dir=f"runs/SAC_Panda_Lift_{int(time.time())}")

class SAC:
    def __init__(self, env, device, gamma=0.99):
        self.device = device
        self.env = env
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.gamma = gamma

        self.policy = PolicyNetwork(self.ob_dim, self.ac_dim).to(device)
        self.Q1 = QNetwork(self.ob_dim, self.ac_dim).to(device)
        self.Q2 = QNetwork(self.ob_dim, self.ac_dim).to(device)
        self.targetQ1 = QNetwork(self.ob_dim, self.ac_dim).to(device)
        self.targetQ2 = QNetwork(self.ob_dim, self.ac_dim).to(device)
        self.log_alpha = torch.tensor(np.log(0.5), requires_grad=True, device=device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=3e-4)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=3e-4)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-5)

        self.rb = ReplayBuffer(self.ob_dim, self.ac_dim, capacity=1000000, device=self.device)
        self.batch_size = 256
        self.start_rb_size = 50000
        self.target_coef = 0.005
        self.total_step = 0
        # self.max_grad_norm = 0.5
        self.reward_scale = 10.0



    def get_action(self, states):
        states_ = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            mu, std = self.policy(states_)
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)

        log_prob = dist.log_prob(z).sum(dim=-1)
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1)

        return action, log_prob, z      
    

    
    def update(self, total_update_steps):
        episodes = 0
        while 1:
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            episodes += 1
     

            while not done:
                action, log_prob, z = self.get_action(state)
                action_for_env = action.cpu().numpy()
                next_state, reward, terminated, truncated, info = self.env.step(action_for_env)
                reward = reward * self.reward_scale
                done = terminated or truncated

                self.rb.insert(state, action_for_env, reward, next_state, done)
                
                if self.rb.curr_size() > self.start_rb_size:
                    self.train_step()
                
                state = next_state
                total_reward += reward
                self.total_step += 1


                if self.total_step % 10000 == 0:
                    torch.save(self.policy.state_dict(), './checkpoints/model_state_dict_sac_Panda_Lift_4.pth')

                if self.total_step % 1000 == 0:
                    print(f"Episode {episodes}, Total step {self.total_step}, Total Reward: {total_reward / self.reward_scale:.2f}")

            writer.add_scalar("Total_Reward", total_reward / self.reward_scale, self.total_step)

            if self.total_step >= total_update_steps:
                break

        torch.save(self.policy.state_dict(), './checkpoints/model_state_dict_sac_Panda_Lift_4.pth')

        writer.close()

    

    def train_step(self):
        if self.rb.curr_size() < self.batch_size:
            return 

        states, actions, rewards, next_states, dones = self.rb.sample(self.batch_size)


        mu, std = self.policy(states)
        dist = Normal(mu,std)
        z = dist.rsample()
        new_actions = torch.tanh(z)
        new_log_probs = dist.log_prob(z).sum(dim=-1)
        new_log_probs -= torch.sum(torch.log(1 - new_actions.pow(2) + 1e-6), dim=-1)


        with torch.no_grad():
            next_mu, next_std = self.policy(next_states)
            next_dist = Normal(next_mu, next_std)
            next_z = next_dist.rsample()
            next_actions = torch.tanh(next_z)
            next_log_probs = next_dist.log_prob(next_z).sum(dim=-1)
            next_log_probs -= torch.sum(torch.log(1 - next_actions.pow(2) + 1e-6), dim=-1)

            alpha = self.log_alpha.exp()

            value = torch.min(self.targetQ1(next_states, next_actions), self.targetQ2(next_states, next_actions)) - alpha * next_log_probs

            Q_update_target = rewards.squeeze(-1) + self.gamma * (1.0 - dones.squeeze(-1)) * value



        target_entropy = -self.ac_dim
        alpha_loss = -(self.log_alpha * (new_log_probs.detach() + target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()


        q1 = self.Q1(states, actions)        
        q1_loss = nn.MSELoss()(q1, Q_update_target)

        q2 = self.Q2(states, actions)
        q2_loss = nn.MSELoss()(q2, Q_update_target)

        self.Q1_optimizer.zero_grad()
        q1_loss.backward()
        self.Q1_optimizer.step()

        self.Q2_optimizer.zero_grad()
        q2_loss.backward()
        self.Q2_optimizer.step()




        Q_1 = self.Q1(states, new_actions)
        Q_2 = self.Q2(states, new_actions)
        min_q = torch.min(Q_1, Q_2)
        policy_loss = (alpha * new_log_probs - min_q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        
        self.soft_update(self.targetQ1, self.Q1, self.target_coef)
        self.soft_update(self.targetQ2, self.Q2, self.target_coef)


        policy_entropy = dist.entropy().mean()

        if self.total_step % 1000 == 0:
            print(self.total_step)
            print(f"policy_entropy {policy_entropy.item()}, alpha: {alpha.item()}")
            print(f"policy loss {policy_loss.item()}, q1 loss: {q1_loss.item()}, new log probs: {new_log_probs.mean().item()}")



    def soft_update(self, target_net, source_net, tau):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)




