import numpy as np
from collections import deque
import random
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device="cpu", dtype=np.float32):
        self.capacity = capacity
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = np.zeros((capacity, state_dim), dtype=dtype)
        self.next_states = np.zeros((capacity, state_dim), dtype=dtype)
        self.actions = np.zeros((capacity, action_dim), dtype=dtype)
        self.rewards = np.zeros((capacity,), dtype=dtype)
        self.dones = np.zeros((capacity,), dtype=dtype)

        self.idx = 0
        self.size = 0

    def insert(self, state, action, reward, next_state, done):
        self.states[self.idx] = np.copy(state)
        self.actions[self.idx] = np.copy(action)
        self.rewards[self.idx] = np.copy(reward)
        self.next_states[self.idx] = np.copy(next_state)
        self.dones[self.idx] = np.copy(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[indices]).to(self.device).float()
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device).float()
        actions = torch.from_numpy(self.actions[indices]).to(self.device).float()
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device).float().unsqueeze(1)
        dones = torch.from_numpy(self.dones[indices]).to(self.device).float().unsqueeze(1)


        return states, actions, rewards, next_states, dones

    def curr_size(self):
        return self.size




