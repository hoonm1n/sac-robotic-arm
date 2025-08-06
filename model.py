import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_output = nn.Linear(256, action_dim)
        self.log_std_output = nn.Linear(256, action_dim)

        # nn.init.constant_(self.log_std_output.weight, 0.0)
        # nn.init.constant_(self.log_std_output.bias, 1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_output(x)
        log_std = self.log_std_output(x)
        log_std = torch.clamp(log_std, min=-2.0, max=2.0)
        std = torch.exp(log_std)
        return mu, std



class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.out(x)
        return q_value.squeeze(-1)