"""
Rainbow DQN Neural Network Models

AI Assistance Disclaimer:
This Rainbow DQN model implementation was developed with assistance from AI tools.
Includes Dueling architecture and Noisy Networks for exploration.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)




class RainbowDQN(nn.Module):
    def __init__(self, state_dim, action_dim, atom_size=51, v_min=-10, v_max=10):
        super(RainbowDQN, self).__init__()
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (atom_size - 1)
        # Register support as buffer so it moves with the model to correct device
        self.register_buffer('support', torch.linspace(v_min, v_max, atom_size))

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        self.value_fc = NoisyLinear(128, atom_size)
        self.advantage_fc = NoisyLinear(128, action_dim * atom_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value_fc(x).view(batch_size, 1, self.atom_size)
        advantage = self.advantage_fc(x).view(batch_size, self.action_dim, self.atom_size)

        q_atoms = value + advantage - advantage.mean(1, keepdim=True)
        probs = F.softmax(q_atoms, dim=2)
        return probs

    def reset_noise(self):
        self.value_fc.reset_noise()
        self.advantage_fc.reset_noise()