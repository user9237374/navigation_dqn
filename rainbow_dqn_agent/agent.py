"""
Rainbow DQN Agent Implementation

AI Assistance Disclaimer:
This Rainbow DQN implementation was developed with assistance from AI tools.
The implementation includes all 6 components of the Rainbow algorithm:
1. Double DQN, 2. Prioritized Experience Replay, 3. Dueling Networks,
4. Multi-step Learning, 5. Distributional RL, 6. Noisy Networks
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from rainbow_dqn_agent.model import RainbowDQN
from rainbow_dqn_agent.multi_step_buffer import MultiStepPrioritizedReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
TAU = 1e-3              # for soft update of target parameters

class RainbowDqnAgent:
    """
    Rainbow DQN Agent implementing all 6 extensions from the Rainbow paper.
    
    Note: This implementation was developed with AI assistance.
    """
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim
        self.atom_size = 51
        self.v_min = -10
        self.v_max = 10
        self.delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)

        self.online_net = RainbowDQN(state_dim, action_dim, self.atom_size, self.v_min, self.v_max).to(device)
        self.target_net = RainbowDQN(state_dim, action_dim, self.atom_size, self.v_min, self.v_max).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        
        self.support = self.online_net.support

        self.gamma = GAMMA
        self.n_steps = 3  # multi-step learning
        self.buffer = MultiStepPrioritizedReplayBuffer(
            buffer_size=BUFFER_SIZE, 
            batch_size=BATCH_SIZE, 
            n_step=self.n_steps, 
            gamma=GAMMA
        )
        self.t_step = 0  # Initialize step counter
    
    def network(self):
        return self.online_net

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.online_net(state)
            dist = dist * self.support
            q_values = dist.sum(2)
            action = q_values.argmax(1).item()
        return action

    def step(self, state, action, reward, next_state, done):
        # Add input validation
        if not isinstance(action, (int, np.integer)):
            action = int(action)
        self.buffer.add(state, action, reward, next_state, done)
        
        # Update step counter
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        if len(self.buffer) < self.buffer.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample()
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # Distributional Bellman target
        with torch.no_grad():
            # Double DQN: Use online network for action selection
            online_next_dist = self.online_net(next_states)
            online_next_q = (online_next_dist * self.support).sum(2)
            next_actions = online_next_q.argmax(1)
            
            # Use target network for Q-value estimation
            target_next_dist = self.target_net(next_states)
            next_dist = target_next_dist[range(target_next_dist.size(0)), next_actions]

            Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (self.gamma ** self.n_steps) * self.support.unsqueeze(0)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            m = torch.zeros_like(next_dist)
            
            # Clamp indices to valid range
            l_clamped = l.clamp(0, self.atom_size - 1)
            u_clamped = u.clamp(0, self.atom_size - 1)
            
            # lower bound projection
            lower_weights = next_dist * (u.float() - b)
            m.scatter_add_(1, l_clamped, lower_weights)
            
            # upper bound projection  
            upper_weights = next_dist * (b - l.float())
            m.scatter_add_(1, u_clamped, upper_weights)

        dist = self.online_net(states)
        log_p = torch.log(dist[range(dist.size(0)), actions])
        elementwise_loss = -(m * log_p).sum(1)
        
        # importance sampling weights
        loss = (weights * elementwise_loss).mean()
        
        # TD errors for priority updates (n-step)
        with torch.no_grad():
            current_dist = self.online_net(states)
            current_q = (current_dist * self.support).sum(2)
            current_q_values = current_q[range(current_q.size(0)), actions]
            
            target_dist = self.target_net(next_states)
            target_q = (target_dist * self.support).sum(2)
            target_q_values = target_q.max(1)[0]
            
            expected_q_values = rewards + (self.gamma ** self.n_steps) * target_q_values * (1 - dones)
            td_errors = abs(current_q_values - expected_q_values).cpu().numpy()

        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        # Update priorities in replay buffer
        self.buffer.update_priorities(indices, td_errors)
        
        # Update target network
        if self.t_step == 0:
            self.soft_update(self.online_net, self.target_net, TAU)
