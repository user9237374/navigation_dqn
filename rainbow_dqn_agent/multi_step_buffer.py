import numpy as np
import torch
from collections import deque
from typing import Tuple


class NStepBuffer:
    """N-Step Buffer for multi-step learning in Rainbow DQN."""
    
    def __init__(self, n_step: int = 3, gamma: float = 0.99):
        """
        Initialize N-Step Buffer.
        
        Args:
            n_step: Number of steps for n-step returns
            gamma: Discount factor
        """
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)
    
    def add(self, state, action, reward, next_state, done):
        """Add transition to n-step buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def get_n_step_info(self) -> Tuple:
        """
        Calculate n-step return and get the resulting transition.
        
        Returns:
            (state, action, n_step_return, n_step_next_state, n_step_done)
        """
        if len(self.buffer) < self.n_step:
            return None
            
        # Get the first transition
        state, action = self.buffer[0][:2]
        
        # Calculate n-step return
        n_step_return = 0
        n_step_done = False
        
        for i in range(self.n_step):
            _, _, reward, next_state, done = self.buffer[i]
            n_step_return += (self.gamma ** i) * reward
            
            if done:
                n_step_done = True
                n_step_next_state = next_state
                break
            
            n_step_next_state = next_state
        
        return (state, action, n_step_return, n_step_next_state, n_step_done)
    
    def __len__(self):
        return len(self.buffer)
    
    def is_full(self):
        return len(self.buffer) == self.n_step


class MultiStepPrioritizedReplayBuffer:
    """Prioritized Replay Buffer with N-Step Learning support."""
    
    def __init__(self, buffer_size: int, batch_size: int, n_step: int = 3, gamma: float = 0.99, 
                 alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize Multi-Step Prioritized Replay Buffer.
        
        Args:
            buffer_size: Maximum size of buffer
            batch_size: Size of each training batch
            n_step: Number of steps for n-step returns
            gamma: Discount factor
            alpha: How much prioritization is used
            beta: Importance sampling correction factor
            beta_increment: How much to increment beta each step
        """
        # Import here to avoid circular imports
        from rainbow_dqn_agent.prioritized_replay_buffer import PrioritizedReplayBuffer
        
        self.n_step = n_step
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Create n-step buffer and main prioritized buffer
        self.n_step_buffer = NStepBuffer(n_step, gamma)
        self.memory = PrioritizedReplayBuffer(buffer_size, batch_size, alpha, beta, beta_increment)
        
        # Store 1-step transitions for immediate learning
        self.use_n_step = True
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer with n-step computation."""
        # Add to n-step buffer
        self.n_step_buffer.add(state, action, reward, next_state, done)
        
        # If n-step buffer is full, add the n-step transition to main buffer
        if self.n_step_buffer.is_full():
            n_step_transition = self.n_step_buffer.get_n_step_info()
            if n_step_transition is not None:
                self.memory.add(n_step_transition)
        
        # If episode is done, flush remaining transitions
        if done:
            # Process remaining transitions in buffer
            for _ in range(len(self.n_step_buffer) - 1):
                # Remove first element and process remaining partial transitions
                self.n_step_buffer.buffer.popleft()
                if len(self.n_step_buffer) > 0:
                    n_step_transition = self.n_step_buffer.get_n_step_info()
                    if n_step_transition is not None:
                        self.memory.add(n_step_transition)
            # Clear the buffer completely
            self.n_step_buffer.buffer.clear()
    
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """Sample batch of n-step experiences."""
        return self.memory.sample()
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities of sampled experiences."""
        self.memory.update_priorities(indices, priorities)
    
    def __len__(self):
        return len(self.memory)