import numpy as np
import torch
import random
from typing import Tuple


class SumTree:
    """Sum Tree data structure for efficient sampling in Prioritized Experience Replay."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float):
        """Retrieve sample index based on priority sum."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return total priority sum."""
        return self.tree[0]

    def add(self, p: float, data):
        """Add new sample with priority p."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float):
        """Update priority of sample at idx."""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        """Get sample based on priority sum s."""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer."""
    
    def __init__(self, buffer_size: int, batch_size: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize Prioritized Experience Replay Buffer.
        
        Args:
            buffer_size: Maximum size of buffer
            batch_size: Size of each training batch
            alpha: How much prioritization is used (0 = uniform, 1 = full prioritization)
            beta: Importance sampling correction factor (0 = no correction, 1 = full correction)
            beta_increment: How much to increment beta each step
        """
        self.tree = SumTree(buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6 
        self.max_priority = 1.0

    def add(self, experience):
        """Add experience to buffer with maximum priority."""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """
        Sample batch of experiences with importance sampling weights.
        
        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / self.batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(self.batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / (self.tree.total() + 1e-8)  
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        max_weight = weights.max()
        if max_weight > 0:
            weights /= max_weight  
        else:
            weights = np.ones_like(weights)  

        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
            np.array(indices),
            torch.FloatTensor(weights)
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities of sampled experiences."""
        for idx, priority in zip(indices, priorities):
            priority = abs(priority) + self.epsilon
            clipped_priority = min(priority, self.max_priority)
            self.tree.update(idx, clipped_priority ** self.alpha)

    def __len__(self):
        """Return current size of buffer."""
        return self.tree.n_entries