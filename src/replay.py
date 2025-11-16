import random
import numpy as np
from collections import deque
import torch


class ReplayBuffer:
    """Simple replay buffer using deque.

    Stores tuples: (state, action, reward, next_state, done)
    """

    def __init__(self, capacity=100000, device="cpu"):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.position = 0
        self.device = device

    def push(self, state, action, reward, next_state, done):
        # Store states as uint8 to save 4x memory (255 instead of float32)
        state_uint8 = (np.array(state) * 255).astype(np.uint8)
        next_state_uint8 = (np.array(next_state) * 255).astype(np.uint8)
        self.buffer.append((state_uint8, action, reward, next_state_uint8, done))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=32):
        transitions = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*transitions)

        # Convert back to float32 and normalize to [0, 1]
        state = torch.FloatTensor(np.array(s) / 255.0).to(self.device)
        action = torch.LongTensor(a).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)
        next_state = torch.FloatTensor(np.array(s2) / 255.0).to(self.device)
        done = torch.FloatTensor(d).to(self.device)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
