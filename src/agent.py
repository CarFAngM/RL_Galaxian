# src/agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dqn import DQN
from .replay import ReplayBuffer


class DQNAgent:
    def __init__(self, state_shape, n_actions, lr=2.5e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.99,
                 replay_capacity=100000, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = DQN(state_shape, n_actions).to(self.device)
        self.target_net = DQN(state_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=replay_capacity, device=self.device)

        self.train_counter = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state_t)
            return int(q.max(1)[1].item())

    def train_step(self, batch_size=32):
        """Double DQN training step."""
        if len(self.memory) < batch_size:
            return None

        self.train_counter += 1
        if self.train_counter % 4 != 0:
            return None

        state, action, reward, next_state, done = self.memory.sample(batch_size)
        reward = torch.clamp(reward, -1, 1)

        current_q = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_state).max(1)[1]
            next_q = self.target_net(next_state).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = reward + (1 - done) * self.gamma * next_q

        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path, save_buffer=False, max_buffer_size=10000):
        """Save model with optional buffer compression"""
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory_capacity': self.memory.capacity
        }
        
        # Solo guardar buffer si se solicita y comprimido
        if save_buffer and len(self.memory.buffer) > 0:
            buffer_size = min(len(self.memory.buffer), max_buffer_size)
            # Tomar las experiencias más recientes
            recent_buffer = list(self.memory.buffer)[-buffer_size:]
            checkpoint['memory'] = recent_buffer
            checkpoint['memory_size'] = len(self.memory.buffer)
            print(f" Guardando {buffer_size} experiencias recientes de {len(self.memory.buffer)} totales")
        
        torch.save(checkpoint, path)

    def load(self, path, load_buffer=False):
        """Load model, optionally with buffer"""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        
        # Cargar buffer solo si se solicita
        if load_buffer and 'memory' in checkpoint:
            self.memory.buffer = checkpoint['memory']
            self.memory.position = len(self.memory.buffer) % self.memory.capacity
            original_size = checkpoint.get('memory_size', len(self.memory.buffer))
            print(f" Buffer cargado: {len(self.memory.buffer)} experiencias (original: {original_size})")
            return True
        
        return False