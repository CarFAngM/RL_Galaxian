import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .actor_critic import ActorCriticNetwork


class ActorCriticAgent:
    """Actor-Critic agent for Atari games.
    
    Uses advantage actor-critic (A2C) with entropy regularization.
    """
    
    def __init__(self, state_shape, n_actions, lr=3e-4, gamma=0.99, 
                 entropy_coef=0.05, value_loss_coef=0.5, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        
        # Actor-Critic network
        self.ac_net = ActorCriticNetwork(state_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)
        
        # For storing episode trajectory
        self.reset_trajectory()
        
    def reset_trajectory(self):
        """Reset trajectory buffers for new episode."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def select_action(self, state, training=True):
        """Select action using current policy."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad() if not training else torch.enable_grad():
            if training:
                action, log_prob, value = self.ac_net.get_action(state_t, training=True)
                
                # Store for training
                self.states.append(state)
                self.actions.append(action.item())
                self.log_probs.append(log_prob)
                self.values.append(value)
                
                return action.item()
            else:
                action, _, _ = self.ac_net.get_action(state_t, training=False)
                return action.item()
    
    def store_reward_done(self, reward, done):
        """Store reward and done flag for current step."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def train_step(self):
        """Perform A2C update after episode ends."""
        if len(self.rewards) == 0:
            return None, None, None
        
        # Calculate returns and advantages
        returns = []
        R = 0
        
        # Calculate discounted returns backward (usando rewards NO clipeadas internamente)
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # NO normalizar returns - queremos que el agente aprenda valores absolutos
        # Normalizar puede hacer que pierda la noción de qué estados son realmente mejores
        
        # Stack stored values
        log_probs = torch.stack(self.log_probs)
        values = torch.cat(self.values).squeeze()  # Remove extra dimension
        
        # Calculate advantages: A(s,a) = R - V(s)
        advantages = returns - values.detach()
        
        # Normalizar SOLO las advantages (no los returns)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor loss (policy gradient with advantage)
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic loss (MSE between predicted value and return)
        critic_loss = nn.MSELoss()(values, returns)
        
        # Entropy bonus (encourage exploration)
        # Recalculate probabilities for entropy
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        policy_logits, _ = self.ac_net(states_tensor)
        probs = torch.softmax(policy_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # Total loss
        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)  # Más agresivo
        self.optimizer.step()
        
        # Reset trajectory
        self.reset_trajectory()
        
        return actor_loss.item(), critic_loss.item(), entropy.item()
    
    def save(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'ac_net': self.ac_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.ac_net.load_state_dict(checkpoint['ac_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
