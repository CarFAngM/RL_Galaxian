import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network with shared convolutional layers.
    
    Actor outputs policy probabilities for each action.
    Critic outputs state value V(s).
    """

    def __init__(self, input_shape, n_actions):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        conv_out_size = self._get_conv_output(input_shape)

        # Shared hidden layer
        self.fc_shared = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )
        
        # Actor head: outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        
        # Critic head: outputs state value
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _get_conv_output(self, shape):
        """Calculate the flattened conv output size."""
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            out = self.conv(dummy)
            return int(np.prod(out.size()))

    def forward(self, x):
        """Forward pass returns both policy logits and state value."""
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc_shared(x)
        
        # Actor: policy logits (apply softmax for probabilities)
        policy_logits = self.actor(x)
        
        # Critic: state value
        state_value = self.critic(x)
        
        return policy_logits, state_value
    
    def get_action(self, state, training=True):
        """Sample action from policy distribution."""
        policy_logits, state_value = self.forward(state)
        
        if training:
            # Sample from categorical distribution
            probs = F.softmax(policy_logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            return action, log_prob, state_value
        else:
            # Greedy: take action with highest probability
            action = policy_logits.argmax(dim=-1)
            return action, None, state_value
