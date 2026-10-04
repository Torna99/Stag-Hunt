### Here we implement the A2C network architecture, which consists of a shared feature extractor
### followed by separate actor and critic. The actor outputs action logits, 
### while the critic outputs state value V(s) (scalar value).
### This architecture allows the agent to learn both policy and value functions simultaneously. 

import torch.nn as nn

class A2CNet(nn.Module):

    def __init__(self, indim, outdim, hidden_dim=32):
        super(A2CNet, self).__init__()

        self.common_layers = nn.Sequential(
            nn.Linear(indim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        self.actor_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, outdim), ## return logits for each action
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), # return a scalar which is the value of the state V(s)
        )

    def forward(self, states):
        feat = self.common_layers(states)
        logits = self.actor_layers(feat)
        value = self.critic_layers(feat)
        return logits, value

