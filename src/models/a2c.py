import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import numpy as np


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
            nn.Linear(hidden_dim // 2, outdim),
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

