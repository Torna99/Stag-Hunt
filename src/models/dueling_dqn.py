### Note: This code is based on the paper "Dueling Network Architectures for Deep Reinforcement Learning" by Ziyu Wang et al. 
###
### Here the is the implementation of the Dueling DQN model, which is an extension of the DQN model. 
### This architecture is runned with the Double DQN algorithm seen previously.

import torch 
import torch.nn as nn


class DuelingDQN(nn.Module):

    def __init__(self, indim, outdim, hidden_dim=32):
        super(DuelingDQN, self).__init__()

        ## Here we have the commmon layers of the architecture, 
        ## shared between the value and advantage streams.
        self.feature_net = nn.Sequential(
            nn.Linear(indim, hidden_dim),
            nn.ReLU()
        )

        ## Here we have the value stream, which outputs a the 
        ## value function estimation for each state.
        self.value_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        ## Here we have the advantage stream, which outputs the
        ## advantage function estimation for each action in a given state.
        self.advantage_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, outdim)
        )

    def forward(self, x):
        """
        Forward pass of the Dueling DQN model.
        Obtains the Q-values for each action in a given state:
            Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        Because A(s,a) = Q(s,a) - V(s).
        The mean is subtracted to ensure that the advantage function has zero mean,
        which helps to stabilize training.
        """
        
        features = self.feature_net(x)

        V_s = self.value_layers(features)

        A_s_a = self.advantage_layers(features)

        ## Compute the Q-values by combining the value and advantage streams.
        Q_s_a = V_s + (A_s_a - A_s_a.mean(dim=1, keepdim=True))

        return Q_s_a


