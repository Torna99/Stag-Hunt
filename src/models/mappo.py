### Here we define the actor and critic networks used in the MAPPO algorithm.

import torch
import torch.nn as nn

class Actor(nn.Module):
    '''
    Actor network for the Stag Hunt environment. 
    Mainly used for the CTDE algorithm (and for MAPPO).
    '''

    def __init__(self, indim, outdim, hidden_dim=64):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(indim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, outdim)
        )

    def forward(self, x):
        return self.layers(x)

class Critic(nn.Module):
    '''
    Critic network for the Stag Hunt environment. 
    Mainly used for the CTDE algorithm (and for MAPPO).
    '''

    def __init__(self, indim, hidden_dim=64):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(indim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.layers(x)


    