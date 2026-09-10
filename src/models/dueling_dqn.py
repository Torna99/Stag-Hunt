### Here the is the implementation of the Dueling DQN model, which is an extension of the DQN model. 
### This architecture in runned with the Double DQN algorithm seen previously.

import torch 
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
import numpy as np


# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

class DuelingDQN(nn.Module):
    """
    A Dueling DQN model.
    """

    def __init__(self, indim, outdim, hidden_dim=32):
        super(DuelingDQN, self).__init__()

        self.feature_net = nn.Sequential(
            nn.Linear(indim, hidden_dim),
            nn.ReLU()
        )


        self.value_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.advantage_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, outdim)
        )

    def forward(self, x):

        features = self.feature_net(x)

        V_s = self.value_layers(features)

        A_s_a = self.advantage_layers(features)

        Q_s_a = V_s + (A_s_a - A_s_a.mean(dim=1, keepdim=True))

        return Q_s_a


