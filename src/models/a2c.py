# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optimizer
# import numpy as np


# class Actor(nn.Module):

#     def __init__(self, indim, outdim, hidden_dim=32):
#         super(Actor, self).__init__()

#         self.layers = nn.Sequential(
#             nn.Linear(indim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim * 2, outdim), # out: logits to convert into prob distribution later
#         )

#     def forward(self, states):
#         # pi = F.softmax(self.layers(states), dim=-1)  # convert logits to prob distribution
#         # return pi
#         logits = self.layers(states)
#         return logits

# class Critic(nn.Module):

#     def __init__(self, indim, outdim=1, hidden_dim=32):
#         super(Critic, self).__init__()

#         self.layers = nn.Sequential(
#             nn.Linear(indim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim * 2, outdim),
#             # nn.ReLU(), <- no relu or we'll not have negative values (e.g. punishment)
#         )

#     def forward(self, states):
#         value = self.layers(states)
#         return value

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

