### Enrico Tornabene (0001191764)
###
### Note: This code is based on the implementation of DQN in the official pytorch tutorial 
###


import torch 
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

class DQN(nn.Module):

    def __init__(self, indim, outdim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(indim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, outdim)
        )

    def forward(self, x):
        return self.fc(x)  


