### Note: This code is based on the original paper and the implementation of DQN in the official pytorch tutorial
###
### Here we implement a simple DQN model with 2 hidden layers and ReLU activation function. 
### The input dimension is the state space dimension and the output dimension is the action space dimension.
### The model is trained using the Adam optimizer and the loss function is the mean squared error between 
### the predicted Q-values and the target Q-values.


import torch 
import torch.nn as nn

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


