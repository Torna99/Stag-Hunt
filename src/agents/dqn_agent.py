### Note: This code is based on the original paper and the implementation of DQN in the official pytorch tutorial
###
### In this code we implement a standard DQN agent. 

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer

from buffers.replay import ReplayMemory, Sample
from models.dqn import DQN

import random
import numpy as np

class DQNAgent:

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1, buffer_size = 1e4, batch_size=32, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        ## initialize hyperparameters: gamma is the discount factor
        self.lr = lr
        self.gamma = gamma

        ## epsilon is the exploration for epsilon-greedy policy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        ## instatiate the replay memory
        self.rep_memory = ReplayMemory(int(buffer_size))

        ## instatiate the policy and target networks
        self.policy_net = DQN(state_dim, action_dim)
        # self.target_net = DQN(state_dim, action_dim)
        
        ## instatiate the optimizer (use Adam inseatd of SGD for better performance) and the loss function (MSE)
        self.optimizer = optimizer.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        ## BOH?
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.target_net.eval()

    def select_action(self, state):
        '''
        Select an action based on the current state using epsilon-greedy policy.
        '''
        sample = random.random()

        ## Exploration
        if sample < self.epsilon:
            return torch.tensor([[random.randrange(self.action_dim)]], dtype=torch.long)

        ## Exploitation
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(state).max(1).indices.view(1, 1)

    def epsilon_decay_step(self):
        '''
        Decay the epsilon value.
        '''
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            #self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def store_sample(self, state, action, reward, next_state, done):
        self.rep_memory.push(state, action, reward, next_state, done)

    def optimize_model(self):
        '''
        This function is the single train step of the DQN agent. 
        It samples a batch of experiences from the replay memory, computes the loss, and updates the policy network.
        '''

        # if rep memory is not filled, return
        if len(self.rep_memory) < self.batch_size:
            return 

        # sampling 
        samples = self.rep_memory.samples_batch(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        # Convert the samples to tensors
        states = torch.cat(states, dim=0).to(self.device).float()
        next_states = torch.cat(next_states, dim=0).to(self.device).float()
        actions = torch.tensor(
            [a.item() if isinstance(a, torch.Tensor) else int(a) for a in actions],
            dtype=torch.long,
            device=self.device
        ).view(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)
        
        # compute Q(s_t, a)
        state_action_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            # get the q values for the next states
            next_state_values = self.policy_net(next_states).max(1)[0].unsqueeze(1)
            #best_next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)

            # compute the expected Q values (target q for the loss function)
            # use (1 - dones) to handle the terminal states
            y_j = rewards + (self.gamma * next_state_values * (1 - dones))

        # compute the loss
        loss = self.loss(state_action_values, y_j)

        # optimize the model 
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

